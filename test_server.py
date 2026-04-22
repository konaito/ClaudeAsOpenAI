"""
server.py のテスト。

実行:
    uv run test_server.py
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "claude-agent-sdk",
#     "anyio",
#     "pytest",
#     "pytest-asyncio",
#     "httpx",
# ]
# ///

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import server
from server import (
    ChatCompletionRequest,
    ChatMessage,
    ClaudeProxyError,
    _compact_exception_text,
    _convert_image_url_to_claude,
    _is_retryable_claude_error,
    _make_prompt,
    _sse_chunk,
    _to_claude_proxy_error,
    app,
    call_claude,
    convert_messages,
    stream_claude,
)


# ---------------------------------------------------------------------------
# convert_messages
# ---------------------------------------------------------------------------


def test_convert_messages_basic():
    msgs = [
        ChatMessage(role="system", content="sys1"),
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
        ChatMessage(role="user", content="how are you?"),
    ]
    sysp, userp = convert_messages(msgs)
    assert sysp == "sys1"
    assert "User: hi" in userp
    assert "Assistant: hello" in userp
    assert "User: how are you?" in userp


def test_convert_messages_multiple_systems_joined():
    msgs = [
        ChatMessage(role="system", content="a"),
        ChatMessage(role="developer", content="b"),
        ChatMessage(role="user", content="q"),
    ]
    sysp, userp = convert_messages(msgs)
    assert sysp == "a\n\nb"
    assert userp == "User: q"


def test_convert_messages_empty_user_fallback():
    msgs = [ChatMessage(role="system", content="only system")]
    sysp, userp = convert_messages(msgs)
    assert sysp == "only system"
    assert userp == "Hello"


def test_convert_messages_multimodal_text_part():
    msgs = [
        ChatMessage(
            role="user",
            content=[{"type": "text", "text": "what is this?"}],
        )
    ]
    sysp, userp = convert_messages(msgs)
    assert sysp is None
    assert "what is this?" in userp


def test_convert_messages_image_produces_blocks():
    msgs = [
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "describe"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,AAAA"
                    },
                },
            ],
        )
    ]
    sysp, userp = convert_messages(msgs)
    assert sysp is None
    assert isinstance(userp, list)
    types = [b["type"] for b in userp]
    assert "text" in types
    assert "image" in types
    img = next(b for b in userp if b["type"] == "image")
    assert img["source"]["type"] == "base64"
    assert img["source"]["media_type"] == "image/png"
    assert img["source"]["data"] == "AAAA"


def test_convert_messages_image_url_http():
    msgs = [
        ChatMessage(
            role="user",
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/x.png"},
                }
            ],
        )
    ]
    _, userp = convert_messages(msgs)
    assert isinstance(userp, list)
    img = userp[0]
    assert img["source"] == {"type": "url", "url": "https://example.com/x.png"}


def test_convert_image_url_invalid_returns_none():
    assert _convert_image_url_to_claude({"image_url": {}}) is None
    assert _convert_image_url_to_claude({"image_url": {"url": "data:broken"}}) is None


def test_convert_messages_multiple_images_preserved():
    msgs = [
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "compare"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAA"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/b.png"},
                },
            ],
        )
    ]
    _, userp = convert_messages(msgs)
    assert isinstance(userp, list)
    # 先頭に text、続いて image を順番通り
    assert userp[0]["type"] == "text"
    images = [b for b in userp if b["type"] == "image"]
    assert len(images) == 2
    assert images[0]["source"]["type"] == "base64"
    assert images[0]["source"]["data"] == "AAA"
    assert images[1]["source"]["type"] == "url"
    assert images[1]["source"]["url"] == "https://example.com/b.png"


def test_convert_messages_image_in_assistant_role_ignored():
    msgs = [
        ChatMessage(
            role="user",
            content=[{"type": "text", "text": "look"}],
        ),
        ChatMessage(
            role="assistant",
            content=[
                {"type": "text", "text": "ok"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAA"},
                },
            ],
        ),
    ]
    _, userp = convert_messages(msgs)
    # 画像が assistant にしかなければ blocks に昇格しない → str のまま
    assert isinstance(userp, str)
    assert "User: look" in userp
    assert "Assistant: ok" in userp


def test_convert_messages_images_across_multiple_user_messages():
    msgs = [
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "first"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAA"},
                },
            ],
        ),
        ChatMessage(role="assistant", content="got it"),
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "second"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,BBB"},
                },
            ],
        ),
    ]
    _, userp = convert_messages(msgs)
    assert isinstance(userp, list)
    images = [b for b in userp if b["type"] == "image"]
    assert len(images) == 2
    assert images[0]["source"]["data"] == "AAA"
    assert images[1]["source"]["data"] == "BBB"
    text_blocks = [b for b in userp if b["type"] == "text"]
    assert len(text_blocks) == 1
    joined = text_blocks[0]["text"]
    assert "User: first" in joined
    assert "Assistant: got it" in joined
    assert "User: second" in joined


# ---------------------------------------------------------------------------
# _make_prompt
# ---------------------------------------------------------------------------


def test_make_prompt_string_passthrough():
    result = asyncio.run(_make_prompt("hello"))
    assert result == "hello"


def test_make_prompt_list_returns_async_iterator():
    blocks = [{"type": "text", "text": "x"}]

    async def _collect():
        stream = await _make_prompt(blocks)
        items = []
        async for item in stream:
            items.append(item)
        return items

    items = asyncio.run(_collect())
    assert len(items) == 1
    msg = items[0]
    assert msg["type"] == "user"
    assert msg["message"]["content"] == blocks


# ---------------------------------------------------------------------------
# _sse_chunk
# ---------------------------------------------------------------------------


def test_sse_chunk_shape():
    chunk = _sse_chunk("id1", 111, "sonnet", {"content": "hi"}, None)
    assert chunk.startswith("data: ")
    assert chunk.endswith("\n\n")
    payload = json.loads(chunk[len("data: "): -2])
    assert payload["id"] == "id1"
    assert payload["created"] == 111
    assert payload["model"] == "sonnet"
    assert payload["choices"][0]["delta"] == {"content": "hi"}
    assert payload["choices"][0]["finish_reason"] is None


# ---------------------------------------------------------------------------
# エラー整形
# ---------------------------------------------------------------------------


def test_compact_exception_text_truncates():
    exc = Exception("\n".join(f"line{i}" for i in range(50)))
    text = _compact_exception_text(exc, max_lines=3, max_chars=1000)
    assert text.count("\n") == 2  # 3行 → 改行2つ


def test_compact_exception_text_empty_fallback():
    class MyErr(Exception):
        pass

    assert _compact_exception_text(MyErr("")) == "MyErr"


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("Command failed with exit code 1", True),
        ("cannot write to process that exited", True),
        ("Broken pipe", True),
        ("Control request timeout: initialize", True),
        ("some other error", False),
    ],
)
def test_is_retryable_claude_error(msg, expected):
    assert _is_retryable_claude_error(Exception(msg)) is expected


def test_to_claude_proxy_error_generic():
    err = _to_claude_proxy_error(RuntimeError("boom"))
    assert isinstance(err, ClaudeProxyError)
    assert err.status_code == 500
    assert err.code == "claude_backend_failed"


def test_to_claude_proxy_error_process_failed_via_text():
    err = _to_claude_proxy_error(Exception("Command failed with exit code 2"))
    assert err.status_code == 502
    assert err.code == "claude_cli_process_failed"


def test_to_claude_proxy_error_initialize_timeout():
    err = _to_claude_proxy_error(Exception("Control request timeout: initialize"))
    assert err.status_code == 503
    assert err.code == "claude_backend_initialize_timeout"


def test_claude_proxy_error_to_openai_shape():
    err = ClaudeProxyError(500, "msg", "server_error", "x")
    payload = err.to_openai_error()
    assert payload == {"error": {"message": "msg", "type": "server_error", "code": "x"}}


# ---------------------------------------------------------------------------
# call_claude のリトライ挙動（SDK をモック）
# ---------------------------------------------------------------------------


class _FakeAssistantMessage:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]


class _FakeTextBlock:
    def __init__(self, text):
        self.text = text


def _make_query_fn(outputs):
    """outputs: list of either str (成功テキスト) or Exception (投げる)。"""
    calls = {"n": 0}

    def _query(prompt, options):  # noqa: ARG001
        idx = calls["n"]
        calls["n"] += 1
        item = outputs[idx]

        async def _gen():
            if isinstance(item, Exception):
                raise item
            yield _FakeAssistantMessage(item)

        return _gen()

    return _query, calls


def _make_capturing_query_fn(outputs):
    """_make_query_fn と同じだが、query() に渡された prompt を記録する。

    AsyncIterable の場合は消費して list として保存するため、SDK に
    実際に届く user message dict を検証できる。
    """
    calls = {"n": 0, "prompts": []}

    def _query(prompt, options):  # noqa: ARG001
        idx = calls["n"]
        calls["n"] += 1
        item = outputs[idx]

        async def _gen():
            if isinstance(prompt, str):
                calls["prompts"].append(prompt)
            else:
                collected = []
                async for msg in prompt:
                    collected.append(msg)
                calls["prompts"].append(collected)

            if isinstance(item, Exception):
                raise item
            yield _FakeAssistantMessage(item)

        return _gen()

    return _query, calls


def _patch_sdk(query_fn):
    """claude_agent_sdk.query を差し替え、AssistantMessage / TextBlock も用意。"""
    fake_mod = SimpleNamespace(
        query=query_fn,
        AssistantMessage=_FakeAssistantMessage,
        TextBlock=_FakeTextBlock,
        ClaudeAgentOptions=lambda **kw: SimpleNamespace(**kw),
        CLIConnectionError=type("CLIConnectionError", (Exception,), {}),
        CLINotFoundError=type("CLINotFoundError", (Exception,), {}),
        ProcessError=type("ProcessError", (Exception,), {}),
    )
    return patch.dict(sys.modules, {"claude_agent_sdk": fake_mod})


def test_call_claude_success():
    query_fn, calls = _make_query_fn(["hello world"])
    with _patch_sdk(query_fn):
        req = ChatCompletionRequest(
            model="sonnet", messages=[ChatMessage(role="user", content="hi")]
        )
        text = asyncio.run(call_claude(req))
    assert text == "hello world"
    assert calls["n"] == 1


def test_call_claude_retries_on_retryable():
    query_fn, calls = _make_query_fn(
        [Exception("Command failed with exit code 1"), "recovered"]
    )
    with _patch_sdk(query_fn), patch("server.asyncio.sleep", new=_async_noop):
        req = ChatCompletionRequest(
            model="sonnet", messages=[ChatMessage(role="user", content="hi")]
        )
        text = asyncio.run(call_claude(req))
    assert text == "recovered"
    assert calls["n"] == 2


def test_call_claude_retries_on_initialize_timeout():
    query_fn, calls = _make_query_fn(
        [Exception("Control request timeout: initialize"), "recovered"]
    )
    with _patch_sdk(query_fn), patch("server.asyncio.sleep", new=_async_noop):
        req = ChatCompletionRequest(
            model="sonnet", messages=[ChatMessage(role="user", content="hi")]
        )
        text = asyncio.run(call_claude(req))
    assert text == "recovered"
    assert calls["n"] == 2


def test_call_claude_no_retry_on_non_retryable():
    query_fn, calls = _make_query_fn([RuntimeError("unexpected")])
    with _patch_sdk(query_fn):
        req = ChatCompletionRequest(
            model="sonnet", messages=[ChatMessage(role="user", content="hi")]
        )
        with pytest.raises(ClaudeProxyError):
            asyncio.run(call_claude(req))
    assert calls["n"] == 1


def test_call_claude_exhausts_retries():
    errs = [
        Exception("Command failed with exit code 1"),
        Exception("Command failed with exit code 1"),
    ]
    query_fn, calls = _make_query_fn(errs)
    with _patch_sdk(query_fn), patch("server.asyncio.sleep", new=_async_noop):
        req = ChatCompletionRequest(
            model="sonnet", messages=[ChatMessage(role="user", content="hi")]
        )
        with pytest.raises(ClaudeProxyError):
            asyncio.run(call_claude(req))
    assert calls["n"] == server._CLAUDE_QUERY_MAX_ATTEMPTS


async def _async_noop(*_a, **_kw):
    return None


# ---------------------------------------------------------------------------
# stream_claude（SSE）
# ---------------------------------------------------------------------------


def test_stream_claude_yields_chunks():
    query_fn, _ = _make_query_fn(["streamed text"])

    async def _collect():
        req = ChatCompletionRequest(
            model="sonnet",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        chunks = []
        async for c in stream_claude(req):
            chunks.append(c)
        return chunks

    with _patch_sdk(query_fn):
        chunks = asyncio.run(_collect())

    assert chunks[-1] == "data: [DONE]\n\n"
    # role chunk + content chunk + stop chunk + [DONE]
    assert any('"role": "assistant"' in c for c in chunks)
    assert any("streamed text" in c for c in chunks)
    assert any('"finish_reason": "stop"' in c for c in chunks)


def test_stream_claude_emits_error_payload():
    query_fn, _ = _make_query_fn([RuntimeError("boom")])

    async def _collect():
        req = ChatCompletionRequest(
            model="sonnet",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        return [c async for c in stream_claude(req)]

    with _patch_sdk(query_fn):
        chunks = asyncio.run(_collect())

    assert chunks[-1] == "data: [DONE]\n\n"
    error_chunks = [c for c in chunks if '"error"' in c]
    assert error_chunks
    payload = json.loads(error_chunks[0][len("data: "): -2])
    assert payload["error"]["code"] == "claude_backend_failed"


def test_stream_claude_with_image_passes_blocks_to_sdk():
    query_fn, calls = _make_capturing_query_fn(["streamed"])

    async def _collect():
        req = ChatCompletionRequest(
            model="sonnet",
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "what's this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAA"},
                        },
                    ],
                )
            ],
            stream=True,
        )
        return [c async for c in stream_claude(req)]

    with _patch_sdk(query_fn):
        chunks = asyncio.run(_collect())

    assert chunks[-1] == "data: [DONE]\n\n"
    assert any("streamed" in c for c in chunks)

    # SDK に image block が届いたことを確認
    captured = calls["prompts"][0]
    assert isinstance(captured, list)
    content = captured[0]["message"]["content"]
    assert any(b["type"] == "image" for b in content)


# ---------------------------------------------------------------------------
# FastAPI エンドポイント
# ---------------------------------------------------------------------------


def test_health_endpoint():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_chat_completions_non_stream():
    query_fn, _ = _make_query_fn(["hi there"])
    with _patch_sdk(query_fn):
        client = TestClient(app)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "sonnet",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hi there"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_error_maps_to_status():
    query_fn, _ = _make_query_fn(
        [
            Exception("Command failed with exit code 7"),
            Exception("Command failed with exit code 7"),
        ]
    )
    with _patch_sdk(query_fn), patch("server.asyncio.sleep", new=_async_noop):
        client = TestClient(app)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "sonnet",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "claude_cli_process_failed"


def test_chat_completions_image_reaches_sdk_as_base64_block():
    """OpenAI image_url (data URL) を投げたとき、SDK に base64 image block が届くこと。"""
    query_fn, calls = _make_capturing_query_fn(["described"])
    with _patch_sdk(query_fn):
        client = TestClient(app)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "sonnet",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what's this?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64,/9j/ABC"
                                },
                            },
                        ],
                    }
                ],
            },
        )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "described"

    captured = calls["prompts"][0]
    assert isinstance(captured, list), "画像があるとき AsyncIterable で送られる"
    assert len(captured) == 1
    msg = captured[0]
    assert msg["type"] == "user"
    content = msg["message"]["content"]
    assert isinstance(content, list)
    img = next(b for b in content if b["type"] == "image")
    assert img["source"] == {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": "/9j/ABC",
    }


def test_chat_completions_image_reaches_sdk_as_url_block():
    """HTTP URL の image_url を投げたとき、SDK に url source の image block が届くこと。"""
    query_fn, calls = _make_capturing_query_fn(["described"])
    with _patch_sdk(query_fn):
        client = TestClient(app)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "sonnet",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/x.png"},
                            }
                        ],
                    }
                ],
            },
        )
    assert r.status_code == 200
    captured = calls["prompts"][0]
    assert isinstance(captured, list)
    content = captured[0]["message"]["content"]
    img = next(b for b in content if b["type"] == "image")
    assert img["source"] == {"type": "url", "url": "https://example.com/x.png"}


def test_chat_completions_multiple_images_reach_sdk_in_order():
    """複数画像が順番通りに SDK まで届くこと。"""
    query_fn, calls = _make_capturing_query_fn(["ok"])
    with _patch_sdk(query_fn):
        client = TestClient(app)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "sonnet",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "compare"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,AAA"
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,BBB"
                                },
                            },
                        ],
                    }
                ],
            },
        )
    assert r.status_code == 200
    captured = calls["prompts"][0]
    content = captured[0]["message"]["content"]
    images = [b for b in content if b["type"] == "image"]
    assert len(images) == 2
    assert images[0]["source"]["data"] == "AAA"
    assert images[1]["source"]["data"] == "BBB"


def test_chat_completions_text_only_stays_string_prompt():
    """画像なしリクエストでは prompt が str のまま（回帰防止）。"""
    query_fn, calls = _make_capturing_query_fn(["hi"])
    with _patch_sdk(query_fn):
        client = TestClient(app)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "sonnet",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    assert r.status_code == 200
    captured = calls["prompts"][0]
    assert isinstance(captured, str), "テキストのみなら str で SDK に渡すべき"
    assert "User: hello" in captured


def test_models_cache_clear():
    server._cached_models = [{"value": "sonnet", "displayName": "Sonnet"}]
    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "sonnet"

    r = client.delete("/v1/models/cache")
    assert r.status_code == 200
    assert server._cached_models is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
