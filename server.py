"""
OpenAI Chat Completions API 互換サーバー（Claude Agent SDK バックエンド）

Claude Agent SDK を使って、OpenAI Chat Completions API と同じインターフェースで
Claude モデルにアクセスできるプロキシサーバー。

起動:
    uv run server.py

使い方:
    # curl
    curl http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model":"sonnet","messages":[{"role":"user","content":"Hello"}]}'

    # OpenAI Python SDK
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
    r = client.chat.completions.create(model="sonnet", messages=[...])

    # Swagger UI
    http://localhost:8000/docs
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "claude-agent-sdk",
#     "anyio",
# ]
# ///

import json
import logging
import os
import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

import uvicorn
from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("claude-proxy")
_CLAUDE_QUERY_MAX_ATTEMPTS = 2
_CLAUDE_RETRY_BASE_DELAY_SEC = 0.25

# ---------------------------------------------------------------------------
# OpenAI 互換の型定義
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """会話メッセージ。OpenAI の messages 配列の各要素。"""

    role: Literal["system", "developer", "user", "assistant", "tool"] = Field(
        ...,
        description="メッセージの送信者ロール",
        json_schema_extra={"examples": ["user"]},
    )
    content: str | list | None = Field(
        None,
        description=(
            "メッセージ本文。文字列またはマルチモーダル content parts の配列。"
            "マルチモーダルの場合、text タイプの部分のみ処理される。"
        ),
        json_schema_extra={"examples": ["Hello, how are you?"]},
    )
    name: str | None = Field(
        None,
        description="送信者の名前（任意）。同一ロールの区別に使う。",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"role": "user", "content": "What is 2+2?"}]
        }
    }


class ChatCompletionRequest(BaseModel):
    """OpenAI Chat Completions API 互換のリクエストボディ。

    OpenAI SDK からそのまま送れる形式。model にはClaude Code CLI が受け付ける
    モデル名をそのまま指定する（例: "sonnet", "opus", "haiku", "default"）。
    Claude 側で未対応のパラメータ（temperature 等）は受け付けるが無視される。
    """

    model: str = Field(
        "default",
        description=(
            "Claude Code CLI に渡すモデル名。"
            'エイリアス ("sonnet", "opus", "haiku", "default") '
            'またはフルネーム ("claude-sonnet-4-6") が使える。'
            "利用可能なモデルは GET /v1/models で確認可能。"
        ),
        json_schema_extra={
            "examples": ["default", "sonnet", "haiku", "opus"],
        },
    )
    messages: list[ChatMessage] = Field(
        ...,
        description="会話メッセージの配列。system → user → assistant の順で会話履歴を構成する。",
        min_length=1,
    )
    temperature: float | None = Field(
        None,
        ge=0,
        le=2,
        description="サンプリング温度 (0-2)。現在は無視される。",
    )
    top_p: float | None = Field(
        None,
        ge=0,
        le=1,
        description="核サンプリング (0-1)。現在は無視される。",
    )
    n: int = Field(
        1,
        ge=1,
        le=1,
        description="生成する応答数。現在は 1 のみサポート。",
    )
    stream: bool = Field(
        False,
        description="True にすると SSE (Server-Sent Events) でトークンを逐次返す。",
    )
    stop: str | list[str] | None = Field(
        None,
        description="生成停止文字列。現在は無視される。",
    )
    max_tokens: int | None = Field(
        None,
        ge=1,
        description="生成トークン上限（レガシー）。現在は無視される。",
    )
    max_completion_tokens: int | None = Field(
        None,
        ge=1,
        description="生成トークン上限。現在は無視される。",
    )
    presence_penalty: float | None = Field(
        None,
        ge=-2,
        le=2,
        description="既出トークンへのペナルティ (-2 to 2)。現在は無視される。",
    )
    frequency_penalty: float | None = Field(
        None,
        ge=-2,
        le=2,
        description="頻出トークンへのペナルティ (-2 to 2)。現在は無視される。",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "sonnet",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "What is the capital of France?"},
                    ],
                    "stream": False,
                }
            ]
        }
    }


class ChatCompletionChoice(BaseModel):
    """応答の選択肢。"""

    index: int = Field(0, description="選択肢のインデックス")
    message: ChatMessage = Field(..., description="アシスタントの応答メッセージ")
    finish_reason: str = Field("stop", description="生成停止の理由 (stop / length)")


class UsageInfo(BaseModel):
    """トークン使用量。"""

    prompt_tokens: int = Field(0, description="プロンプトのトークン数")
    completion_tokens: int = Field(0, description="生成のトークン数")
    total_tokens: int = Field(0, description="合計トークン数")


class ChatCompletionResponse(BaseModel):
    """OpenAI Chat Completions API 互換のレスポンス。"""

    id: str = Field(
        ...,
        description="一意な補完 ID",
        json_schema_extra={"examples": ["chatcmpl-abc123"]},
    )
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(..., description="作成タイムスタンプ (Unix epoch)")
    model: str = Field(..., description="リクエストで指定されたモデル名")
    choices: list[ChatCompletionChoice] = Field(..., description="応答の選択肢")
    usage: UsageInfo = Field(
        default_factory=UsageInfo, description="トークン使用量（概算）"
    )


class ModelInfo(BaseModel):
    """モデル情報。"""

    id: str = Field(..., description="モデル ID（CLI の --model に渡す値）")
    object: Literal["model"] = "model"
    created: int = Field(..., description="作成タイムスタンプ")
    owned_by: str = Field("anthropic", description="所有者")
    description: str | None = Field(None, description="モデルの説明")


class ModelListResponse(BaseModel):
    """モデル一覧レスポンス。"""

    object: Literal["list"] = "list"
    data: list[ModelInfo]


class ClaudeProxyError(Exception):
    """Claude SDK 起因のエラーを OpenAI 互換形式へ変換するための内部例外。"""

    def __init__(self, status_code: int, message: str, error_type: str, code: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.code = code

    def to_openai_error(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.code,
            }
        }


def _compact_exception_text(exc: Exception, *, max_lines: int = 8, max_chars: int = 1200) -> str:
    """例外メッセージを API レスポンス向けに短く整形。"""
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    if not lines:
        return exc.__class__.__name__
    text = "\n".join(lines[:max_lines])
    if len(text) > max_chars:
        return f"{text[:max_chars]}..."
    return text


def _is_retryable_claude_error(exc: Exception) -> bool:
    """一時的な CLI 失敗として再試行可能かを判定。"""
    message = str(exc).lower()
    retry_markers = (
        "command failed with exit code",
        "cannot write to process that exited",
        "broken pipe",
    )
    return any(marker in message for marker in retry_markers)


def _to_claude_proxy_error(exc: Exception) -> ClaudeProxyError:
    """SDK 例外を HTTP ステータスと OpenAI 互換エラーへマップ。"""
    from claude_agent_sdk import CLIConnectionError, CLINotFoundError, ProcessError

    detail = _compact_exception_text(exc)

    if isinstance(exc, CLINotFoundError):
        return ClaudeProxyError(
            status_code=503,
            message="Claude Code CLI is not installed or not available on PATH.",
            error_type="service_unavailable",
            code="claude_cli_not_found",
        )
    if isinstance(exc, CLIConnectionError):
        return ClaudeProxyError(
            status_code=503,
            message=f"Failed to connect to Claude Code CLI.\n{detail}",
            error_type="service_unavailable",
            code="claude_cli_connection_failed",
        )
    if isinstance(exc, ProcessError) or "command failed with exit code" in detail.lower():
        return ClaudeProxyError(
            status_code=502,
            message=f"Claude Code CLI process failed.\n{detail}",
            error_type="server_error",
            code="claude_cli_process_failed",
        )
    return ClaudeProxyError(
        status_code=500,
        message=f"Claude backend request failed.\n{detail}",
        error_type="server_error",
        code="claude_backend_failed",
    )


# ---------------------------------------------------------------------------
# messages 変換
# ---------------------------------------------------------------------------


def _convert_image_url_to_claude(part: dict) -> dict | None:
    """OpenAI image_url content part → Claude image content block に変換。"""
    try:
        url = part["image_url"]["url"]
    except (KeyError, TypeError):
        logger.warning("不正な image_url パート: %s", part)
        return None

    if url.startswith("data:"):
        # "data:image/jpeg;base64,/9j/4AAQ..."
        try:
            header, data = url.split(",", 1)
            media_type = header.split(":")[1].split(";")[0]
        except (ValueError, IndexError):
            logger.warning("不正な data URL: %s", url[:80])
            return None
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    else:
        return {
            "type": "image",
            "source": {"type": "url", "url": url},
        }


def convert_messages(messages: list[ChatMessage]) -> tuple[str | None, str | list]:
    """OpenAI messages 配列 → (system_prompt, user_prompt) に変換。

    system/developer ロールは system_prompt に集約。
    user/assistant/tool ロールは会話形式のテキストに結合。
    画像が含まれる場合、user_prompt は Claude content blocks のリストになる。
    """
    system_parts: list[str] = []
    conversation_parts: list[str] = []
    image_blocks: list[dict] = []

    for msg in messages:
        content = msg.content
        if isinstance(content, list):
            text_parts: list[str] = []
            for p in content:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "text":
                    text_parts.append(p["text"])
                elif p.get("type") == "image_url" and msg.role == "user":
                    block = _convert_image_url_to_claude(p)
                    if block:
                        image_blocks.append(block)
            content = "\n".join(text_parts) if text_parts else ""

        if not content and not image_blocks:
            continue

        if msg.role in ("system", "developer"):
            if content:
                system_parts.append(content)
        elif msg.role == "user":
            if content:
                conversation_parts.append(f"User: {content}")
        elif msg.role == "assistant":
            if content:
                conversation_parts.append(f"Assistant: {content}")
        elif msg.role == "tool":
            if content:
                conversation_parts.append(f"Tool result: {content}")

    system_prompt = "\n\n".join(system_parts) if system_parts else None

    if image_blocks:
        # 画像がある場合は Claude content blocks のリストとして返す
        content_blocks: list[dict] = []
        text = "\n\n".join(conversation_parts)
        if text:
            content_blocks.append({"type": "text", "text": text})
        content_blocks.extend(image_blocks)
        return system_prompt, content_blocks if content_blocks else "Hello"

    user_prompt = "\n\n".join(conversation_parts) if conversation_parts else "Hello"
    return system_prompt, user_prompt


def _make_options(
    model: str, system_prompt: str | None
) -> "ClaudeAgentOptions":
    """共通の ClaudeAgentOptions を生成。"""
    from claude_agent_sdk import ClaudeAgentOptions

    return ClaudeAgentOptions(
        model=model if model != "default" else None,
        system_prompt=system_prompt,
        max_turns=1,
        cwd=os.path.expanduser("~"),
        disallowed_tools=["Bash", "Write", "Edit", "MultiEdit", "NotebookEdit"],
    )


# ---------------------------------------------------------------------------
# Claude SDK 呼び出し
# ---------------------------------------------------------------------------


async def _make_prompt(
    user_prompt: str | list,
) -> "str | AsyncIterator[dict[str, Any]]":
    """user_prompt が list（画像含む）の場合は AsyncIterator に変換。

    SDK の query() は str か AsyncIterable[dict] を受け付ける。
    list content blocks を送るには AsyncIterable モードで
    user message dict を直接送る。
    """
    if isinstance(user_prompt, str):
        return user_prompt

    # list の場合は AsyncIterable を返す
    async def _stream() -> AsyncIterator[dict[str, Any]]:
        yield {
            "type": "user",
            "message": {"role": "user", "content": user_prompt},
            "parent_tool_use_id": None,
            "session_id": "",
        }

    return _stream()


async def call_claude(req: ChatCompletionRequest) -> str:
    """Claude Agent SDK でテキスト応答を取得（非ストリーミング）。"""
    from claude_agent_sdk import AssistantMessage, TextBlock, query

    system_prompt, user_prompt = convert_messages(req.messages)
    options = _make_options(req.model, system_prompt)
    prompt = await _make_prompt(user_prompt)

    last_error: Exception | None = None
    for attempt in range(1, _CLAUDE_QUERY_MAX_ATTEMPTS + 1):
        text_parts: list[str] = []
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
            return "\n".join(text_parts) if text_parts else ""
        except Exception as exc:
            last_error = exc
            if attempt < _CLAUDE_QUERY_MAX_ATTEMPTS and _is_retryable_claude_error(exc):
                logger.warning(
                    "Claude query failed on attempt %s/%s; retrying. error=%s",
                    attempt,
                    _CLAUDE_QUERY_MAX_ATTEMPTS,
                    _compact_exception_text(exc),
                )
                await asyncio.sleep(_CLAUDE_RETRY_BASE_DELAY_SEC * attempt)
                continue
            logger.exception("Claude query failed")
            raise _to_claude_proxy_error(exc) from exc

    if last_error is not None:
        raise _to_claude_proxy_error(last_error) from last_error
    return ""


async def stream_claude(req: ChatCompletionRequest):
    """Claude Agent SDK で SSE ストリーミング応答を生成。"""
    from claude_agent_sdk import AssistantMessage, TextBlock, query

    system_prompt, user_prompt = convert_messages(req.messages)
    options = _make_options(req.model, system_prompt)
    prompt = await _make_prompt(user_prompt)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    # role チャンク
    yield _sse_chunk(completion_id, created, req.model, {"role": "assistant", "content": ""}, None)

    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text:
                        yield _sse_chunk(completion_id, created, req.model, {"content": block.text}, None)
    except Exception as exc:
        logger.exception("Claude streaming query failed")
        proxy_error = _to_claude_proxy_error(exc)
        yield f"data: {json.dumps(proxy_error.to_openai_error())}\n\n"
        yield "data: [DONE]\n\n"
        return

    # 終了
    yield _sse_chunk(completion_id, created, req.model, {}, "stop")
    yield "data: [DONE]\n\n"


def _sse_chunk(
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, str],
    finish_reason: str | None,
) -> str:
    """SSE チャンクを1行生成。"""
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


# ---------------------------------------------------------------------------
# モデル一覧キャッシュ
# ---------------------------------------------------------------------------

_cached_models: list[dict[str, Any]] | None = None


async def _fetch_models() -> list[dict[str, Any]]:
    """Claude Code CLI から利用可能なモデル一覧を取得してキャッシュ。"""
    global _cached_models
    if _cached_models is not None:
        return _cached_models

    from claude_agent_sdk import ClaudeSDKClient

    try:
        async with ClaudeSDKClient() as client:
            info = await client.get_server_info()
            if info and "models" in info:
                _cached_models = info["models"]
                return _cached_models
    except Exception as e:
        logger.warning(f"Failed to fetch models from CLI: {e}")

    _cached_models = []
    return _cached_models


# ---------------------------------------------------------------------------
# FastAPI アプリケーション
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Claude-as-OpenAI Proxy",
    description=(
        "Claude Agent SDK をバックエンドに使い、OpenAI Chat Completions API と"
        "互換のインターフェースを提供するローカルプロキシサーバー。\n\n"
        "## 特徴\n"
        "- OpenAI SDK / curl からそのまま使える\n"
        "- model 名は Claude Code CLI にそのまま渡される（マッピングなし）\n"
        "- ストリーミング (SSE) 対応\n"
        "- セッション管理不要（毎回ワンショット）\n"
        "- モデル一覧は CLI から動的に取得\n\n"
        "## クイックスタート\n"
        "```bash\n"
        "# 起動\n"
        "uv run server.py\n\n"
        "# テスト\n"
        'curl http://localhost:8000/v1/chat/completions \\\n'
        '  -H "Content-Type: application/json" \\\n'
        '  -d \'{"model":"sonnet","messages":[{"role":"user","content":"Hello"}]}\'\n'
        "```\n\n"
        "## OpenAI SDK からの利用\n"
        "```python\n"
        "from openai import OpenAI\n"
        'client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")\n'
        "r = client.chat.completions.create(\n"
        '    model="sonnet",\n'
        '    messages=[{"role": "user", "content": "Hello"}]\n'
        ")\n"
        "print(r.choices[0].message.content)\n"
        "```\n\n"
        "## 利用可能モデル\n"
        "GET `/v1/models` で Claude Code CLI が提供するモデル一覧を取得できる。\n"
        '`model` には `"default"`, `"sonnet"`, `"opus"`, `"haiku"` などのエイリアスか、\n'
        '`"claude-sonnet-4-6"` のようなフルネームを指定する。\n'
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    response_model_exclude_none=True,
    summary="Chat Completions を生成",
    description=(
        "OpenAI Chat Completions API 互換エンドポイント。\n\n"
        "messages 配列を受け取り、Claude モデルで応答を生成して返す。\n"
        "`stream: true` で SSE ストリーミングにも対応。\n\n"
        "**model**: Claude Code CLI が受け付けるモデル名をそのまま指定。\n"
        "**認証**: このローカルプロキシでは API キー不要（任意の値で OK）。"
    ),
    responses={
        200: {
            "description": "生成成功",
            "content": {
                "application/json": {
                    "example": {
                        "id": "chatcmpl-abc123def456",
                        "object": "chat.completion",
                        "created": 1700000000,
                        "model": "sonnet",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "Paris is the capital of France.",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                    }
                }
            },
        }
    },
    tags=["Chat"],
)
async def chat_completions(
    req: ChatCompletionRequest,
    authorization: Annotated[
        str | None, Header(description="Bearer トークン（任意、検証なし）")
    ] = None,
):
    if req.stream:
        return StreamingResponse(
            stream_claude(req),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        text = await call_claude(req)
    except ClaudeProxyError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.to_openai_error())
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    return ChatCompletionResponse(
        id=completion_id,
        created=int(time.time()),
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(),
    )


@app.get(
    "/v1/models",
    response_model=ModelListResponse,
    summary="利用可能モデル一覧",
    description=(
        "Claude Code CLI から利用可能なモデル一覧を動的に取得して返す。\n"
        "各モデルの `id` をそのまま `/v1/chat/completions` の `model` に使える。"
    ),
    tags=["Models"],
)
async def list_models():
    raw_models = await _fetch_models()
    created = int(time.time())
    data = []
    for m in raw_models:
        desc_parts = []
        if m.get("displayName"):
            desc_parts.append(m["displayName"])
        if m.get("description"):
            desc_parts.append(m["description"])
        data.append(
            ModelInfo(
                id=m.get("value", "unknown"),
                created=created,
                owned_by="anthropic",
                description=" - ".join(desc_parts) if desc_parts else None,
            )
        )
    return ModelListResponse(data=data)


@app.delete(
    "/v1/models/cache",
    summary="モデルキャッシュをクリア",
    description="モデル一覧のキャッシュを削除して、次回 GET /v1/models 時に CLI から再取得させる。",
    tags=["Models"],
)
async def clear_model_cache():
    global _cached_models
    _cached_models = None
    return {"status": "cleared"}


@app.get(
    "/health",
    summary="ヘルスチェック",
    description="サーバーの生存確認用。",
    tags=["System"],
)
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
