# CAS - Claude-as-OpenAI Proxy Server

Claude Agent SDK をバックエンドに、OpenAI Chat Completions API と互換のインターフェースを提供するローカルプロキシサーバー。

OpenAI SDK や curl からそのまま Claude モデルにアクセスできる。

## 特徴

- **OpenAI API 互換** — `/v1/chat/completions` と `/v1/models` をサポート
- **ドロップインリプレース** — OpenAI SDK の `base_url` を変えるだけで動く
- **ストリーミング対応** — SSE (Server-Sent Events) によるトークン逐次返却
- **モデル一覧** — Claude Code CLI から利用可能モデルを動的取得
- **Swagger UI** — `/docs` でインタラクティブに API を試せる

## 必要なもの

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) がインストール済みで認証済みであること

## 起動

```bash
uv run server.py
```

サーバーが `http://localhost:8000` で起動する。

## 使い方

### curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"sonnet","messages":[{"role":"user","content":"Hello"}]}'
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="sonnet",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

### ストリーミング

```python
stream = client.chat.completions.create(
    model="sonnet",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## API エンドポイント

| メソッド | パス | 説明 |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | Chat Completions（ストリーミング対応） |
| `GET` | `/v1/models` | 利用可能モデル一覧 |
| `DELETE` | `/v1/models/cache` | モデルキャッシュクリア |
| `GET` | `/health` | ヘルスチェック |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc |

## モデル指定

`model` にはエイリアスまたはフルネームを指定できる。

| エイリアス | 説明 |
| --- | --- |
| `default` | デフォルトモデル |
| `sonnet` | Claude Sonnet |
| `opus` | Claude Opus |
| `haiku` | Claude Haiku |

フルネーム（例: `claude-sonnet-4-6`）もそのまま使える。利用可能なモデルは `GET /v1/models` で確認。

## 注意事項

- **ローカル専用** — 認証なしで動作するため、外部公開は非推奨
- **ワンショット** — セッション管理なし。リクエストごとに独立した会話
- **一部パラメータ無視** — `temperature`, `top_p`, `stop` 等は受け付けるが Claude 側では無視される
- **ツール制限** — 安全のため Bash, Write, Edit 等の破壊的ツールは無効化済み

## ライセンス

MIT
