# CAS - Claude-as-OpenAI Proxy Server

Claude Agent SDK をバックエンドに、OpenAI Chat Completions API と互換のインターフェースを提供するローカルプロキシサーバー。

OpenAI SDK や curl からそのまま Claude モデルにアクセスできる。

## 特徴

- **OpenAI API 互換** — `/v1/chat/completions` と `/v1/models` をサポート
- **ドロップインリプレース** — OpenAI SDK の `base_url` を変えるだけで動く
- **ストリーミング対応** — SSE (Server-Sent Events) によるトークン逐次返却
- **画像入力対応** — OpenAI 形式の `image_url`（data URL / HTTP URL）をそのまま受け付ける
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

### 画像入力

OpenAI のマルチモーダル形式（`content` を配列にして `image_url` パートを混ぜる）をそのまま受け付ける。画像は `user` ロールのみ有効（`assistant` 等に混ぜても無視される）。

```python
import base64

with open("photo.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="sonnet",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "この画像を説明して"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
            ],
        }
    ],
)
```

HTTP(S) URL の画像もそのまま渡せる。

```python
{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
```

1 メッセージに複数画像を入れた場合、**入力順のまま** Claude に渡される。

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
- **会話履歴の平坦化** — Claude Agent SDK の `query()` は assistant ロール message を直接送れないため、multi-turn 履歴は `User: ...` / `Assistant: ...` のプレフィックスを付けて 1 つのテキストに結合して送られる。モデルはそれを「過去会話を含む 1 件のユーザー入力」として扱う（厳密な multi-turn 会話としては扱わない）
- **一部パラメータ無視** — `temperature`, `top_p`, `stop` 等は受け付けるが Claude 側では無視される
- **ツール制限** — 安全のため Bash, Write, Edit 等の破壊的ツールは無効化済み

## ライセンス

MIT
