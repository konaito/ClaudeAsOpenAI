# OpenAI Completions API ガイド

> 調査日: 2026-03-22
> ソース: https://developers.openai.com/api/docs/guides/completions

---

## 1. API の全体像

OpenAI には テキスト生成用の API が3つ存在する:

| API | 状態 | エンドポイント | 推奨度 |
|-----|------|---------------|--------|
| **Completions API** | レガシー（2023年7月最終更新） | `POST /v1/completions` | 非推奨 |
| **Chat Completions API** | 現行の業界標準 | `POST /v1/chat/completions` | 既存プロジェクト向け |
| **Responses API** | 最新（2025年〜） | `POST /v1/responses` | 新規プロジェクト推奨 |

Chat Completions API は引き続き無期限サポートされる。Responses API は Chat Completions の上位互換で、会話状態のサーバー管理・組み込みツール・より良いキャッシュ効率を提供する。

---

## 2. Completions API（レガシー）

### 概要

フリーテキストの `prompt` を受け取り、テキスト補完を返すシンプルなエンドポイント。Chat Completions と違い、メッセージリストではなく単一のテキスト入力。

### 基本リクエスト

```python
from openai import OpenAI
client = OpenAI()

response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="Write a tagline for an ice cream shop."
)

print(response.choices[0].text)
```

### 固有機能: テキスト挿入（suffix）

`prompt`（前方）と `suffix`（後方）を指定して、その間のテキストを生成できる:

```python
response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="def fibonacci(n):\n",
    suffix="\n    return result",
)
```

用途: 段落途中の補完、アウトラインベースの文章生成、関数内のコード挿入など。

### レスポンス構造

```json
{
  "choices": [
    {
      "text": "生成されたテキスト",
      "finish_reason": "length",
      "logprobs": null
    }
  ],
  "created": 1234567890,
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

---

## 3. Chat Completions API（現行標準）

### エンドポイント

```
POST https://api.openai.com/v1/chat/completions
```

### 基本リクエスト

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### メッセージロール

| ロール | 用途 |
|--------|------|
| `system` / `developer` | モデルへの指示・コンテキスト設定 |
| `user` | ユーザーの入力（テキスト、画像、音声、ファイル対応） |
| `assistant` | モデルの応答（会話履歴として再送信用） |
| `tool` | ツール実行結果の返却 |

### 全パラメータ一覧

#### 必須パラメータ

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `model` | string | 使用モデル（`gpt-4o`, `gpt-4o-mini`, `o1`, `gpt-4.1` 等） |
| `messages` | array | 会話メッセージの配列 |

#### サンプリング制御（非推論モデル用）

| パラメータ | 型 | デフォルト | 範囲 | 説明 |
|-----------|-----|----------|------|------|
| `temperature` | float | 1.0 | 0〜2 | 出力のランダム性。低い=決定的、高い=多様 |
| `top_p` | float | 1.0 | 0〜1 | 核サンプリング。temperatureとの併用は非推奨 |
| `frequency_penalty` | float | 0.0 | -2.0〜2.0 | 頻出トークンへのペナルティ |
| `presence_penalty` | float | 0.0 | -2.0〜2.0 | 既出トークンへのペナルティ |
| `logit_bias` | object | null | -100〜100 | トークンIDごとのバイアス |
| `stop` | string/array | null | - | 生成停止文字列（最大4つ） |
| `logprobs` | boolean | false | - | トークンごとの対数確率を返す |
| `top_logprobs` | integer | - | 1〜20 | 上位N個の対数確率（logprobs=true必須） |

#### 出力制御

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `max_completion_tokens` | integer | null | 生成トークン上限（`max_tokens`の後継） |
| `n` | integer | 1 | 生成する応答数 |
| `stream` | boolean | false | SSE でストリーミング |
| `response_format` | object | - | `text` / `json_object` / `json_schema` |

#### 推論モデル用制御

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `reasoning_effort` | string | `"low"` / `"medium"` / `"high"` / `"minimal"` |
| `verbosity` | string | `"low"` / `"medium"` / `"high"` |

#### ツール / Function Calling

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `tools` | array | 関数定義の配列（name, description, parameters, strict） |
| `tool_choice` | string/object | `"auto"` / `"none"` / 特定関数指定 |
| `parallel_tool_calls` | boolean | 並列ツール呼び出しの有効化 |

#### マルチモーダル

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `modalities` | array | `["text"]` / `["text", "audio"]` |
| `audio` | object | 音声フォーマット（mp3/wav/flac/opus/pcm16）と声の選択 |

#### API管理

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `store` | boolean | サーバー側ログの抑制 |
| `seed` | integer | 再現性のためのシード値 |
| `service_tier` | string | `"priority"` / `"flex"` |
| `safety_identifier` | string | 安全追跡用のエンドユーザー識別子 |
| `prompt_cache_key` | string | キャッシュマッチング用ルーティングキー |

### ストリーミング

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a poem about APIs"}],
    stream=True,
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="")
```

- トークンは Server-Sent Events (SSE) として逐次送信される
- `data: [DONE]` でストリーム終了

### 構造化出力（Structured Outputs）

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Extract user info from: John, 30, NYC"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "user_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "city": {"type": "string"}
                },
                "required": ["name", "age", "city"],
                "additionalProperties": False
            }
        }
    }
)
```

`strict: True` でモデルはスキーマに厳密に従う。JSON Schemaのサブセットのみサポート。

### Function Calling / Tools

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            },
            "strict": True
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"
)

# モデルがツール呼び出しを返した場合
tool_call = response.choices[0].message.tool_calls[0]
# tool_call.function.name -> "get_weather"
# tool_call.function.arguments -> '{"location": "Tokyo", "unit": "celsius"}'
```

---

## 4. Responses API（最新・推奨）

### Chat Completions との主な違い

| 項目 | Chat Completions | Responses |
|------|-----------------|-----------|
| 会話状態 | クライアント側管理（毎回全履歴送信） | サーバー側管理可能 |
| レスポンス構造 | `choices[].message` | `output[]`（Items配列） |
| 組み込みツール | なし | Web検索、ファイル検索、コードインタプリタ、コンピュータ操作、リモートMCP |
| パフォーマンス | ベースライン | 推論モデルでSWE-bench +3%改善 |
| コスト | ベースライン | キャッシュ効率40〜80%改善 |

### 移行

```
# エンドポイント変更のみ（シンプルなケース）
POST /v1/chat/completions → POST /v1/responses
```

関数呼び出しやマルチモーダル入力を使っていない場合、メッセージ入力はそのまま互換。

---

## 5. ベストプラクティス

### プロンプト設計
- 明確な指示と関連コンテキストを含める
- `system`メッセージでモデルの役割・制約を定義
- 具体的なタスク指示を出す（曖昧な指示を避ける）

### エラーハンドリング
- **レートリミット**: リトライ with exponential backoff
- **タイムアウト**: 適切なタイムアウト設定
- **コンテンツポリシー**: フォールバック処理

### コスト最適化
- `max_completion_tokens` でトークン上限を設定
- 適切なモデル選択（`gpt-4o-mini` は安価で高速）
- ストリーミングでUX改善（体感速度向上）
- Responses API のキャッシュ効率を活用

### セキュリティ
- APIキーは環境変数で管理
- ユーザー入力のサニタイズ
- `safety_identifier` でユーザー追跡

---

## 6. 主要モデル一覧（2025年時点）

| モデル | 特徴 |
|--------|------|
| `gpt-4o` | マルチモーダルフラグシップ |
| `gpt-4o-mini` | 高速・安価 |
| `gpt-4.1` | 最新世代GPT-4 |
| `gpt-4.1-mini` | 4.1の軽量版 |
| `o1` | 高度な推論モデル |
| `o3` / `o4-mini` | 次世代推論モデル |
| `gpt-3.5-turbo-instruct` | Completions API用（レガシー） |

---

## 参考リンク

- [Chat Completions API Reference](https://platform.openai.com/docs/api-reference/chat/create)
- [Completions API Guide](https://developers.openai.com/api/docs/guides/completions)
- [Responses API vs Chat Completions](https://platform.openai.com/docs/guides/responses-vs-chat-completions)
- [Responses API 移行ガイド](https://developers.openai.com/api/docs/guides/migrate-to-responses)
- [Complete Guide to the OpenAI API 2025 (Zuplo)](https://zuplo.com/learning-center/openai-api)
