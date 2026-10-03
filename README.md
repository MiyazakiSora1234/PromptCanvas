# PromptCanvas

Hugging Face Diffusers を使った画像生成 Web アプリです。ブラウザからプロンプトを入力して画像を生成し、画面で確認して PNG でダウンロードできます。

```
PromptCanvas/
├── Makefile                     # セットアップ・起動・テストのタスク
├── backend/                     # API（FastAPI + Diffusers）
│   ├── app/
│   │   ├── main.py              # アプリ生成（ミドルウェア・起動時ロード・静的配信）
│   │   ├── api.py               # /api のルート
│   │   ├── error_handlers.py    # 例外 → エラー JSON への変換
│   │   ├── config.py            # 環境変数による設定（pydantic-settings）
│   │   ├── schemas.py           # リクエスト/レスポンス型とサーバー側入力検証
│   │   ├── generator.py         # Diffusers パイプラインのロードと生成
│   │   ├── limiter.py           # 同時実行数と待ち行列の制御
│   │   └── errors.py            # ユーザー向けエラーと例外の分類
│   ├── scripts/download_model.py
│   ├── tests/                   # pytest（GPU・torch 不要）
│   ├── requirements.txt / requirements-dev.txt
│   └── .env.example
└── frontend/                    # Web 画面（Vite + React + TypeScript + Tailwind CSS）
    ├── src/
    │   ├── api/                 # API の型とクライアント（fetch・エラー型）
    │   ├── hooks/               # 設定取得・状態ポーリング・生成処理
    │   ├── lib/                 # 入力検証・エラー表示ロジック（純粋関数）
    │   ├── components/          # 画面部品
    │   └── App.tsx
    ├── vite.config.ts           # 開発時に /api を FastAPI へプロキシ
    └── package.json
```

## 技術選定

| 領域 | 採用 | 理由 |
| --- | --- | --- |
| 画像生成 | Diffusers `AutoPipelineForText2Image` | モデル ID を変えるだけで SD1.5 / SDXL / SD-Turbo などを切り替えられる |
| API | FastAPI + Uvicorn | Diffusers と同じ Python で書け、Pydantic による型付き入力検証と OpenAPI ドキュメントが標準で付く |
| 設定 | pydantic-settings | 環境変数・`.env` を型付きで読み込み、不正な設定を起動時に検出できる |
| フロントエンド | React + TypeScript（Vite） | 状態（入力・生成中・結果・エラー）をコンポーネントとフックに分けて管理でき、型で API との契約を表現できる。Vite は設定が少なく開発サーバーが速い |
| スタイル | Tailwind CSS v4 | クラスだけでライト/ダークモード両対応の UI を組め、CSS ファイルの管理が不要 |
| テスト | pytest / Vitest + Testing Library | API は GPU なしで、画面は fetch をモックしてユーザー操作単位で検証できる |

フロントエンドと API は責務を分けています。フロントエンドは `/api/*` を呼ぶだけの静的アプリで、本番ではビルド成果物（`frontend/dist`）を FastAPI が同一オリジンで配信します。開発時は Vite の開発サーバーが `/api` を FastAPI にプロキシします。別オリジンで配信する場合は、ビルド時に `VITE_API_BASE`（例: `http://localhost:8000`）を設定し、API 側で `PROMPTCANVAS_CORS_ORIGINS` を設定してください。

## 前提条件

- **Python 3.11 以上**（3.14 で動作確認済み。使う Python バージョン向けの PyTorch ホイールがあることを確認してください）
- **Node.js 22 以上**（24 LTS で動作確認済み。フロントエンドのビルドに使用。Windows: `winget install OpenJS.NodeJS.LTS`）
- ディスク空き容量：既定モデル（SD1.5）で約 5GB、SDXL なら約 10GB 以上（`~/.cache/huggingface` に保存）
- 初回起動時にモデルをダウンロードするためのインターネット接続

### GPU / CPU について

| 環境 | 動作 | 目安（SD1.5, 512×512, 25 ステップ） |
| --- | --- | --- |
| NVIDIA GPU（VRAM 6GB 以上推奨）+ CUDA 版 PyTorch | `cuda` を自動選択、float16 で実行 | 数秒 |
| Apple Silicon（MPS） | `mps` を自動選択 | 数十秒 |
| GPU なし | **CPU で実行**（float32）。起動ログと画面上部に「CPU」と表示 | 数分以上 |

- `PROMPTCANVAS_DEVICE=auto`（既定）は CUDA → MPS → CPU の順に選びます。`cuda` を明示したのに使えない場合は CPU に**黙って切り替えず**、モデル読み込み失敗として画面とログに理由を表示します。
- **RTX 50 シリーズ（Blackwell）は CUDA 12.8 以上の PyTorch（`cu128`）が必要**です。古い CUDA 版では `no kernel image is available` で失敗します。
- VRAM が少ない場合は `PROMPTCANVAS_ENABLE_ATTENTION_SLICING=true` や `PROMPTCANVAS_ENABLE_CPU_OFFLOAD=true` を設定してください（速度は落ちます）。
- 生成は 1 プロセスにつき同時に 1 件です（Diffusers のパイプラインはスレッドセーフではないため）。それ以上のリクエストは最大 `PROMPTCANVAS_MAX_QUEUE_SIZE` 件まで待機し、超えると 429 を返します。

## セットアップ

### Makefile を使う場合（リポジトリ直下で実行）

GNU Make が必要です（Windows: `winget install ezwinports.make`）。Windows（cmd / Git Bash）と macOS / Linux のどちらでも動きます。

```bash
make setup                 # venv + PyTorch(CUDA 12.8) + Python/npm 依存関係 + フロントエンドのビルド
make setup TORCH=cpu       # GPU がない場合
make env                   # backend/.env を .env.example から作成（既存なら何もしない）
make download-model        # モデルを事前取得（任意。起動時にも自動取得される）
make run                   # フロントエンドをビルドして http://127.0.0.1:8000 で起動（PORT=8080 などで変更可）
make check                 # lint + 型チェック + テスト（バックエンド・フロントエンド両方）
make help                  # ターゲット一覧
```

### 手動で行う場合

Windows（PowerShell）の例です。macOS / Linux では `.venv\Scripts\Activate.ps1` を `source .venv/bin/activate` に読み替えてください。

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 1. PyTorch をインストール（環境に合わせて 1 つ選ぶ）

```powershell
# NVIDIA GPU（CUDA 12.8。RTX 50 シリーズを含む新しい GPU）
pip install torch --index-url https://download.pytorch.org/whl/cu128

# CPU のみ
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

その他の組み合わせは https://pytorch.org/get-started/locally/ を参照してください。インストール後、GPU が認識されているか確認できます：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 2. アプリの依存関係をインストール

```powershell
pip install -r requirements.txt
```

### 3. 設定（任意）

```powershell
Copy-Item .env.example .env
```

`backend/.env` を編集します（すべて省略可能。環境変数でも指定できます）。

### 4. フロントエンドをビルド

```powershell
cd ..\frontend
npm ci
npm run build      # frontend/dist に出力され、FastAPI が配信する
```

## 環境変数

| 変数 | 既定値 | 説明 |
| --- | --- | --- |
| `PROMPTCANVAS_MODEL_ID` | `stable-diffusion-v1-5/stable-diffusion-v1-5` | Hugging Face のモデル ID またはローカルパス |
| `PROMPTCANVAS_MODEL_REVISION` | なし | ブランチ / タグ / コミット |
| `PROMPTCANVAS_MODEL_VARIANT` | なし | `fp16` などの重みバリアント（存在するモデルのみ） |
| `HF_TOKEN` | なし | ゲート付き・非公開モデル用のトークン。ログ・レスポンスには出力しません |
| `PROMPTCANVAS_DEVICE` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `PROMPTCANVAS_TORCH_DTYPE` | `auto` | `auto`（GPU: float16 / CPU: float32）/ `float16` / `bfloat16` / `float32` |
| `PROMPTCANVAS_ENABLE_ATTENTION_SLICING` | `false` | VRAM 節約（やや低速） |
| `PROMPTCANVAS_ENABLE_CPU_OFFLOAD` | `false` | モデルを必要時のみ GPU に載せる（CUDA のみ、低速だが大幅に VRAM 節約） |
| `PROMPTCANVAS_MIN_IMAGE_SIZE` / `MAX_IMAGE_SIZE` | `256` / `1024` | 幅・高さの範囲（8 の倍数） |
| `PROMPTCANVAS_DEFAULT_WIDTH` / `DEFAULT_HEIGHT` | `512` / `512` | 既定サイズ |
| `PROMPTCANVAS_MAX_STEPS` / `DEFAULT_STEPS` | `50` / `25` | ステップ数の上限と既定値 |
| `PROMPTCANVAS_MAX_GUIDANCE_SCALE` / `DEFAULT_GUIDANCE_SCALE` | `20` / `7.5` | ガイダンススケールの上限と既定値 |
| `PROMPTCANVAS_MAX_PROMPT_LENGTH` | `1000` | プロンプト・ネガティブプロンプトの最大文字数 |
| `PROMPTCANVAS_MAX_QUEUE_SIZE` | `4` | 生成中に待機できるリクエスト数。超えると 429 |
| `PROMPTCANVAS_QUEUE_TIMEOUT_SECONDS` | `300` | 待機の上限秒数。超えると 503 |
| `PROMPTCANVAS_CORS_ORIGINS` | `[]` | 別オリジンからアクセスする場合の許可リスト（JSON 配列） |
| `PROMPTCANVAS_SERVE_FRONTEND` | `true` | フロントエンドのビルド成果物を `/` で配信するか |
| `PROMPTCANVAS_FRONTEND_DIR` | `frontend/dist` | 配信するビルド成果物の場所。存在しなければ API のみ起動（警告ログ） |
| `PROMPTCANVAS_LOG_LEVEL` | `INFO` | ログレベル |

設定値の整合性（既定サイズが範囲内か、8 の倍数か等）は起動時に検証され、不正なら起動に失敗します。

### モデルの例

| モデル ID | 推奨設定 | VRAM 目安 |
| --- | --- | --- |
| `stable-diffusion-v1-5/stable-diffusion-v1-5`（既定） | 512×512、25 ステップ、ガイダンス 7.5 | 4GB〜 |
| `stabilityai/stable-diffusion-xl-base-1.0` | `DEFAULT_WIDTH/HEIGHT=1024`、`MODEL_VARIANT=fp16` | 10GB〜 |
| `stabilityai/sd-turbo` | `DEFAULT_STEPS=1`、`DEFAULT_GUIDANCE_SCALE=0`（ネガティブプロンプトは無効） | 4GB〜 |

## モデルの取得

初回起動時に `PROMPTCANVAS_MODEL_ID` のモデルを自動でダウンロードし、`~/.cache/huggingface/hub` にキャッシュします（2 回目以降はキャッシュを使用）。ダウンロード中も API は起動しており、画面上部に「モデル読み込み中…」と表示されます。

事前に取得しておく場合：

```powershell
cd backend
.venv\Scripts\python.exe -m scripts.download_model   # または make download-model
```

サーバーと同じ設定（`PROMPTCANVAS_MODEL_ID` / `REVISION` / `VARIANT` / `HF_TOKEN`）を使い、パイプラインに必要なファイルだけを取得します（リポジトリ内の不要な大容量 ckpt はダウンロードしません）。

- ゲート付きモデル（例: SD3、FLUX.1-dev）は Hugging Face のモデルページで利用規約に同意し、`HF_TOKEN` を環境変数か `backend/.env` に設定してください。トークンはコードやリポジトリにコミットしないでください（`.env` は `.gitignore` 済み）。
- 新しい `huggingface_hub` は Xet ストレージ経由でダウンロードし、受信データをメモリに溜めてから書き出します。そのためダウンロード中でもキャッシュフォルダのサイズが増えないことがあります（進行状況は `GET /api/health` が `loading` かどうかで確認）。
- Windows で「symlinks に対応していない」という警告が出ますが、動作に問題はありません（開発者モードを有効にするとディスク使用量を抑えられます）。
- キャッシュ先は `HF_HOME` で変更できます。オフライン運用では取得後に `HF_HUB_OFFLINE=1` を設定してください。
- モデルのライセンス（例: CreativeML Open RAIL-M）を確認のうえ利用してください。

## 起動

### 通常の起動（ビルド済みフロントエンドを FastAPI が配信）

```powershell
make run
# または手動で
cd frontend; npm run build; cd ..\backend
.venv\Scripts\python.exe -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

ブラウザで http://127.0.0.1:8000 を開きます。API ドキュメントは http://127.0.0.1:8000/docs です。

### フロントエンド開発（ホットリロード）

ターミナルを 2 つ使います。

```powershell
make dev-api    # API（:8000）
make dev-web    # Vite 開発サーバー（:5173）。/api は :8000 にプロキシ
```

http://localhost:5173 を開くと、`frontend/src` の変更が即座に反映されます。API の転送先は環境変数 `PROMPTCANVAS_API_URL` で変更できます。

> `--reload` を付けるとファイル変更のたびにモデルを再ロードするため、開発時も通常は付けないことを推奨します。
> `--workers` を 2 以上にするとプロセスごとにモデルがロードされ、GPU メモリを倍以上消費します。

## API

| メソッド | パス | 内容 |
| --- | --- | --- |
| `GET` | `/api/health` | モデル状態（`loading` / `ready` / `failed`）、デバイス、待ち行列 |
| `GET` | `/api/config` | 入力値の範囲と既定値（画面の検証に使用） |
| `POST` | `/api/generate` | 画像生成。成功時は `image/png` を返し、`X-Seed` ヘッダに使用したシード値 |

リクエスト例：

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/generate `
  -H "Content-Type: application/json" `
  -d '{\"prompt\": \"a cat astronaut, digital art\", \"width\": 512, \"height\": 512, \"num_inference_steps\": 25, \"seed\": 42}' `
  -o out.png -D -
```

エラー時は次の形式の JSON を返します。`message` はユーザー向けの案内で、内部情報（例外メッセージ、パス、トークン）は含みません。詳細はサーバーログに `request_id` 付きで出力されます。

```json
{"error": {"code": "gpu_out_of_memory", "message": "GPUメモリが不足しました。…", "fields": [], "request_id": "3f2a9c1b7d4e"}}
```

| code | HTTP | 状況 |
| --- | --- | --- |
| `invalid_input` | 422 | 入力値が範囲外・形式不正（`fields` に項目別メッセージ） |
| `content_filtered` | 422 | セーフティチェッカーが画像をブロック |
| `server_busy` | 429 | 待ち行列が満杯 |
| `model_loading` | 503 | モデル読み込み中 |
| `model_unavailable` | 503 | モデル読み込みに失敗（理由を `message` に表示） |
| `gpu_out_of_memory` | 503 | GPU メモリ不足（サイズ・ステップを下げるよう案内） |
| `queue_timeout` | 503 | 待機時間の上限超過 |
| `generation_failed` | 500 | その他の生成失敗 |

## テスト・静的解析

```powershell
make check            # 以下をすべて実行
```

| 対象 | コマンド | 内容 |
| --- | --- | --- |
| API | `pytest` / `ruff check .` / `mypy`（`backend/`） | 生成処理をフェイクに差し替えるため torch・GPU 不要。入力検証（空・範囲外・8 の倍数でない・型不正・未知の項目・不正 JSON）、PNG 応答とシードヘッダ、モデル読み込み中/失敗時の 503、GPU メモリ不足の判別、内部情報が応答に漏れないこと、待ち行列の満杯・タイムアウト、デバイス/dtype 解決、設定値の整合性 |
| 画面 | `npm test` / `npm run lint` / `npm run typecheck`（`frontend/`） | fetch をモックし、生成〜表示〜ダウンロードリンク、送信前の入力検証とフォーカス移動、サーバーエラーの案内表示、生成中の重複送信防止、モデル読み込み中/失敗時のボタン無効化を検証 |

画面の動作確認は、サーバー起動後にブラウザで以下を確認してください：

1. 画面上部が「準備完了」になる
2. プロンプトを入力して「画像を生成」→ 生成中表示と経過秒数が出て、ボタンが無効化される
3. 画像が表示され「PNGをダウンロード」で保存できる
4. 幅に `500` を入れると、送信前に「8の倍数で指定してください」と表示される

## 制約と今後の改善点

- 生成は 1 プロセス 1 件ずつ。スループットを上げるには GPU ごとにプロセスを立ててロードバランサで振り分ける、またはジョブキュー（Redis + ワーカー）化が必要です。
- 生成の途中キャンセルや進捗率（ステップ単位）の表示は未対応です（経過秒数のみ）。`callback_on_step_end` と SSE/WebSocket で実装できます。
- CLIP のトークン上限（77 トークン）を超えるプロンプトは切り詰められます。
- 認証・レート制限（IP 単位）はありません。外部公開する場合はリバースプロキシ等で追加してください。
- 生成画像はサーバーに保存しません（履歴機能なし）。
