# PromptCanvas

Hugging Face Diffusers を使った画像生成 Web アプリです。ブラウザからプロンプトを入力して画像を生成し、画面で確認してダウンロードできます。

主な機能：

| 機能 | 内容 |
| --- | --- |
| モデル選択 | `catalog.json` に登録したモデル（既定: SD 1.5 / SDXL / Animagine XL）から選択。GPU に載せるのは常に 1 つで、選択に応じて入れ替え |
| サンプラー | モデル既定 / Euler / Euler a / DPM++ 2M / DPM++ 2M Karras / UniPC / DDIM |
| スタイル | 「リアルな写真（肌の質感）」を選ぶと、肌のきめ・毛穴を出す語句と、つるつるの肌・CG 感を避ける語句をプロンプトに自動で追加 |
| 設定プリセット | モデル・スタイル・サイズ・ステップ数などをまとめて保存・適用。組み込みの「リアルな人間」付き。自分の設定はブラウザに保存 |
| バッチ | 1 回で 1〜4 枚（シードは 1 枚ごとに +1）。サムネイルから選んで個別にダウンロード |
| 画像形式 | PNG / JPEG / WebP（JPEG・WebP は画質を指定可） |
| img2img | 画像をアップロードし、変換強度を指定して描き直し。出力サイズは元画像の縦横比に自動調整 |
| LoRA | `catalog.json` に登録した LoRA を強さ付きで適用。モデルの系統（SD1.5 / SDXL）が合うものだけ選択可 |
| 顔・ポーズの参照 | 顔の写真の人物のまま、服装・場面をプロンプトで変える（InstantID）。ポーズ参考画像と同じ姿勢にする（OpenPose ControlNet）。SDXL 系のみ |

```
PromptCanvas/
├── Makefile                     # セットアップ・起動・テストのタスク
├── backend/                     # API（FastAPI + Diffusers）
│   ├── app/
│   │   ├── main.py              # アプリ生成（ミドルウェア・起動時ロード・静的配信）
│   │   ├── api.py               # /api のルート
│   │   ├── error_handlers.py    # 例外 → エラー JSON への変換
│   │   ├── config.py            # 環境変数による設定（pydantic-settings）
│   │   ├── catalog.py           # catalog.json（選択可能なモデル・LoRA）の読み込みと検証
│   │   ├── schemas.py           # リクエスト/レスポンス型とサーバー側入力検証
│   │   ├── generator.py         # モデルの入れ替え・txt2img/img2img・LoRA・バッチ生成
│   │   ├── identity.py          # 顔・ポーズの参照（InstantID + OpenPose ControlNet）
│   │   ├── schedulers.py        # サンプラー一覧
│   │   ├── styles.py            # スタイル（プロンプトに加える語句）
│   │   ├── imaging.py           # アップロード画像のデコード、PNG/JPEG/WebP エンコード
│   │   ├── model_cache.py       # モデルがダウンロード済みかの判定
│   │   ├── limiter.py           # 同時実行数と待ち行列の制御
│   │   └── errors.py            # ユーザー向けエラーと例外の分類
│   ├── catalog.json             # 選択可能なモデルと LoRA の許可リスト
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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CPU のみ
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
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
| `PROMPTCANVAS_CATALOG_PATH` | `backend/catalog.json` | 選択可能なモデル・LoRA の一覧 |
| `PROMPTCANVAS_DEFAULT_MODEL` | カタログの `default_model` | 起動時に読み込むモデル（カタログの ID） |
| `HF_TOKEN` | なし | ゲート付き・非公開モデル用のトークン。ログ・レスポンスには出力しません |
| `PROMPTCANVAS_DEVICE` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `PROMPTCANVAS_TORCH_DTYPE` | `auto` | `auto`（GPU: float16 / CPU: float32）/ `float16` / `bfloat16` / `float32` |
| `PROMPTCANVAS_ENABLE_ATTENTION_SLICING` | `false` | VRAM 節約（やや低速） |
| `PROMPTCANVAS_ENABLE_CPU_OFFLOAD` | `false` | モデルを必要時のみ GPU に載せる（CUDA のみ、低速だが大幅に VRAM 節約） |
| `PROMPTCANVAS_MIN_IMAGE_SIZE` / `MAX_IMAGE_SIZE` | `256` / `1536` | 幅・高さの範囲（8 の倍数） |
| `PROMPTCANVAS_MAX_STEPS` | `50` | ステップ数の上限 |
| `PROMPTCANVAS_MAX_GUIDANCE_SCALE` | `20` | ガイダンススケールの上限 |
| `PROMPTCANVAS_MAX_PROMPT_LENGTH` | `1000` | プロンプト・ネガティブプロンプトの最大文字数 |
| `PROMPTCANVAS_MAX_BATCH_SIZE` | `4` | 1 回の生成枚数の上限（枚数に比例して時間と VRAM が増える） |
| `PROMPTCANVAS_MAX_LORAS` | `3` | 同時に適用できる LoRA の数（`0` で LoRA 機能を非表示） |
| `PROMPTCANVAS_MAX_INIT_IMAGE_MB` | `10` | img2img でアップロードできる画像の最大サイズ |
| `PROMPTCANVAS_MAX_QUEUE_SIZE` | `4` | 生成中に待機できるリクエスト数。超えると 429 |
| `PROMPTCANVAS_QUEUE_TIMEOUT_SECONDS` | `300` | 待機の上限秒数。超えると 503 |
| `PROMPTCANVAS_CORS_ORIGINS` | `[]` | 別オリジンからアクセスする場合の許可リスト（JSON 配列） |
| `PROMPTCANVAS_SERVE_FRONTEND` | `true` | フロントエンドのビルド成果物を `/` で配信するか |
| `PROMPTCANVAS_FRONTEND_DIR` | `frontend/dist` | 配信するビルド成果物の場所。存在しなければ API のみ起動（警告ログ） |
| `PROMPTCANVAS_LOG_LEVEL` | `INFO` | ログレベル |

設定値と `catalog.json` の整合性（各モデルの既定サイズが範囲内か、8 の倍数か、既定サンプラーが存在するか等）は起動時に検証され、不正なら起動に失敗します。

## モデルと LoRA（catalog.json）

利用者が選べるのは `backend/catalog.json` に登録したものだけです（任意のリポジトリを指定させると、任意のダウンロードを引き起こせてしまうため）。画面・API にはカタログの `id` だけが公開され、リポジトリ名は公開されません。

| モデル（id） | リポジトリ | 系統 | 基本サイズ | 容量 | 備考 |
| --- | --- | --- | --- | --- | --- |
| `sd15`（既定） | stable-diffusion-v1-5/stable-diffusion-v1-5 | sd15 | 512×512 | 約5GB | 軽量・高速。VRAM 4GB〜 |
| `sdxl` | stabilityai/stable-diffusion-xl-base-1.0（fp16） | sdxl | 1024×1024 | 約7GB | 高画質。VRAM 10GB〜 |
| `realvis-xl` | SG161222/RealVisXL_V5.0（fp16） | sdxl | 1024×1024 | 約6.6GB | 写実・人物の肌の質感に強い。ガイダンス 3.5、DPM++ 2M Karras、スタイル「リアルな写真」が既定。顔・ポーズ参照にもおすすめ |
| `animagine-xl` | cagliostrolab/animagine-xl-4.0 | sdxl | 832×1216 | 約7GB | アニメ・イラスト調。推奨サンプラー Euler a |

| LoRA（id） | リポジトリ | 系統 | 備考 |
| --- | --- | --- | --- |
| `pixel-art-xl` | nerijs/pixel-art-xl | sdxl | ドット絵風。プロンプトに `pixel art` を含めると効果的 |

追加するときは次の形式で書きます（`family` が一致しない LoRA は画面に出ず、API でも拒否されます）。

```json
{
  "models": [
    {
      "id": "my-model", "label": "表示名", "description": "説明",
      "repo": "org/repo（またはローカルのディレクトリ）", "variant": "fp16", "family": "sdxl",
      "download_size_gb": 6.9,
      "defaults": { "width": 1024, "height": 1024, "num_inference_steps": 30, "guidance_scale": 7.0, "scheduler": "default" }
    }
  ],
  "loras": [
    { "id": "my-lora", "label": "表示名", "repo": "org/repo", "weight_name": "file.safetensors",
      "family": "sdxl", "trigger_words": "trigger", "default_scale": 1.0 }
  ]
}
```

- モデルは diffusers 形式（`model_index.json` を含むリポジトリ）である必要があります。
- GPU に載せるモデルは常に 1 つです。別のモデルを選ぶと入れ替えに数秒〜数十秒かかり、その間ほかのリクエストは「読み込み中」になります。
- **未ダウンロードのモデルを選ぶと、初回の生成時に数 GB のダウンロードが走り、数分〜数十分待つことになります。** 画面ではモデル名に「（要ダウンロード）」と表示し、選ぶと警告を出します。事前に `make download-model` で取得しておくことを推奨します。

## 設定プリセット

画面上部の「設定プリセット」で、生成設定をまとめて保存・適用できます。

- 保存されるもの：モデル、スタイル、サンプラー、サイズ、ステップ数、ガイダンス、ネガティブプロンプト、枚数、画像形式・画質、LoRA と強さ、顔の再現度・ポーズの強さ。
- 保存されないもの：プロンプト、シード、画像（毎回変えるもののため）。適用してもプロンプトはそのまま残ります。
- **組み込みプリセット**は `catalog.json` の `presets` に定義し、サーバー起動時に検証されます。標準で「リアルな人間」（RealVisXL・スタイル「リアルな写真」・896×1152・30 ステップ・ガイダンス 3.5・手や目の崩れを防ぐネガティブプロンプト）を用意しています。
- **自分で保存したプリセット**はブラウザ（localStorage）に保存されます。同じ名前で保存すると上書きされます。別のブラウザ・PC とは共有されず、ブラウザのデータを消すと消えます。
- サーバーから削除されたモデル・LoRA・サンプラーを含むプリセットを適用すると、その項目だけ外して（または既定値にして）理由を表示します。

## モデルの取得

起動時に既定モデルを、それ以外のモデル・LoRA は初めて使うときに自動でダウンロードし、`~/.cache/huggingface/hub` にキャッシュします（2 回目以降はキャッシュを使用）。ダウンロード中も API は起動しており、画面上部に「… を読み込み中」と表示されます。

使う予定のものは事前に取得しておくと、初回の待ち時間がなくなります（カタログ全体で約19GB）：

```powershell
make download-model                                    # カタログのモデル・LoRA をすべて
cd backend; .venv\Scripts\python.exe -m scripts.download_model sdxl pixel-art-xl   # 一部だけ
```

`catalog.json` と同じ設定（revision / variant / weight_name、`HF_TOKEN`）を使い、パイプラインに必要なファイルだけを取得します（リポジトリ内の不要な大容量 ckpt はダウンロードしません）。

- ゲート付きモデル（例: SD3、FLUX.1-dev）は Hugging Face のモデルページで利用規約に同意し、`HF_TOKEN` を環境変数か `backend/.env` に設定してください。トークンはコードやリポジトリにコミットしないでください（`.env` は `.gitignore` 済み）。
- 新しい `huggingface_hub` は Xet ストレージ経由でダウンロードし、受信データをメモリに溜めてから書き出します。そのためダウンロード中でもキャッシュフォルダのサイズが増えないことがあります（進行状況は `GET /api/health` が `loading` かどうかで確認）。
- Windows で「symlinks に対応していない」という警告が出ますが、動作に問題はありません（開発者モードを有効にするとディスク使用量を抑えられます）。
- キャッシュ先は `HF_HOME` で変更できます。オフライン運用では取得後に `HF_HUB_OFFLINE=1` を設定してください。
- モデルのライセンス（例: CreativeML Open RAIL-M）を確認のうえ利用してください。

## 顔・ポーズの参照（InstantID + OpenPose）

「顔は元画像の人物のまま、ポーズや服装を変える」機能です。SDXL 系モデル（SDXL / Animagine XL）で使えます。

| 入力 | 役割 |
| --- | --- |
| 顔の写真 | この人物の顔の特徴（ArcFace の顔特徴量）を IP-Adapter で、顔の位置・向きを IdentityNet（ControlNet）で反映 |
| ポーズ参考画像 | OpenPose で骨格を検出し、ControlNet で同じ姿勢にする。顔の写真と併用時は、この画像の人物の顔の位置に顔を配置 |
| プロンプト | 服装・場面・画風（例: `photo of a woman in a red evening dress, ballroom`） |

- 顔だけ・ポーズだけ・両方の組み合わせで使えます。img2img とは同時に使えません。
- **顔の再現度**（既定 0.8）を上げるほど元の顔に近づきます。内部では IdentityNet（顔の位置・向き）にこの値を、IP-Adapter（顔の特徴）にはこの値 × `ip_adapter_ratio`（既定 0.65、`catalog.json` で変更可）を使います。IP-Adapter が強すぎると肌がつるつる・色が濃くなるため、弱めにしても顔の似方はほとんど変わらないことを比較して決めました。
- **リアルな肌の質感にするには**、モデルに RealVisXL を選んでください（スタイル「リアルな写真」、ガイダンス 3.5 が自動で設定されます）。同じ顔・シードで比較した結果、SDXL 標準モデルより肌のきめ・毛穴・そばかすが自然に出て、色も落ち着きます。ガイダンスを上げすぎる（7 以上）と、肌が硬く・ざらついた質感になりがちです。
- 写真の人物向けです。イラストの顔や横顔・小さく写った顔は検出できない、または似にくいことがあります。全身の構図では顔が小さくなるため、似る度合いが下がります。
- 追加のダウンロードは約 6.6GB（初回使用時、または `make download-model`）。顔の検出・特徴抽出は CPU で行い（1 枚 0.2 秒程度）、2 つの ControlNet は使用時だけ GPU に載せます（それ以外の生成の VRAM を圧迫しないため）。

| 部品 | リポジトリ | ライセンス |
| --- | --- | --- |
| InstantID（IdentityNet + IP-Adapter） | InstantX/InstantID | Apache-2.0 |
| 顔検出・顔特徴量（antelopev2: SCRFD / glintr100） | immich-app/antelopev2（revision 固定） | **InsightFace のモデルは非商用研究目的のみ** |
| OpenPose ControlNet | xinsir/controlnet-openpose-sdxl-1.0 | Apache-2.0 |
| 骨格検出（OpenPose body） | lllyasviel/Annotators | 各モデルのライセンスに従う |

> **注意:** 実在の人物の写真は、本人の同意を得たものだけを使ってください。生成した画像で他人になりすましたり、名誉・肖像権を侵害したりしないでください。画面にも同じ注意を表示しています。

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
| `GET` | `/api/health` | モデル状態（`loading` / `ready` / `failed`）、読み込み中のモデル、ダウンロード済みモデル、デバイス、待ち行列 |
| `GET` | `/api/config` | 入力値の範囲、選択肢（モデル・サンプラー・LoRA・画像形式）と各モデルの既定値 |
| `POST` | `/api/generate` | 画像生成。画像は JSON 内に base64 で返す |

`POST /api/generate` のリクエスト（`prompt` 以外は省略可。省略時は選んだモデルの既定値）：

```json
{
  "prompt": "a red fox in a forest",
  "negative_prompt": "blurry",
  "model": "sdxl",
  "scheduler": "dpmpp_2m_karras",
  "width": 1024, "height": 1024, "num_inference_steps": 30, "guidance_scale": 7.0,
  "seed": 42,
  "num_images": 2,
  "output_format": "webp", "quality": 90,
  "init_image": "data:image/png;base64,...", "strength": 0.6,
  "loras": [{ "id": "pixel-art-xl", "scale": 1.0 }]
}
```

レスポンス：

```json
{
  "images": [{ "seed": 42, "mime_type": "image/webp", "data": "<base64>" }, { "seed": 43, "...": "..." }],
  "model": "sdxl", "scheduler": "dpmpp_2m_karras", "width": 1024, "height": 1024,
  "num_inference_steps": 30, "guidance_scale": 7.0, "output_format": "webp",
  "elapsed_ms": 25148, "filtered_count": 0
}
```

PowerShell から 1 枚保存する例：

```powershell
$r = Invoke-RestMethod http://127.0.0.1:8000/api/generate -Method Post -ContentType "application/json" -Body '{"prompt":"a cat astronaut, digital art","seed":42}'
[IO.File]::WriteAllBytes("$PWD\out.png", [Convert]::FromBase64String($r.images[0].data))
```

エラー時は次の形式の JSON を返します。`message` はユーザー向けの案内で、内部情報（例外メッセージ、パス、トークン）は含みません。詳細はサーバーログに `request_id` 付きで出力されます。

```json
{"error": {"code": "gpu_out_of_memory", "message": "GPUメモリが不足しました。…", "fields": [], "request_id": "3f2a9c1b7d4e"}}
```

| code | HTTP | 状況 |
| --- | --- | --- |
| `invalid_input` | 422 | 入力値が範囲外・形式不正、カタログにないモデル/LoRA、系統の合わない LoRA、読めない画像（`fields` に項目別メッセージ） |
| `content_filtered` | 422 | セーフティチェッカーが全画像をブロック（一部だけなら `filtered_count` で通知） |
| `server_busy` | 429 | 待ち行列が満杯 |
| `model_loading` | 503 | モデル読み込み中 |
| `model_unavailable` | 503 | モデル読み込みに失敗（理由を `message` に表示）。次のリクエストで再試行される |
| `lora_unavailable` | 503 | LoRA の読み込みに失敗 |
| `reference_unavailable` | 503 | 顔・ポーズ参照用のモデルの読み込みに失敗 |
| `gpu_out_of_memory` | 503 | GPU メモリ不足（サイズ・ステップを下げるよう案内） |
| `queue_timeout` | 503 | 待機時間の上限超過 |
| `generation_failed` | 500 | その他の生成失敗 |

## テスト・静的解析

```powershell
make check            # 以下をすべて実行
```

| 対象 | コマンド | 内容 |
| --- | --- | --- |
| API | `pytest` / `ruff check .` / `mypy`（`backend/`） | 生成処理をフェイクに差し替えるため torch・GPU 不要。入力検証（範囲・型・未知の項目・不正 JSON・カタログ外のモデル/LoRA/サンプラー・系統違いの LoRA・壊れた/巨大な画像・変換強度）、モデルごとの既定値、バッチのシード、PNG/JPEG/WebP 出力、img2img、読み込み中の 503・失敗後の再試行、GPU メモリ不足の判別、内部情報が応答に漏れないこと、待ち行列、カタログ検証、ダウンロード済み判定 |
| 画面 | `npm test` / `npm run lint` / `npm run typecheck`（`frontend/`） | fetch をモックし、生成〜表示〜ダウンロード、送信前の入力検証とフォーカス移動、エラー案内、重複送信防止、モデル切替時の既定値と LoRA の絞り込み、未ダウンロード警告、バッチのサムネイル選択、JPEG 出力、img2img のアップロードと縦横比調整を検証 |

画面の動作確認は、サーバー起動後にブラウザで以下を確認してください：

1. 画面上部が「準備完了」になる
2. プロンプトを入力して「画像を生成」→ 生成中表示と経過秒数が出て、ボタンが無効化される
3. 画像が表示され「PNGをダウンロード」で保存できる
4. 幅に `500` を入れると、送信前に「8の倍数で指定してください」と表示される

## 制約と今後の改善点

- 生成は 1 プロセス 1 件ずつ、GPU 上のモデルも 1 つです。利用者ごとに違うモデルを選ぶと入れ替えが頻発して遅くなります。スループットを上げるには GPU・モデルごとにプロセスを立てて振り分ける、またはジョブキュー（Redis + ワーカー）化が必要です。
- 未ダウンロードのモデルは、そのモデルを最初に選んだリクエストの中でダウンロードされます（画面で警告は出ますが、ダウンロードの進捗率は表示されません）。
- 生成の途中キャンセルや進捗率（ステップ単位）の表示は未対応です（経過秒数のみ）。`callback_on_step_end` と SSE/WebSocket で実装できます。
- 画像は JSON 内の base64 で返すため、PNG 4 枚（1024px）では応答が 10MB 程度になります。
- インペイント（部分修正）には未対応です。ControlNet はポーズ（OpenPose）と InstantID のみです。
- 顔・ポーズ参照は 1 枚あたり 30〜40 秒程度かかります（SDXL + ControlNet 2 つ、848×1240・30 ステップ、RTX 5060 Ti）。
- CLIP のトークン上限（77 トークン）を超えるプロンプトは切り詰められます。
- 認証・レート制限（IP 単位）はありません。外部公開する場合はリバースプロキシ等で追加してください。
- 生成画像はサーバーに保存しません（履歴機能なし）。
