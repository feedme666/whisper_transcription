# Whisper Transcription Tool

Whisper Transcription Tool は、OpenAI Whisper を利用した日本語対応の文字起こしアプリケーションです。Streamlit 製の Web UI とコマンドラインツールを備えているため、音声ファイルを気軽にテキストへ変換できます。

## 特徴

- **Streamlit アプリ (`src/app.py`)**  
  ブラウザ上で Whisper モデルのサイズや対象言語を選択し、音声ファイルをアップロードするだけで文字起こしが可能です。セグメント単位のタイムスタンプ一覧やテキストダウンロード機能も備えています。

- **詳細オプション版 (`src/app_advanced.py`)**  
  温度、ビーム幅、初期プロンプトなど Whisper の主要パラメータを調整可能です。専門的な用途やチューニングが必要な場合に利用してください。

- **CLI ツール (`src/transcribe.py`)**  
  スクリプトやバッチ処理から利用する場合に便利です。`--device auto` で CUDA / MPS / CPU を自動判定し、利用可能なデバイスで実行します（MPS は一部演算で CPU フォールバックします）。

## 動作環境

- Python 3.10 以上を推奨
- macOS (Apple Silicon) / Linux / Windows
- GPU を利用する場合は CUDA もしくは MPS 対応の PyTorch が必要です

## インストール

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows の場合は .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

WhisperX など追加機能を使う場合は、必要に応じて別途依存ライブラリを導入してください。

## 使い方

### Streamlit アプリ

```bash
streamlit run src/app.py
```

または高度な設定版:

```bash
streamlit run src/app_advanced.py
```

ブラウザが開いたら、サイドバーからモデルと言語を選び、音声ファイルをアップロードしてください。

### CLI ツール

```bash
python src/transcribe.py --file path/to/audio.m4a --model base --output result.txt
```

主なオプション:

- `--model`: `tiny`, `base`, `small`, `medium`, `large` から選択（デフォルト: `base`）
- `--language`: 音声言語コード（例: `ja`, `en`）。指定しない場合は自動判定
- `--device`: `auto` / `cpu` / `cuda` / `mps`。`auto` は CUDA → MPS → CPU の順で利用可能なデバイスを選択します

## Hugging Face Spaces へのデプロイ

Streamlit アプリを Hugging Face で公開する場合は、`app_advanced.py` を `app.py` にリネームしてアップロードし、`requirements.txt` も一緒に配置します。Space 作成後は Git push もしくは Web UI でファイルをアップロードしてください。

## ライセンス

本プロジェクトは MIT License の下で公開されています。
