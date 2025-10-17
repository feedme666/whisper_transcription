#!/usr/bin/env python3
"""
Whisper文字起こしコマンドラインツール
"""

import os
import sys
import time
import argparse
import whisper
import torch

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Whisper文字起こしツール")
    parser.add_argument("--file", required=True, help="文字起こしを行う音声ファイルへのパス")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="使用するWhisperモデルのサイズ")
    parser.add_argument("--language", help="音声の言語（例: ja, en）。指定しない場合は自動検出")
    parser.add_argument("--output", help="出力テキストファイルのパス。指定しない場合は標準出力")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="推論デバイスを指定します。auto は利用可能なデバイスから自動選択します。",
    )

    args = parser.parse_args()

    # ファイルの存在確認
    if not os.path.exists(args.file):
        print(f"エラー: ファイル '{args.file}' が見つかりません。", file=sys.stderr)
        return 1

    # デバイス選択
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            if torch.backends.mps.is_available():
                print("情報: MPS はサポートされていない演算があるため自動選択から除外しました。必要であれば --device mps を明示してください。", file=sys.stderr)
    else:
        device = args.device

    # MPS がサポート外の組み合わせでは CPU にフォールバック
    if device == "mps" and not torch.backends.mps.is_available():
        print("警告: MPS が利用できないため CPU を使用します。", file=sys.stderr)
        device = "cpu"

    print(f"モデル '{args.model}' をロード中... (device={device})", file=sys.stderr)
    start_time = time.time()

    # モデルのロード
    try:
        model = whisper.load_model(args.model, device=device)
    except RuntimeError as exc:
        if device == "mps":
            print(f"警告: MPS デバイスでロードできませんでした ({exc}). CPU に切り替えます。", file=sys.stderr)
            device = "cpu"
            model = whisper.load_model(args.model, device=device)
        else:
            raise

    print(f"モデルロード完了（{time.time() - start_time:.2f}秒）", file=sys.stderr)
    print("文字起こし処理中...", file=sys.stderr)

    # 文字起こしオプション
    options = {}
    if args.language:
        options["language"] = args.language

    # 文字起こし実行
    try:
        result = model.transcribe(args.file, **options)
    except RuntimeError as exc:
        if device == "mps" and "MPS" in str(exc):
            print("警告: MPS 実行で未対応の演算が発生したため CPU にフォールバックします。", file=sys.stderr)
            device = "cpu"
            model = whisper.load_model(args.model, device=device)
            result = model.transcribe(args.file, **options)
        else:
            raise

    # 結果の出力
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print(f"文字起こし結果を '{args.output}' に保存しました。", file=sys.stderr)
    else:
        print("\n" + "="*80, file=sys.stderr)
        print("文字起こし結果:", file=sys.stderr)
        print("="*80, file=sys.stderr)
        print(result["text"])

    print(f"\n処理時間: {time.time() - start_time:.2f}秒", file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(main())
