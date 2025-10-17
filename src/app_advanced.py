#!/usr/bin/env python3
"""
Whisper文字起こしWebアプリ（詳細設定版）
"""

import os
import sys
import time
import tempfile
import inspect
from functools import lru_cache
from datetime import datetime

import streamlit as st
import torch
import whisper

# ページ設定
st.set_page_config(
    page_title="Whisper文字起こしツール（詳細版）",
    page_icon="🎛️",
    layout="wide"
)


@st.cache_resource
def load_whisper_model(model_name: str):
    """Whisperモデルをロードする（キャッシュ使用）"""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return whisper.load_model(model_name, device=device)


def check_ffmpeg():
    """FFmpegがインストールされているか確認"""
    if os.system("ffmpeg -version > /dev/null 2>&1") != 0:
        st.error(
            "⚠️ FFmpegがインストールされていません。https://ffmpeg.org/download.html からダウンロードしてください。"
        )
        st.stop()


def get_available_models():
    """利用可能なWhisperモデルの一覧を返す"""
    return ["tiny", "base", "small", "medium", "large"]


@lru_cache(maxsize=1)
def decoding_option_names():
    """利用可能なDecodingOptionsのキーワードを取得"""
    try:
        return set(inspect.signature(whisper.decoding.DecodingOptions.__init__).parameters.keys())
    except (AttributeError, ValueError):
        return set()


@lru_cache(maxsize=1)
def transcribe_option_names():
    """Whisper.transcribe が受け付けるキーワードを取得"""
    try:
        transcribe_fn = whisper.model.Whisper.transcribe  # type: ignore[attr-defined]
        params = set(inspect.signature(transcribe_fn).parameters.keys())
        params.discard("self")
        return params
    except (AttributeError, ValueError):
        return set()


def main():
    """メイン関数"""
    st.title("🎛️ Whisper文字起こしツール（詳細設定版）")
    st.markdown(
        """
        OpenAI Whisper の高度なオプションを調整しながら、音声ファイルからテキストへの文字起こしを行います。
        """
    )

    # FFmpeg確認
    check_ffmpeg()

    # サイドバー設定
    st.sidebar.title("設定")

    # モデル選択
    model_option = st.sidebar.selectbox(
        "モデルサイズを選択",
        options=get_available_models(),
        index=1,
        help="モデルが大きいほど精度は上がりますが、処理時間も増えます。"
    )

    # 言語選択
    language_option = st.sidebar.selectbox(
        "言語を選択（自動検出する場合は空欄）",
        options=["", "en", "ja", "zh", "de", "fr", "es", "ko", "ru"],
        index=0,
        format_func=lambda x: {
            "": "自動検出", "en": "英語", "ja": "日本語", "zh": "中国語",
            "de": "ドイツ語", "fr": "フランス語", "es": "スペイン語",
            "ko": "韓国語", "ru": "ロシア語"
        }.get(x, x),
        help="音声の言語を指定します。自動検出も可能です。"
    )

    # タスク選択
    task_option = st.sidebar.radio(
        "タスクを選択",
        options=["transcribe", "translate"],
        format_func=lambda x: "文字起こし（同じ言語）" if x == "transcribe" else "翻訳（英語）",
        help="translate を選ぶと英語への翻訳結果が出力されます。"
    )

    st.sidebar.markdown("---")

    # 詳細設定
    with st.sidebar.expander("詳細オプション", expanded=False):
        temperature = st.slider(
            "temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            help="サンプリング温度。0.0で決定的、値を上げると多様性が増えます。"
        )
        temperature_increment = st.slider(
            "temperature_increment_on_fallback",
            min_value=0.0,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="デコード失敗時に温度を上げる量。"
        )
        best_of = st.number_input(
            "best_of（サンプリング回数）",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            help="温度 > 0 のとき有効。候補を複数生成してベストを選びます。"
        )
        beam_size = st.number_input(
            "beam_size（ビーム幅）",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="ビームサーチの幅。高いほど精度は上がりますが遅くなります。"
        )
        patience = st.slider(
            "patience",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="ビームサーチをどの程度続けるか。1.0がデフォルト。"
        )
        length_penalty = st.slider(
            "length_penalty",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="1.0で無効。高くすると長い出力が抑制されます。"
        )
        condition_on_previous_text = st.checkbox(
            "condition_on_previous_text（文脈を引き継ぐ）",
            value=True,
            help="長時間ファイルで誤認識が多い場合、オフにすると改善することがあります。"
        )
        vad_filter = st.checkbox(
            "vad_filter（無音部分を除去）",
            value=False,
            help="Voice Activity Detection を使い、長い無音をスキップします。"
        )
        word_timestamps = st.checkbox(
            "word_timestamps（単語ごとのタイムスタンプ）",
            value=False,
            help="セグメント内の単語単位でタイムスタンプを取得します。処理が遅くなる場合があります。"
        )
        initial_prompt = st.text_area(
            "initial_prompt / prompt",
            value="",
            help="固有名詞や文体など、モデルに覚えておいてほしい文脈を入力します。"
        )
        compression_ratio_threshold = st.slider(
            "compression_ratio_threshold",
            min_value=1.0,
            max_value=5.0,
            value=2.4,
            step=0.1,
            help="ハフマン圧縮で高すぎる場合は失敗として扱います。"
        )
        logprob_threshold = st.slider(
            "logprob_threshold",
            min_value=-5.0,
            max_value=0.0,
            value=-1.0,
            step=0.1,
            help="平均対数確率がこの値を下回ると失敗として扱います。"
        )
        no_speech_threshold = st.slider(
            "no_speech_threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="無音判定の閾値。高くすると無音扱いされやすくなります。"
        )
        fp16 = st.checkbox(
            "fp16（半精度）",
            value=True,
            help="GPU での半精度計算を有効化。CPU/MPSで問題がある場合はオフにします。"
        )

    # デバイス情報表示
    if torch.cuda.is_available():
        device_for_display = "GPU (CUDA)"
    elif torch.backends.mps.is_available():
        device_for_display = "Apple Silicon (MPS)"
    else:
        device_for_display = "CPU"
    st.sidebar.info(f"使用デバイス: {device_for_display}")

    if device_for_display == "CPU":
        st.sidebar.warning("GPUが検出されませんでした。処理が遅くなる可能性があります。")
    elif device_for_display == "Apple Silicon (MPS)":
        st.sidebar.warning("Apple Silicon の MPS サポートは限定的です。問題が発生する場合があります。")

    st.sidebar.markdown("---")
    st.sidebar.markdown("[GitHubリポジトリ](https://github.com/yourusername/whisper-transcription)")

    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "音声ファイルをアップロード",
        type=["mp3", "wav", "m4a", "ogg", "flac"],
        help="対応フォーマット: MP3, WAV, M4A, OGG, FLAC"
    )

    if uploaded_file is None:
        st.info("👆 音声ファイルをアップロードしてください")
        with st.expander("使い方"):
            st.markdown(
                """
                1. サイドバーでモデル・言語・タスク・オプションを設定
                2. 音声ファイルをアップロード
                3. 「文字起こし開始」ボタンをクリック
                4. 結果を確認し、必要に応じてダウンロード
                """
            )
        return

    # ファイル情報表示
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.info(f"ファイル: {uploaded_file.name} ({file_size_mb:.2f} MB)")

    # 音声再生
    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")

    # 文字起こしボタン
    if not st.button("文字起こし開始", type="primary"):
        return

    # 処理開始
    with st.spinner("文字起こし処理中..."):
        # 一時ファイルとして保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_filename = tmp_file.name

        try:
            # モデルロード
            load_start = time.time()
            status_placeholder = st.empty()
            status_placeholder.text("モデルをロード中...")
            model = load_whisper_model(model_option)
            load_end = time.time()
            status_placeholder.text(f"モデルロード完了（{load_end - load_start:.2f}秒）")

            # オプション構築
            transcribe_options = {
                "temperature": temperature,
                "best_of": best_of,
                "beam_size": beam_size,
                "patience": patience,
                "length_penalty": length_penalty,
                "condition_on_previous_text": condition_on_previous_text,
                "vad_filter": vad_filter,
                "word_timestamps": word_timestamps,
                "compression_ratio_threshold": compression_ratio_threshold,
                "logprob_threshold": logprob_threshold,
                "no_speech_threshold": no_speech_threshold,
                "task": task_option,
                "fp16": fp16,
            }

            if "temperature_increment_on_fallback" in decoding_option_names():
                transcribe_options["temperature_increment_on_fallback"] = temperature_increment
            elif temperature_increment > 0 and temperature < 1.0:
                # 未対応環境では温度の候補をリスト指定してフォールバックを模倣
                steps = int((1.0 - temperature) / temperature_increment) + 1
                temp_candidates = [
                    min(1.0, round(temperature + i * temperature_increment, 4))
                    for i in range(max(1, steps))
                ]
                # 順序を保ったまま重複を除去
                transcribe_options["temperature"] = tuple(dict.fromkeys(temp_candidates))

            if language_option:
                transcribe_options["language"] = language_option
            if initial_prompt.strip():
                transcribe_options["initial_prompt"] = initial_prompt.strip()

            allowed_transcribe_keys = transcribe_option_names()
            if allowed_transcribe_keys:
                transcribe_options = {
                    key: value for key, value in transcribe_options.items()
                    if key in allowed_transcribe_keys
                }

            # 文字起こし実行
            status_placeholder.text("文字起こし処理中...")
            transcribe_start = time.time()
            result = model.transcribe(temp_filename, **transcribe_options)
            transcribe_end = time.time()
            status_placeholder.empty()

            # 時間計算
            transcribe_time = transcribe_end - transcribe_start
            total_time = transcribe_end - load_start

            # 結果表示
            st.markdown("### 文字起こし結果")
            st.success(f"処理完了（文字起こし: {transcribe_time:.2f}秒、合計: {total_time:.2f}秒）")

            st.markdown("#### テキスト")
            st.text_area(
                label="",
                value=result.get("text", ""),
                height=200
            )

            st.download_button(
                label="テキストをダウンロード",
                data=result.get("text", ""),
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.txt",
                mime="text/plain"
            )

            with st.expander("詳細（タイムスタンプ付き）"):
                table_data = []
                timestamp_text = ""

                for segment in result.get("segments", []):
                    start_time = segment.get("start", 0.0)
                    end_time = segment.get("end", 0.0)
                    text = segment.get("text", "")

                    start_formatted = datetime.utcfromtimestamp(start_time).strftime("%H:%M:%S.%f")[:-3]
                    end_formatted = datetime.utcfromtimestamp(end_time).strftime("%H:%M:%S.%f")[:-3]

                    table_data.append({
                        "開始": start_formatted,
                        "終了": end_formatted,
                        "テキスト": text
                    })

                    timestamp_text += f"[{start_formatted} --> {end_formatted}] {text}\n"

                    if word_timestamps and segment.get("words"):
                        timestamp_text += "".join(
                            f"    ({datetime.utcfromtimestamp(word['start']).strftime('%H:%M:%S.%f')[:-3]}"
                            f" - {datetime.utcfromtimestamp(word['end']).strftime('%H:%M:%S.%f')[:-3]}) {word['word']}\n"
                            for word in segment["words"]
                        )

                if table_data:
                    st.table(table_data)
                else:
                    st.info("セグメント情報がありません。")

                st.download_button(
                    label="タイムスタンプ付きテキストをダウンロード",
                    data=timestamp_text,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript_timestamps.txt",
                    mime="text/plain"
                )

            if word_timestamps:
                with st.expander("単語ごとのタイムスタンプ"):
                    if any(segment.get("words") for segment in result.get("segments", [])):
                        word_lines = []
                        for segment in result.get("segments", []):
                            for word in segment.get("words", []):
                                start_fmt = datetime.utcfromtimestamp(word["start"]).strftime("%H:%M:%S.%f")[:-3]
                                end_fmt = datetime.utcfromtimestamp(word["end"]).strftime("%H:%M:%S.%f")[:-3]
                                word_lines.append(f"{start_fmt} - {end_fmt}: {word['word']}")
                        st.text_area("単語タイムスタンプ", "\n".join(word_lines), height=200)
                    else:
                        st.info("単語タイムスタンプ情報は取得できませんでした。")

        except Exception as exc:
            st.error(f"エラーが発生しました: {exc}")
        finally:
            if "temp_filename" in locals() and os.path.exists(temp_filename):
                os.unlink(temp_filename)


if __name__ == "__main__":
    main()
