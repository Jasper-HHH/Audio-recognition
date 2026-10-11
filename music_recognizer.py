#!/usr/bin/env python3
"""Offline song matching against a local audio-and-jianpu library."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SAMPLE_RATE = 22050
FRAME_SIZE = 4096
HOP_SIZE = 5512
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"}


def load_audio(path: Path):
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as error:
        raise RuntimeError(
            "缺少依赖，请先运行: python -m pip install -r requirements.txt"
        ) from error

    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as error:
        raise RuntimeError(f"无法解码音频 {path}: {error}") from error

    mono = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        output_length = round(len(mono) * SAMPLE_RATE / sample_rate)
        old_positions = np.arange(len(mono), dtype=np.float32)
        new_positions = np.arange(output_length, dtype=np.float32) * sample_rate / SAMPLE_RATE
        mono = np.interp(new_positions, old_positions, mono).astype(np.float32)

    if len(mono) < FRAME_SIZE:
        raise RuntimeError(f"音频太短，至少需要 {FRAME_SIZE / SAMPLE_RATE:.1f} 秒: {path}")
    return mono


def extract_chroma(path: Path):
    import numpy as np

    audio = load_audio(path)
    frames = np.lib.stride_tricks.sliding_window_view(audio, FRAME_SIZE)[::HOP_SIZE]
    window = np.hanning(FRAME_SIZE).astype(np.float32)
    frequencies = np.fft.rfftfreq(FRAME_SIZE, 1 / SAMPLE_RATE)
    valid_bins = (frequencies >= 65) & (frequencies <= 5000)
    midi_notes = np.rint(69 + 12 * np.log2(frequencies[valid_bins] / 440)).astype(int)
    pitch_classes = midi_notes % 12

    weights = np.zeros((12, len(pitch_classes)), dtype=np.float32)
    weights[pitch_classes, np.arange(len(pitch_classes))] = 1
    chroma = np.empty((12, len(frames)), dtype=np.float32)

    for start in range(0, len(frames), 256):
        batch = frames[start : start + 256] * window
        magnitudes = np.abs(np.fft.rfft(batch, axis=1)).astype(np.float32)
        chroma[:, start : start + len(batch)] = weights @ magnitudes[:, valid_bins].T

    norms = np.linalg.norm(chroma, axis=0)
    chroma /= np.maximum(norms, 1e-8)
    return chroma, len(audio) / SAMPLE_RATE


def compare_chroma(reference, query):
    import numpy as np

    best_score = -1.0
    for scale in np.linspace(0.8, 1.25, 10):
        frame_count = max(1, round(query.shape[1] * scale))
        positions = np.linspace(0, query.shape[1] - 1, frame_count)
        resized = np.vstack(
            [np.interp(positions, np.arange(query.shape[1]), row) for row in query]
        ).astype(np.float32)
        if frame_count > reference.shape[1]:
            continue

        windows = np.lib.stride_tricks.sliding_window_view(
            reference, frame_count, axis=1
        )
        scores = np.einsum("cpw,cw->pw", windows, resized, optimize=True).mean(axis=1)
        best_score = max(best_score, float(scores.max()))

    return best_score


def find_references(library: Path):
    references = []
    for audio_path in sorted(library.rglob("*")):
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        score_path = audio_path.with_suffix(".jianpu.txt")
        if not score_path.is_file():
            print(f"跳过（缺少简谱文件）: {audio_path}", file=sys.stderr)
            continue
        references.append((audio_path, score_path))
    return references


def main() -> int:
    parser = argparse.ArgumentParser(
        description="用本地参考曲库识别歌曲，并输出对应的简谱文本。"
    )
    parser.add_argument("--input", required=True, type=Path, help="待识别音频文件")
    parser.add_argument("--library", required=True, type=Path, help="本地参考曲库目录")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.62,
        help="最低匹配分数，范围 0 到 1，默认 0.62",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"输入音频不存在: {args.input}")
    if not args.library.is_dir():
        parser.error(f"曲库目录不存在: {args.library}")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold 必须在 0 到 1 之间")

    references = find_references(args.library)
    if not references:
        parser.error("曲库中没有找到同时具备音频和同名 .jianpu.txt 的曲目")

    try:
        query, query_seconds = extract_chroma(args.input)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 2

    if query_seconds < 5:
        print("待识别片段不足 5 秒，匹配结果不可靠。", file=sys.stderr)
        return 2

    best_match = None
    for audio_path, score_path in references:
        try:
            reference, _ = extract_chroma(audio_path)
        except RuntimeError as error:
            print(error, file=sys.stderr)
            continue
        score = compare_chroma(reference, query)
        if best_match is None or score > best_match[0]:
            best_match = (score, audio_path, score_path)

    if best_match is None:
        print("曲库中的参考音频均无法读取。", file=sys.stderr)
        return 2

    score, audio_path, score_path = best_match
    print(f"匹配曲目: {audio_path.stem}")
    print(f"匹配分数: {score:.3f}（不是概率）")
    if score < args.threshold:
        print("低于阈值，未输出简谱；请补充参考曲目或调整阈值。")
        return 1

    print("简谱:")
    print(score_path.read_text(encoding="utf-8").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())