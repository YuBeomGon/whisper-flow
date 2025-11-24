#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create JSONL manifests for LibriSpeech splits")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to LibriSpeech split directory (e.g., data/raw/librispeech/train-clean-100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL manifest path",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code stored in manifest entries",
    )
    parser.add_argument(
        "--audio-extension",
        type=str,
        default=".flac",
        help="Audio file extension to expect for each utterance",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root path not found: {root}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total, written = 0, 0
    with output_path.open("w", encoding="utf-8") as out_file:
        for transcript_path in root.rglob("*.txt"):
            with transcript_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ", 1)
                    if len(parts) != 2:
                        continue
                    utt_id, transcript = parts
                    audio_path = transcript_path.with_name(utt_id + args.audio_extension)
                    total += 1
                    if not audio_path.exists():
                        continue
                    record = {
                        "audio_path": str(audio_path.resolve()),
                        "text": transcript,
                        "language": args.language,
                    }
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
    print(f"Wrote {written} / {total} entries to {output_path}")


if __name__ == "__main__":
    main()
