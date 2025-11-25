from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from jiwer import cer, wer
from src.inference.pipeline import FlowInferencePipeline


class ManifestEvaluator:
    """Evaluate WER/CER over a JSONL manifest."""

    def __init__(
        self,
        pipeline: FlowInferencePipeline,
        manifest_path: str,
        output_path: Optional[str] = None,
    ):
        self.pipeline = pipeline
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        self.output_path = Path(output_path) if output_path else None

    def run(self) -> Dict[str, float]:
        references: list[str] = []
        predictions: list[str] = []

        output_file = (
            self.output_path.open("w", encoding="utf-8") if self.output_path is not None else None
        )

        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                audio_path = record["audio_path"]
                reference = record["text"]
                language = record.get("language")

                result = self.pipeline(audio_path, language=language)
                hypothesis = result["text"]

                references.append(reference)
                predictions.append(hypothesis)

                if output_file:
                    output_file.write(
                        json.dumps(
                            {
                                "audio_path": audio_path,
                                "reference": reference,
                                "prediction": hypothesis,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        if output_file:
            output_file.close()

        return {
            "wer": wer(references, predictions),
            "cer": cer(references, predictions),
        }
