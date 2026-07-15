import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf

from evaluation import build_evaluation_metrics, evaluate_with_whisper
from model_registry import MODEL_REGISTRY


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_manifest() -> List[Dict[str, Any]]:
    return [
        {
            "id": "en_001",
            "language": "en",
            "text": "Hello from a local multilingual speech benchmark.",
            "reference_audio": None,
        },
        {
            "id": "ar_001",
            "language": "ar",
            "text": "مرحبا من مقاييس توليد الصوت المحلية.",
            "reference_audio": None,
        },
        {
            "id": "hi_001",
            "language": "hi",
            "text": "स्थानीय भाषाई आवाज़ बेंचमार्क से नमस्ते।",
            "reference_audio": None,
        },
    ]


def run_smoke_test() -> Dict[str, Any]:
    manifest = create_manifest()
    results: List[Dict[str, Any]] = []
    for item in manifest:
        start = time.perf_counter()
        sr = 22050
        audio = np.zeros(int(sr * 0.8), dtype=np.float32)
        out_path = OUTPUT_DIR / f"{item['id']}_smoke.wav"
        sf.write(out_path, audio, sr)
        elapsed = time.perf_counter() - start
        metrics = build_evaluation_metrics(str(out_path))
        transcript_result = evaluate_with_whisper(str(out_path), model_name="openai/whisper-tiny")
        results.append(
            {
                "id": item["id"],
                "language": item["language"],
                "text": item["text"],
                "output_audio": str(out_path),
                "latency_seconds": round(elapsed, 4),
                "duration_seconds": round(len(audio) / sr, 4),
                "status": "generated",
                "evaluation": metrics,
                "transcript": transcript_result,
            }
        )

    summary = {
        "generated_count": len(results),
        "results": results,
        "models": [model.name for model in MODEL_REGISTRY],
        "hardware": {
            "note": "The benchmark uses local CPU inference and generated placeholder waveforms for the first verified run.",
        },
    }
    with (OUTPUT_DIR / "smoke_results.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    run_smoke_test()
    print("Smoke test completed. Results saved to outputs/.")
