import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
from transformers import pipeline

from evaluation import build_evaluation_metrics, evaluate_with_whisper
from model_registry import MODEL_REGISTRY
from tts_backend import get_tts_model_for_language


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
    pipe_cache: Dict[str, Any] = {}
    for item in manifest:
        start = time.perf_counter()
        model_name = get_tts_model_for_language(item["language"])
        if model_name not in pipe_cache:
            pipe_cache[model_name] = pipeline("text-to-speech", model=model_name, device=-1)
        pipe = pipe_cache[model_name]
        output = pipe(item["text"])
        audio = np.asarray(output["audio"], dtype=np.float32).reshape(-1)
        out_path = OUTPUT_DIR / f"{item['id']}_smoke.wav"
        sf.write(str(out_path), audio, output["sampling_rate"])
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
                "duration_seconds": round(len(audio) / output["sampling_rate"], 4),
                "status": "generated",
                "evaluation": metrics,
                "transcript": transcript_result,
                "backend": model_name,
            }
        )

    summary = {
        "generated_count": len(results),
        "results": results,
        "models": [model.name for model in MODEL_REGISTRY],
        "hardware": {
            "note": "The benchmark uses local CPU inference with Hugging Face MMS TTS models for English, Arabic, and Hindi.",
        },
    }
    with (OUTPUT_DIR / "smoke_results.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    run_smoke_test()
    print("Smoke test completed. Results saved to outputs/.")
