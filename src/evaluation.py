from typing import Any, Dict

import librosa
import numpy as np
import soundfile as sf
from transformers import pipeline


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def compute_duration_seconds(path: str) -> float:
    audio, sr = load_audio(path)
    return round(len(audio) / sr, 4)


def compute_rms(path: str) -> float:
    audio, _ = load_audio(path)
    return float(np.sqrt(np.mean(np.square(audio))))


def compute_spectral_centroid(path: str) -> float:
    audio, sr = load_audio(path)
    centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
    return float(np.mean(centroids))


def build_evaluation_metrics(audio_path: str) -> Dict[str, Any]:
    duration = compute_duration_seconds(audio_path)
    rms = compute_rms(audio_path)
    centroid = compute_spectral_centroid(audio_path)
    return {
        "duration_seconds": duration,
        "rms": round(rms, 6),
        "spectral_centroid": round(centroid, 6),
    }


def evaluate_with_whisper(audio_path: str, model_name: str = "openai/whisper-tiny") -> Dict[str, Any]:
    try:
        audio, sr = load_audio(audio_path)
        pipe = pipeline("automatic-speech-recognition", model=model_name, device=-1)
        result = pipe({"raw": audio, "sampling_rate": sr}, max_new_tokens=50)
        return {
            "transcript": result.get("text", ""),
            "model": model_name,
            "status": "ok",
        }
    except Exception as exc:  # pragma: no cover - runtime fall-back
        return {
            "transcript": "",
            "model": model_name,
            "status": "error",
            "error": str(exc),
        }
