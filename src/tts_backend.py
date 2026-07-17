from typing import Dict


_LANGUAGE_MODEL_MAP: Dict[str, str] = {
    "en": "facebook/mms-tts-eng",
    "ar": "facebook/mms-tts-ara",
    "hi": "facebook/mms-tts-hin",
}


def get_tts_model_for_language(language: str) -> str:
    normalized_language = language.lower()
    if normalized_language not in _LANGUAGE_MODEL_MAP:
        raise ValueError(f"No TTS model configured for language '{language}'.")
    return _LANGUAGE_MODEL_MAP[normalized_language]
