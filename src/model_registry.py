from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelSpec:
    name: str
    family: str
    description: str
    multilingual: bool
    supports_cloning: bool
    notes: Optional[str] = None


MODEL_REGISTRY = [
    ModelSpec(
        name="xtts_v2",
        family="coqui",
        description="Multilingual XTTS v2 model with voice cloning support.",
        multilingual=True,
        supports_cloning=True,
        notes="Best fit for speaker similarity experiments if runtime permits.",
    ),
    ModelSpec(
        name="mms_tts",
        family="facebook",
        description="MMS TTS for multilingual speech synthesis.",
        multilingual=True,
        supports_cloning=False,
        notes="Useful as an open-source multilingual baseline.",
    ),
    ModelSpec(
        name="piper",
        family="rhasspy",
        description="Lightweight local TTS engine with low-latency inference.",
        multilingual=False,
        supports_cloning=False,
        notes="Good for speed/latency trade-off analysis.",
    ),
]
