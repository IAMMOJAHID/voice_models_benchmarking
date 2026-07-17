import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tts_backend import get_tts_model_for_language


class TTSBackendTests(unittest.TestCase):
    def test_get_tts_model_for_language_uses_supported_mapping(self) -> None:
        self.assertEqual(get_tts_model_for_language("en"), "facebook/mms-tts-eng")
        self.assertEqual(get_tts_model_for_language("ar"), "facebook/mms-tts-ara")
        self.assertEqual(get_tts_model_for_language("hi"), "facebook/mms-tts-hin")

    def test_get_tts_model_for_language_raises_for_unknown_language(self) -> None:
        with self.assertRaises(ValueError):
            get_tts_model_for_language("fr")


if __name__ == "__main__":
    unittest.main()
