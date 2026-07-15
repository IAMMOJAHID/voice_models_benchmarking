# Voice Models Benchmarking

This project builds a reproducible, local benchmark harness for multilingual open-source TTS pipelines.

## Current status

- A minimal runnable smoke-test pipeline is implemented in [src/benchmark_runner.py](src/benchmark_runner.py).
- It generates placeholder WAV files for English, Arabic, and Hindi prompts and writes a JSON result manifest to the outputs directory.
- The project is ready for the next step: installing heavier TTS and evaluation dependencies and replacing the placeholder generation with real model inference.

## How to run

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python src/benchmark_runner.py
```

## Expected outputs

- WAV files under [outputs](outputs)
- JSON metrics under [outputs/smoke_results.json](outputs/smoke_results.json)
