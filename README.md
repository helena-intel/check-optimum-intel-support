---
title: Check Optimum Intel Support
emoji: 🌖
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 6.14.0
python_version: 3.12.13
app_file: app.py
pinned: false
short_description: Check if a model is supported by optimum-intel
---

# Check Optimum Intel Support

A Gradio app (hosted on [Hugging Face Spaces](https://huggingface.co/spaces/helenai/check-optimum-intel-support)) that checks whether a given Hugging Face model is supported by [optimum-intel](https://github.com/huggingface/optimum-intel).

## How it works

Enter a model ID (e.g. `openai/whisper-small`) and the app will:

1. Clone the [optimum-intel](https://github.com/huggingface/optimum-intel) test suite
2. Mock multiple `transformers` versions to collect all supported architectures across the compatibility range
3. Look up the model's `model_type` or diffusion class on the Hub and check it against the supported list

## Running the Gradio app locally

```bash
pip install -r requirements.txt
pip install gradio
python app.py
```

## Running the Python script directly

Replace `openai/whisper-small` with the model you want to check.

```bash
pip install -r requirements.txt
python optimum_support.py openai/whisper-small
```

## Tests

```bash
pip install pytest
python -m pytest test_optimum_support.py
```

## Project structure

- `app.py` - Gradio interface
- `optimum_support.py` - Core logic: version mocking, architecture collection, Hub lookup
- `test_optimum_support.py` - Parametrized tests against known models
- `optimum-intel/` - Cloned at runtime (git-ignored), provides the test modules that define supported architectures per version
