import pytest

from optimum_support import show_is_supported

# (model_id, expected_substring)
test_cases = [
    ("openai/whisper-small", "is **supported**"),
    ("Ultralytics/YOLO11", "is **not supported**"),
    ("test/non-existing", "was not found on the Hugging Face hub"),
    ("openai/gpt-oss-20b", "is **supported**"),
    ("ibm-granite/granite-3.1-8b-instruct", "is **supported**"),
    ("stabilityai/stable-diffusion-xl-base-1.0", "is **supported**"),
    ("microsoft/Phi-4-multimodal-instruct", "is **supported**"),
    ("google-bert/bert-base-uncased", "is **supported**"),
    ("rednote-hilab/dots.ocr", "is not in the list of supported architectures"),
    ("LiquidAI/LFM2-350M", "is **supported**"),
    ("google/mobilenet_v2_1.0_224", "is **supported**"),
    ("stabilityai/stable-diffusion-3.5-large", "is **supported**"),
    ("stabilityai/sp4d", "is **not supported**"),
    ("SimianLuo/LCM_Dreamshaper_v7", "is **supported**"),
    ("stabilityai/sd-x2-latent-upscaler", "is not in the list"),
    ("openbmb/MiniCPM3-4B", "is **supported**"),
    ("Efficient-Large-Model/SANA-Video_2B_480p", "is **not supported**"),
    ("optimum-intel-internal-testing/tiny-random-sana-sprint", "is **supported**"),
]


@pytest.mark.parametrize("model_id,expected", test_cases)
def test_show_is_supported(model_id, expected):
    result = show_is_supported(model_id)
    assert expected in result, f"For {model_id}, expected '{expected}' in result, got: {result}"
