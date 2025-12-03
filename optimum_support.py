import importlib.metadata
import inspect
import re
import subprocess
import sys
from importlib import reload
from pathlib import Path

import optimum.intel.utils.import_utils as import_utils
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError
from packaging.requirements import Requirement

if Path("optimum-intel").is_dir():
    subprocess.run(["git", "pull"], cwd="optimum-intel")
else:
    subprocess.run(["git", "clone", "https://github.com/huggingface/optimum-intel.git"])
test_path = Path(__file__).parent / "optimum-intel" / "tests" / "openvino"
sys.path.append(str(test_path))

# Stable Diffusion does not have a model_type in the config
# Get supported diffusion classes from SUPPORTED_OV_PIPELINES, which lists all OV pipeline wrappers
import optimum.intel.openvino.modeling_diffusion as _modeling_diffusion
# Import the test modules from the cloned repository. This must be imported globally to avoid issues with reloading in Gradio
import test_decoder
import test_diffusion
import test_modeling
import test_seq2seq

SUPPORTED_DIFFUSION_CLASSES = [
    cls.auto_model_class.__name__
    for cls in _modeling_diffusion.SUPPORTED_OV_PIPELINES
    if hasattr(cls, "auto_model_class") and cls.auto_model_class is not None
]


def get_supported_models_for_version(version):
    import_utils._transformers_version = version
    test_seq2seq._transformers_version = version
    test_modeling._transformers_version = version
    test_diffusion._transformers_version = version
    test_decoder._transformers_version = version

    seq2seq = reload(test_seq2seq)
    decoder = reload(test_decoder)
    modeling = reload(test_modeling)
    diffusion = reload(test_diffusion)

    d = {}
    modules = [seq2seq, decoder, modeling, diffusion]
    for mod in modules:
        for name, obj in inspect.getmembers(mod):
            if inspect.isclass(obj):
                if re.match(r"(OVModelFor.*IntegrationTest)", name) or re.match(r"(OVPipelineFor.*Test)", name):
                    task = name.replace("IntegrationTest", "").replace("Test", "")
                    if "CustomTasks" not in task:
                        d[task] = obj.SUPPORTED_ARCHITECTURES
    all_archs = []
    for archs in d.values():
        all_archs += archs
    return sorted(set(all_archs))


def get_min_max_transformers():
    meta = importlib.metadata.metadata("optimum-intel")
    requires = meta.get_all("Requires-Dist") or []
    transformers_versions = [item for item in requires if "transformers" in item and "extra" not in item][0]
    req = Requirement(transformers_versions)
    maxver, minver = [ver.version for ver in list(req.specifier)]
    return (minver, maxver)


def show_is_supported(model_id):
    print(f"Checking {model_id}...")
    minver, maxver = get_min_max_transformers()
    versions = [minver, "4.53.0", maxver]

    all_supported_models = set()
    for v in versions:
        archs = get_supported_models_for_version(v)
        all_supported_models.update(archs)
    try:
        model_info = HfApi().model_info(model_id)
    except RepositoryNotFoundError:
        message = f"Model {model_id} was not found on the Hugging Face hub. Make sure you entered the correct model_id. If the model requires authentication, use `hf auth login` or a token to authenticate."
    else:
        if (model_info.config is not None) and model_info.config != {}:
            model_type = model_info.config.get("model_type")
            if model_type is None:  # Check for diffusion class if model_type is not available
                class_name = model_info.config.get("diffusers", {}).get("_class_name")
                if class_name in SUPPORTED_DIFFUSION_CLASSES:
                    message = (
                        f"`{model_id}` with diffusion class `{class_name}` is **supported** by optimum-intel[openvino]."
                    )
                else:
                    message = f"`{model_id}` is not in the list of supported architectures by optimum-intel[openvino]. It is **likely not supported**, but it is wise to doublecheck"
            elif model_type in all_supported_models:
                message = f"`{model_id}` with model type `{model_type}` is **supported** by optimum-intel[openvino]."
            else:
                message = f"`{model_id}` with model type `{model_type}` is not in the list of supported architectures by optimum-intel[openvino]. It is **likely not supported**, but it is wise to doublecheck"
        else:
            message = f"`{model_id}` is **not supported** by optimum-intel[openvino]."
    print(f"Using transformers: {versions}. Total number of supported architectures: {len(all_supported_models)}")
    print(message)
    return message


if __name__ == "__main__":
    show_is_supported(sys.argv[1])
