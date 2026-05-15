from pathlib import Path
import re
import tomllib

from packaging.requirements import Requirement
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]
TRANSFORMERS_CUSTOM_OP_BREAKAGE_START = Version("4.56")


def _docker_torch_version() -> Version:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    match = re.search(r"^FROM pytorch/pytorch:(\d+\.\d+\.\d+)-", dockerfile, re.MULTILINE)
    assert match, "Dockerfile must pin a pytorch/pytorch base image version"
    return Version(match.group(1))


def _requirement_for(name: str, requirements: list[str]) -> Requirement:
    for raw_requirement in requirements:
        requirement = Requirement(raw_requirement)
        if requirement.name == name:
            return requirement

    raise AssertionError(f"Missing dependency constraint for {name}")


def _assert_transformers_is_capped_for_torch_24(requirements: list[str]) -> None:
    if _docker_torch_version() >= Version("2.5"):
        return

    requirement = _requirement_for("transformers", requirements)
    assert requirement.specifier.contains("4.55.999")
    assert not requirement.specifier.contains(TRANSFORMERS_CUSTOM_OP_BREAKAGE_START)


def _assert_sentence_transformers_supports_onnx(requirements: list[str]) -> None:
    requirement = _requirement_for("sentence-transformers", requirements)
    assert "onnx" in requirement.extras


def test_requirements_cap_transformers_for_pytorch_24_image() -> None:
    requirements = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    _assert_transformers_is_capped_for_torch_24(requirements)
    _assert_sentence_transformers_supports_onnx(requirements)


def test_pyproject_caps_transformers_for_pytorch_24_image() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = pyproject["project"]["dependencies"]

    _assert_transformers_is_capped_for_torch_24(requirements)
    _assert_sentence_transformers_supports_onnx(requirements)
