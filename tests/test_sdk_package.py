"""Regression check for a wheel that previously contained only metadata."""

from pathlib import Path
import subprocess
import zipfile


def test_sdk_wheel_contains_importable_package(tmp_path: Path) -> None:
    """Build the distribution and verify the public import path is shipped."""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["uv", "build", str(root / "llm_sdk"), "--wheel",
         "--out-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        code = wheel.read("llm_sdk/__init__.py").decode("utf-8")
    assert "class Small_LLM_Model:" in code
