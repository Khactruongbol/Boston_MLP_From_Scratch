from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_does_not_use_removed_load_boston_api():
    source = "\n".join(path.read_text(encoding="utf-8") for path in (PROJECT_ROOT / "src").glob("*.py"))
    assert "from sklearn.datasets" not in source
    assert "load_boston(" not in source


def test_required_project_files_exist():
    required_paths = [
        "src/data.py",
        "src/modeling.py",
        "src/train.py",
        "app.py",
        "requirements.txt",
        "README.md",
    ]
    for relative_path in required_paths:
        assert (PROJECT_ROOT / relative_path).exists(), relative_path

