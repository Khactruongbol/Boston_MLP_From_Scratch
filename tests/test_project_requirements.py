from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "99_boston_housing_model_images_report.ipynb"


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
        "notebooks/99_boston_housing_model_images_report.ipynb",
        "requirements.txt",
        "README.md",
    ]
    for relative_path in required_paths:
        assert (PROJECT_ROOT / relative_path).exists(), relative_path


def test_report_notebook_has_only_markdown_and_existing_images():
    notebook = json.loads(REPORT_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    assert cells
    assert all(cell["cell_type"] == "markdown" for cell in cells)

    image_links: list[Path] = []
    for cell in cells:
        source = "".join(cell.get("source", []))
        for match in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", source):
            image_links.append((REPORT_NOTEBOOK_PATH.parent / match).resolve())

    assert len(image_links) >= 8
    missing = [path for path in image_links if not path.exists()]
    assert not missing


def test_report_notebook_contains_required_vietnamese_sections():
    notebook = json.loads(REPORT_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    report_text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    required_terms = [
        "Define problem",
        "Định nghĩa bài toán",
        "Hướng giải quyết",
        "Sử dụng những gì",
        "Data lấy từ đâu",
        "cách lấy data",
        "Cách lọc data sạch",
        "Số lượng feature: **13**",
        "Hình ảnh sau khi train",
        "Kết luận model",
    ]
    for term in required_terms:
        assert term in report_text
    assert "\ufffd" not in report_text
