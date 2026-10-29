import importlib.util
from pathlib import Path
import sys


def load_exporter_module():
    scripts_dir = Path(__file__).parent
    sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "export_libero90_retrieval_benchmark_xlsx",
        scripts_dir / "export_libero90_retrieval_benchmark_xlsx.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retrieval_benchmark_xlsx_is_valid_and_contains_expected_sheets(tmp_path):
    exporter = load_exporter_module()
    path = tmp_path / "retrieval.xlsx"
    workbook = {
        "Summary": [["Method", "Score"], ["dino_top", 0.2]],
        "PerTask": [["Task", "dino_top"], [20, 0.2]],
        "Pairwise": [["Method", "Delta"], ["state", -0.1]],
        "Queries": [["Task", "Retrieved"], [20, "1:10"]],
        "Calibration": [["Beta", 1.0]],
        "Artifacts": [["Checkpoint", "normal/9999"]],
    }

    exporter.write_xlsx(path, workbook)

    exporter.validate_xlsx(path, expected_sheets=set(workbook))
