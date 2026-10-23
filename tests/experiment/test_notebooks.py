import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB_DIR = ROOT / "experiment" / "clef_unseen_attack" / "notebooks"


def _sources(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_probe_notebook_covers_all_four_stages():
    text = _sources(NB_DIR / "clef_unseen_attack_probe.ipynb")
    for needle in ("run_stage0", "run_stage1", "build_datasets", "train_one", "evaluate"):
        assert needle in text, needle
    assert "src.clef_probe" not in text


def test_results_notebook_is_model_free():
    text = _sources(NB_DIR / "clef_stage0_results.ipynb")
    assert "import torch" not in text
    assert "src.clef_probe" not in text
