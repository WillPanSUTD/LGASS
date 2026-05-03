from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
README = (REPO / "README.md").read_text(encoding="utf-8")

REQUIRED_HEADINGS = [
    "LGASS",                                      # title
    "Defect Classes",
    "Data Acquisition",
    "Architecture",
    "Results",
    "Ablation",
    "Hyperparameters",
    "Dataset Structure",
    "Installation",
    "Usage",
    "Citation",
    "License",
    "Acknowledgements",
]

def test_required_headings_present():
    for h in REQUIRED_HEADINGS:
        assert h in README, f"missing section: {h}"

def test_results_table_contains_lgass_row_with_correct_numbers():
    # Paper Table 3, ours row.
    for token in ["99.47", "92.37", "79.23", "76.22", "72.17", "71.33", "91.61", "64.95"]:
        assert token in README, f"missing LGASS metric: {token}"

def test_no_phantom_script_references():
    assert (REPO / "visualize.py").is_file()
    assert (REPO / "export_report.py").is_file()

def test_dataset_path_is_npz():
    assert "sealingNail_npz" in README
    assert "sealingNail_normal" not in README

def test_links_to_huggingface_dataset():
    assert "huggingface.co/datasets/vpan1226/OPT-SND" in README
