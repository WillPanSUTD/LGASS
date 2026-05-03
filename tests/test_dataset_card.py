from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parent.parent
CARD = REPO / "docs" / "hf-dataset-card" / "README.md"


def _split_frontmatter(text):
    assert text.startswith("---\n"), "missing YAML frontmatter"
    _, fm, body = text.split("---\n", 2)
    return yaml.safe_load(fm), body


def test_card_exists():
    assert CARD.is_file()


def test_card_has_required_yaml_keys():
    fm, _ = _split_frontmatter(CARD.read_text(encoding="utf-8"))
    for key in ["license", "task_categories", "language",
                "size_categories", "tags", "pretty_name"]:
        assert key in fm, f"missing frontmatter key: {key}"
    assert fm["license"] == "apache-2.0"


def test_card_has_required_sections():
    text = CARD.read_text(encoding="utf-8")
    for section in [
        "Gated access", "Quick start", "Dataset summary",
        "Dataset structure", "Data fields", "Class definitions",
        "Acquisition setup", "Annotation protocol",
        "Considerations", "Citation",
    ]:
        assert section in text, f"missing section: {section}"


def test_card_quick_start_has_load_dataset_snippet():
    text = CARD.read_text(encoding="utf-8")
    assert 'load_dataset("vpan1226/OPT-SND")' in text
    assert "huggingface-cli login" in text


def test_card_describes_8_class_raw_schema_and_6_class_eval():
    text = CARD.read_text(encoding="utf-8")
    for raw in ["Background1", "Background2", "Burst2"]:
        assert raw in text, f"raw class missing: {raw}"
    assert "Background1 ∪ Background2" in text or "Background1 + Background2" in text


def test_card_describes_npz_schema():
    text = CARD.read_text(encoding="utf-8")
    assert "(N, 7)" in text or "(P, 7)" in text
    for col in ["nx", "ny", "nz", "label"]:
        assert col in text
