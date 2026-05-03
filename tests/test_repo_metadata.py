from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

def test_license_file_exists():
    assert (REPO / "LICENSE").is_file()

def test_license_is_apache_2():
    text = (REPO / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in text
    assert "Version 2.0" in text
    assert "2026" in text
