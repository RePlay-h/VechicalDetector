from pathlib import Path

def test_ml_src_exists():
    assert Path("ml/src").exists(), "ml/src directory does not exist"

def test_readme_exists_and_not_empty():
    readme_path = Path("README.md")
    assert readme_path.exists(), "README.md file does not exist"
    assert readme_path.stat().st_size > 0, "README.md file is empty"