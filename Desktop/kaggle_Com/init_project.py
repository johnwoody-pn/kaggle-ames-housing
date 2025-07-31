import os
from pathlib import Path

def setup_kaggle_project(base_dir="."):
    folders = [
        "data/raw",
        "data/processed",
        "notebooks",
        "scripts",
        "models",
        "outputs",
        "eda",
        "reports"
    ]
    for folder in folders:
        os.makedirs(Path(base_dir) / folder, exist_ok=True)

    Path(base_dir, "README.md").write_text("# Kaggle Practice Project\n\nThis is a structured workspace for Kaggle projects.\n")
    Path(base_dir, ".gitignore").write_text("__pycache__/\n*.pyc\n.ipynb_checkpoints/\ndata/processed/\nmodels/\noutputs/\n")

    print(f"✔ Project folders created in: {Path(base_dir).resolve()}")

if __name__ == "__main__":
    setup_kaggle_project()