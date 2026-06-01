from pathlib import Path

# Always use current working directory
base_path = Path.cwd()

folders = [
    "config",
    "data/01-raw",
    "data/02-preprocessed",
    "data/03-features",
    "data/04-predictions",
    "entrypoint",
    "notebooks",
    "src/pipelines",
    "tests",
]

files = [
    "config/config",
    "config/config_dev",
    "entrypoint/inference.py",
    "entrypoint/train.py",
    "notebooks/Baseline.ipynb",
    "notebooks/EDA.ipynb",
    "src/pipelines/__init__.py",
    "src/pipelines/feature_eng_pipeline.py",
    "src/pipelines/inference_pipeline.py",
    "src/pipelines/training_pipeline.py",
    "tests/__init__.py",
    "tests/test_training.py",
    ".gitlab-ci.yml",
    "docker-compose.yml",
    "Dockerfile",
    "env.yaml",
    "env-dev.yaml",
    "Makefile",
    "README.md",
    "requirements-dev.txt",
    "requirements-prod.txt",
]

for folder in folders:
    (base_path / folder).mkdir(parents=True, exist_ok=True)

for file in files:
    file_path = base_path / file
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch(exist_ok=True)

print(" Structure created in:", base_path)
