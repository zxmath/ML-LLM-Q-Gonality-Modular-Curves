from pathlib import Path

def find_project_root(indicators=("modular.toml")):
    path = Path(__file__).resolve()
    for parent in [path] + list(path.parents):
        if any((parent / name).exists() for name in indicators):
            return parent
    raise FileNotFoundError(f"No project root found with any of: {indicators}")
