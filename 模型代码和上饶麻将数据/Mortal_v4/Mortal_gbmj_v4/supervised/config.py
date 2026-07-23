from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.toml"


with CONFIG_PATH.open("rb") as f:
    config = tomllib.load(f)
