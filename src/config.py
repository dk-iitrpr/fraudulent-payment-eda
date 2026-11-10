#src/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # project root
DATA_PATH = ROOT / "dataset.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
