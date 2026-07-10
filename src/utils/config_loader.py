from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parent / "class_names.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

SEG_CLASS_NAMES = config["seg_class_names"]
OD_CLASS_NAME = config["od_class_name"]
COLORS = config["colors"]