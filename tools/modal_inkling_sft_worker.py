"""Run the Miles SFT launcher inside a time-bounded Modal subprocess."""

import json
import sys
from pathlib import Path

from scripts.run_inkling_small_sft import execute


if __name__ == "__main__":
    execute(**json.loads(Path(sys.argv[1]).read_text()))
