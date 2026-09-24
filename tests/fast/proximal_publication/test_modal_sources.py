from pathlib import Path

import pytest

pytest.importorskip("modal")

from miles.utils.chat_template_utils.tito_tokenizer import TEMPLATE_DIR  # noqa: E402
from miles_plugins.proximal.modal_sources import IGNORE  # noqa: E402

MILES = Path(__file__).resolve().parents[3] / "miles"


def test_images_carry_every_fixed_chat_template():
    templates = sorted(TEMPLATE_DIR.glob("*.jinja"))
    assert templates
    for template in templates:
        assert not IGNORE(template.relative_to(MILES)), template


def test_images_still_skip_other_non_python_files():
    assert not IGNORE(Path("utils/chat_template_utils/tito_tokenizer.py"))
    assert IGNORE(Path("utils/notes.md"))
