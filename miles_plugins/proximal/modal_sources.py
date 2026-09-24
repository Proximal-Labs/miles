"""The parts of this fork's source tree that the Modal images carry.

Modal's ``add_local_python_source`` ships only ``.py`` files by default. Capture also
loads its chat-template family's fixed template (a ``.jinja`` file under
``miles/utils/chat_template_utils/templates``) at startup, so the images add those too.
"""

import modal
from modal.file_pattern_matcher import FilePatternMatcher

PACKAGES = ("miles", "miles_plugins")
# Ignore everything except Python sources and chat templates.
IGNORE = ~FilePatternMatcher("**/*.py", "**/*.jinja")


def add_fork_sources(image: modal.Image) -> modal.Image:
    """This fork's Miles and plugin sources, over the Miles image's installed copy."""
    return image.add_local_python_source(*PACKAGES, ignore=IGNORE)
