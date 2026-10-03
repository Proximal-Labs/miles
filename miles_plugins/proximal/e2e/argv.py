"""Command-line edits shared by the e2e tools that run Miles's trainer with changed flags."""


def set_flag(argv: list[str], flag: str, value: str | None) -> list[str]:
    """Drop every occurrence of ``flag`` (and its value); re-add it last when ``value`` is set."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i].split("=", 1)[0] == flag:
            inline = "=" in argv[i]
            i += 2 if not inline and i + 1 < len(argv) and not argv[i + 1].startswith("--") else 1
            continue
        out.append(argv[i])
        i += 1
    return out + ([flag, value] if value is not None else [])
