import ast
from pathlib import Path


_EXEMPTION = "config-access-exempt:"


def main() -> None:
    violations = [message for path in Path("miles").rglob("*.py") for message in _violations(path)]
    if violations:
        raise SystemExit("\n".join(violations))


def _violations(path: Path) -> list[str]:
    source = path.read_text()
    lines = source.splitlines()
    tree = ast.parse(source)
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in {"getattr", "hasattr"} or not node.args:
            continue
        _, marker, reason = lines[node.lineno - 1].partition(_EXEMPTION)
        if not marker or not reason.strip() or reason.strip() == "runtime reflection is required":
            violations.append(f"{path}:{node.lineno}: dynamic attribute access needs a specific inline exemption")
    return violations


if __name__ == "__main__":
    main()
