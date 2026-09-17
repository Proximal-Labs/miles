import ast
from pathlib import Path


_CONFIG_TYPES = {
    "AllConfig",
    "InferenceControllerConfig",
    "MultiLoraConfig",
    "OrchestratorConfig",
    "RolloutConfig",
    "TrainerConfig",
}


def _annotation_names(annotation: ast.expr | None) -> set[str]:
    if annotation is None:
        return set()
    return {node.id for node in ast.walk(annotation) if isinstance(node, ast.Name)}


def _config_parameter_names(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        names.update(
            argument.arg
            for argument in arguments
            if _annotation_names(argument.annotation) & _CONFIG_TYPES
        )
    return names


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    config_names = _config_parameter_names(tree)
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in {"getattr", "hasattr"} or not node.args:
            continue
        if isinstance(node.args[0], ast.Name) and node.args[0].id in config_names:
            violations.append(f"{path}:{node.lineno}: dynamic access on typed configuration")
    return violations


def main() -> None:
    violations = [message for path in Path("miles").rglob("*.py") for message in _violations(path)]
    if violations:
        raise SystemExit("\n".join(violations))


if __name__ == "__main__":
    main()
