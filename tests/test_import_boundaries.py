"""Features use each other's public names only; private helpers stay inside their package."""

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("h3_app", "h3_ui")


def module_name(path: Path) -> str:
    parts = path.relative_to(ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def owner(module: str) -> str:
    """The feature a module belongs to: its package below h3_app/h3_ui, or the module itself."""
    parts = module.split(".")
    return ".".join(parts[:2])


def resolve(module: str, path: Path, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    package = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
    package = package[: len(package) - (node.level - 1)]
    return ".".join(package + ([node.module] if node.module else []))


def private_imports():
    """(importer, imported module, name) for private names taken from another feature."""
    found = []
    for package in PACKAGES:
        for path in sorted((ROOT / package).rglob("*.py")):
            module = module_name(path)
            tree = ast.parse(path.read_text(encoding="utf-8"))
            aliases = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    source = resolve(module, path, node)
                    for alias in node.names:
                        target = f"{source}.{alias.name}"
                        if alias.name.startswith("_") and not alias.name.startswith("__"):
                            if owner(source) != owner(module):
                                found.append((module, source, alias.name))
                        else:
                            aliases[alias.asname or alias.name] = target
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        aliases[alias.asname or alias.name.split(".")[0]] = alias.name
            for node in ast.walk(tree):
                if (isinstance(node, ast.Attribute) and node.attr.startswith("_") and not node.attr.startswith("__")
                        and isinstance(node.value, ast.Name) and node.value.id in aliases):
                    source = aliases[node.value.id]
                    if source.startswith(PACKAGES) and owner(source) != owner(module):
                        found.append((module, source, node.attr))
    return sorted(set(found))


PAGE_MODULES = (
    "h3_ui/h3/page.py",
    "h3_ui/ltx/page.py",
    "h3_ui/image/page.py",
    "h3_ui/song/page.py",
    "h3_ui/media/page.py",
    "h3_ui/remake_page.py",
    "h3_ui/storyboard/page.py",
    "h3_ui/drama3/page.py",
    "h3_ui/dubbing/page.py",
)
SHARED_PAGE_NAMES = {"plain", "Run", "Runs", "guarded", "OwnerPageState", "private_page_root", "install_static_page"}


def peer_page_imports():
    """(importer, source, name) where a standalone page imports shared infrastructure from a peer page."""
    peer_sources = {module_name(ROOT / rel) for rel in PAGE_MODULES}
    found = []
    for rel in PAGE_MODULES:
        path = ROOT / rel
        module = module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                source = resolve(module, path, node)
                if source in peer_sources and source != module:
                    for alias in node.names:
                        if alias.name in SHARED_PAGE_NAMES:
                            found.append((module, source, alias.name))
    return sorted(set(found))


class ImportBoundaryTests(unittest.TestCase):
    def test_no_private_names_cross_feature_boundaries(self):
        self.assertEqual(private_imports(), [])

    def test_standalone_pages_use_shared_page_infrastructure(self):
        self.assertEqual(peer_page_imports(), [])


if __name__ == "__main__":
    for violation in private_imports():
        print(*violation)
