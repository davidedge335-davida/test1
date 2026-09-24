"""Dependency-free checks for the official E3CoverNet package layout."""

import ast
import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "e3covernet"

OFFICIAL_MODEL_FILES = {
    "models/backbone/e3covernet/__init__.py",
    "models/backbone/e3covernet/attention.py",
    "models/backbone/e3covernet/backbone.py",
    "models/backbone/e3covernet/blocks.py",
    "models/backbone/e3covernet/covering.py",
    "models/backbone/e3covernet/rbf.py",
    "models/fusion/__init__.py",
    "models/fusion/cross_attention_fusion.py",
    "models/grounding/__init__.py",
    "models/grounding/e3cover_grounding_net.py",
    "models/losses/__init__.py",
    "models/losses/info_nce.py",
    "models/retrieval/__init__.py",
    "models/retrieval/text2shape_model.py",
    "scripts/train_grounding.py",
    "scripts/train_text2shape.py",
}

FORBIDDEN_TERMS = re.compile(
    r"(?<!\d)" + "20" + r"20(?!\d)|" + "ec" + "cv|"
    + "European Conference on " + "Computer Vision|"
    + "refer" + r"[\s-]?it3d|"
    + "Neural Listeners for Fine-Grained "
    + "3D Object Identification in Real-World Scenes",
    re.IGNORECASE,
)


class PackageLayoutTests(unittest.TestCase):
    def test_official_model_manifest_is_complete(self):
        missing = sorted(
            relative for relative in OFFICIAL_MODEL_FILES
            if not (PACKAGE_ROOT / relative).is_file()
        )
        self.assertEqual(missing, [])

    def test_backbone_parent_remains_an_implicit_namespace(self):
        self.assertFalse(
            (PACKAGE_ROOT / "models/backbone/__init__.py").exists()
        )

    def test_internal_import_modules_exist(self):
        failures = []
        for source in PACKAGE_ROOT.rglob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8"), str(source))
            current = list(
                source.relative_to(PROJECT_ROOT).with_suffix("").parts[:-1]
            )
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.level:
                    module_parts = node.module.split(".") if node.module else []
                    parts = current[:len(current) - node.level + 1] + module_parts
                elif node.module and (
                        node.module == "e3covernet"
                        or node.module.startswith("e3covernet.")):
                    parts = node.module.split(".")
                else:
                    continue
                target = PROJECT_ROOT.joinpath(*parts)
                if not (target.with_suffix(".py").is_file()
                        or (target / "__init__.py").is_file()):
                    failures.append(f"{source}:{node.lineno}: {node.module}")
        self.assertEqual(failures, [])

    def test_legacy_metadata_is_absent_from_text_files(self):
        failures = []
        for source in PROJECT_ROOT.rglob("*"):
            if not source.is_file() or ".git" in source.parts:
                continue
            try:
                contents = source.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if FORBIDDEN_TERMS.search(contents):
                failures.append(str(source.relative_to(PROJECT_ROOT)))
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
