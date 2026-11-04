"""Workshop package changes must not invalidate Studio's expensive image recipe."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class WorkshopImageIsolationTests(unittest.TestCase):
    def recipe(self, requirements, *, local=True):
        path = Path(__file__).resolve().parents[1] / "modal_h3.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        studio, workshop = Mock(), Mock()
        for image in (studio, workshop):
            for method in ("entrypoint", "apt_install", "uv_pip_install", "run_commands",
                           "add_local_file", "add_local_dir", "run_function"):
                getattr(image, method).return_value = image
        modal = SimpleNamespace(
            is_local=lambda: local,
            Image=SimpleNamespace(from_registry=lambda *a, **k: studio,
                                  debian_slim=lambda *a, **k: workshop),
            Volume=Mock(), Secret=Mock(), App=Mock(),
        )
        body = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "layout":
                break  # Execute image construction, without service decorators or Modal resources.
            if isinstance(node, ast.Import) and any(alias.name == "modal" for alias in node.names):
                continue
            body.append(node)
        namespace = {"__file__": str(path), "modal": modal}
        module = SimpleNamespace(WORKSHOP_REQUIREMENTS=requirements) if local else None
        with (patch.dict(sys.modules, {"h3_workshop_requirements": module}),
              patch.object(sys, "path", list(sys.path))):
            exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
        return namespace, studio, workshop

    def test_workshop_package_change_only_changes_cpu_install(self):
        before, studio_before, cpu_before = self.recipe(("cpu-package==1",))
        after, studio_after, cpu_after = self.recipe(("cpu-package==2",))
        self.assertEqual(before["REVISION"], after["REVISION"])
        self.assertNotIn(before["LOCAL_WORKSHOP_REQUIREMENTS"], before["_BUILD_LOCAL_FILES"])
        # run_function's Python callable is a new object in each exec; compare its
        # cache-relevant revision separately from the rest of the image recipe.
        self.assertEqual(studio_before.run_function.call_args.kwargs,
                         studio_after.run_function.call_args.kwargs)
        self.assertEqual([c for c in studio_before.mock_calls if c[0] != "run_function"],
                         [c for c in studio_after.mock_calls if c[0] != "run_function"])
        self.assertEqual(cpu_before.uv_pip_install.call_args.args, ("cpu-package==1",))
        self.assertEqual(cpu_after.uv_pip_install.call_args.args, ("cpu-package==2",))

    def test_remote_build_import_does_not_need_workshop_module(self):
        # The module is deliberately unavailable, as it is in Studio's build image.
        _, _, cpu = self.recipe((), local=False)
        cpu.uv_pip_install.assert_not_called()


if __name__ == "__main__":
    unittest.main()
