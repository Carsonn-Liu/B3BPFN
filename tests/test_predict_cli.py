import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "predict_peptide.py"


class PredictCliTests(unittest.TestCase):
    def test_default_model_dir_matches_repository_layout(self):
        module = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        constants = {}

        for node in module.body:
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if isinstance(node.value, ast.Constant):
                constants[target.id] = node.value.value

        model_dir_default = None
        for node in ast.walk(module):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "add_argument":
                continue

            flags = []
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    flags.append(arg.value)

            if "--model_dir" not in flags:
                continue

            for keyword in node.keywords:
                if keyword.arg != "default":
                    continue
                if isinstance(keyword.value, ast.Constant):
                    model_dir_default = keyword.value.value
                elif isinstance(keyword.value, ast.Name):
                    model_dir_default = constants.get(keyword.value.id)
                    break

        self.assertEqual(model_dir_default, "models")


if __name__ == "__main__":
    unittest.main()
