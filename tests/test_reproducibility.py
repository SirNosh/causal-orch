import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
ARE_COMMIT = "7946367413129784139e785ae4c351090002a0bb"


class ReproducibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "pyproject.toml").open("rb") as stream:
            cls.project = tomllib.load(stream)["project"]
        with (ROOT / "uv.lock").open("rb") as stream:
            cls.lock = tomllib.load(stream)

    def test_project_declares_supported_runtime_and_test_dependencies(self):
        self.assertEqual(self.project["requires-python"], ">=3.11,<3.12")
        self.assertIn(
            f"meta-agents-research-environments @ git+https://github.com/facebookresearch/"
            f"meta-agents-research-environments.git@{ARE_COMMIT}",
            self.project["dependencies"],
        )
        self.assertIn("PyYAML==6.0.2", self.project["dependencies"])
        self.assertEqual(self.project["optional-dependencies"]["test"], ["pytest==8.3.4"])

    def test_lock_records_exact_python_and_are_revision(self):
        self.assertEqual(self.lock["requires-python"], "==3.11.*")
        packages = {package["name"]: package for package in self.lock["package"]}
        are_source = packages["meta-agents-research-environments"]["source"]["git"]
        self.assertEqual(
            are_source,
            "https://github.com/facebookresearch/meta-agents-research-environments.git"
            f"?rev={ARE_COMMIT}#{ARE_COMMIT}",
        )

    def test_lock_has_hashed_artifacts_for_direct_registry_dependencies(self):
        packages = {package["name"]: package for package in self.lock["package"]}
        for name, version in (("pyyaml", "6.0.2"), ("pytest", "8.3.4")):
            package = packages[name]
            self.assertEqual(package["version"], version)
            self.assertTrue(package["wheels"])
            self.assertTrue(all("hash" in wheel for wheel in package["wheels"]))
