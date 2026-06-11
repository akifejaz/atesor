"""Tests for the skip-already-released planner used by batch-port.

Covers ``.github/scripts/plan-remaining.py``:
* Regex parsing of release asset filenames (per-platform isolation).
* Declared-order preservation.
* Shard count math from the remaining set.
* Edge cases: empty release, everything released, malformed names,
  mixed platforms.

Also covers the polymorphic loader in
``.github/scripts/missing-pkgs.py``: it must accept BOTH the
``full.json`` schema (list of ``{name: ...}`` dicts) AND the
``remaining-<platform>.json`` schema (list of plain strings).
"""

import importlib.util
import json
import os
import tempfile
import unittest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        os.path.join(_REPO, rel_path),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plan_remaining = _load(
    ".github/scripts/plan-remaining.py",
    "plan_remaining_mod",
)
missing_pkgs = _load(
    ".github/scripts/missing-pkgs.py",
    "missing_pkgs_pr_mod",
)
slice_pkgs = _load(
    ".github/scripts/slice-pkgs.py",
    "slice_pkgs_mod",
)


class TestReleasedNames(unittest.TestCase):
    """``_released_names`` extracts package names from asset filenames."""

    def test_basic_match(self) -> None:
        """Matching alpine asset filenames yield their package names."""
        names = plan_remaining._released_names(
            [
                "afrog-20260604-143632-alpine.zip",
                "age-20260605-064655-alpine.zip",
            ],
            "alpine",
        )
        self.assertEqual(names, {"afrog", "age"})

    def test_per_platform_isolation(self) -> None:
        """Alpine assets must not mark a package released for debian."""
        names = plan_remaining._released_names(
            [
                "afrog-20260604-143632-alpine.zip",
                "age-20260604-092304-debian.zip",
            ],
            "debian",
        )
        self.assertEqual(names, {"age"})

    def test_hyphenated_package_name(self) -> None:
        """Hyphenated package names are extracted before the timestamp."""
        names = plan_remaining._released_names(
            ["lzip-1.0-20260604-150544-alpine.zip"],
            "alpine",
        )
        self.assertEqual(names, {"lzip-1.0"})

    def test_camelcase_package_name(self) -> None:
        """Mixed-case package names are preserved when extracted."""
        names = plan_remaining._released_names(
            ["AnalyticsRelationships-20260604-150544-debian.zip"],
            "debian",
        )
        self.assertEqual(names, {"AnalyticsRelationships"})

    def test_malformed_filename_ignored(self) -> None:
        """Malformed filenames are ignored while valid assets are kept."""
        names = plan_remaining._released_names(
            [
                "foo.zip",  # no pattern
                "bar-20260601-debian.zip",  # missing time
                "baz-2026-06-01-150000-alpine.zip",  # wrong date fmt
                "quux-20260601-150000-alpine.zip.bak",  # extra suffix
                "good-20260601-150000-alpine.zip",  # OK
            ],
            "alpine",
        )
        self.assertEqual(names, {"good"})

    def test_empty_input(self) -> None:
        """Empty asset inputs return an empty released-name set."""
        self.assertEqual(
            plan_remaining._released_names([], "alpine"),
            set(),
        )
        self.assertEqual(
            plan_remaining._released_names(["", "   ", "\n"], "alpine"),
            set(),
        )

    def test_unknown_platform_not_matched(self) -> None:
        """Unknown-platform assets do not match another platform."""
        names = plan_remaining._released_names(
            ["foo-20260601-150000-windows.zip"],
            "alpine",
        )
        self.assertEqual(names, set())


class TestComputePlan(unittest.TestCase):
    """``compute_plan`` skips released names + computes shard count."""

    def test_preserves_declared_order(self) -> None:
        """Remaining packages keep declared order after released names skip."""
        declared = ["c", "a", "b", "d"]
        released = {"a"}
        remaining, total, groups = plan_remaining.compute_plan(
            declared,
            released,
            group_size=50,
        )
        self.assertEqual(remaining, ["c", "b", "d"])
        self.assertEqual(total, 1)
        self.assertEqual(groups, [0])

    def test_no_skips_when_release_empty(self) -> None:
        """Empty released set leaves all declared packages in one shard."""
        declared = ["a", "b", "c"]
        remaining, total, groups = plan_remaining.compute_plan(
            declared,
            set(),
            group_size=50,
        )
        self.assertEqual(remaining, declared)
        self.assertEqual(total, 1)
        self.assertEqual(groups, [0])

    def test_everything_released_yields_zero_shards(self) -> None:
        """All released packages yield no remaining packages or shards."""
        declared = ["a", "b"]
        released = {"a", "b"}
        remaining, total, groups = plan_remaining.compute_plan(
            declared,
            released,
            group_size=50,
        )
        self.assertEqual(remaining, [])
        self.assertEqual(total, 0)
        self.assertEqual(groups, [])

    def test_shard_math_matches_group_size(self) -> None:
        """Shard count uses ceil division over the remaining packages."""
        # 172 remaining at 50 per shard -> 4 shards (ceil)
        declared = [f"pkg{i}" for i in range(200)]
        released = {f"pkg{i}" for i in range(28)}  # leaves 172
        remaining, total, groups = plan_remaining.compute_plan(
            declared,
            released,
            group_size=50,
        )
        self.assertEqual(len(remaining), 172)
        self.assertEqual(total, 4)
        self.assertEqual(groups, [0, 1, 2, 3])

    def test_invalid_group_size(self) -> None:
        """Zero or negative group sizes raise ValueError."""
        with self.assertRaises(ValueError):
            plan_remaining.compute_plan(["a"], set(), group_size=0)
        with self.assertRaises(ValueError):
            plan_remaining.compute_plan(["a"], set(), group_size=-1)

    def test_released_not_in_declared_is_noop(self) -> None:
        """Released names outside declared packages do not affect planning."""
        # Spurious assets (e.g. a package no longer on the list)
        # must not affect the remaining computation.
        declared = ["a", "b"]
        released = {"a", "ghost"}
        remaining, total, _ = plan_remaining.compute_plan(
            declared,
            released,
            group_size=50,
        )
        self.assertEqual(remaining, ["b"])
        self.assertEqual(total, 1)


class TestLoadReleasedAssets(unittest.TestCase):
    """``_load_released_assets`` is robust to missing/empty files."""

    def test_missing_path(self) -> None:
        """Missing or unset asset paths load as an empty list."""
        self.assertEqual(plan_remaining._load_released_assets(None), [])
        self.assertEqual(
            plan_remaining._load_released_assets("/no/such/file"),
            [],
        )

    def test_strips_blanks(self) -> None:
        """Released asset loading strips blank lines from files."""
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".txt",
            delete=False,
        ) as fh:
            fh.write("a-20260601-150000-alpine.zip\n")
            fh.write("\n")
            fh.write("b-20260601-150000-alpine.zip\n")
            path = fh.name
        try:
            self.assertEqual(
                plan_remaining._load_released_assets(path),
                [
                    "a-20260601-150000-alpine.zip",
                    "b-20260601-150000-alpine.zip",
                ],
            )
        finally:
            os.unlink(path)


class TestMissingPkgsPolymorphicLoader(unittest.TestCase):
    """missing-pkgs must read both full.json and remaining-*.json."""

    def _write_json(self, payload: dict) -> str:
        fh = tempfile.NamedTemporaryFile(
            "w",
            suffix=".json",
            delete=False,
        )
        json.dump(payload, fh)
        fh.close()
        return fh.name

    def test_loads_dict_schema(self) -> None:
        """Dict package entries load by their name fields."""
        # .github/packages/*.json shape.
        path = self._write_json(
            {
                "packages": [
                    {"name": "a", "url": "..."},
                    {"name": "b", "url": "..."},
                ],
            }
        )
        try:
            self.assertEqual(
                missing_pkgs._load_package_names(path),
                ["a", "b"],
            )
        finally:
            os.unlink(path)

    def test_loads_string_schema(self) -> None:
        """String package entries load unchanged from remaining files."""
        # remaining-<platform>.json shape, written by plan-remaining.py.
        path = self._write_json({"packages": ["x", "y", "z"]})
        try:
            self.assertEqual(
                missing_pkgs._load_package_names(path),
                ["x", "y", "z"],
            )
        finally:
            os.unlink(path)

    def test_skips_entries_without_name(self) -> None:
        """Package entries without names are skipped during loading."""
        path = self._write_json(
            {
                "packages": [
                    {"name": "ok"},
                    {"url": "no-name"},  # dropped
                    "",  # dropped
                    "also-ok",
                ],
            }
        )
        try:
            self.assertEqual(
                missing_pkgs._load_package_names(path),
                ["ok", "also-ok"],
            )
        finally:
            os.unlink(path)


class TestPlanRemainingWritesArtifact(unittest.TestCase):
    """End-to-end: ``_write_remaining`` + reload via missing-pkgs."""

    def test_roundtrip(self) -> None:
        """Written remaining packages round-trip through the loader."""
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "remaining-alpine.json")
            plan_remaining._write_remaining(out, ["a", "b", "c"])
            with open(out) as fh:
                data = json.load(fh)
            self.assertEqual(data, {"packages": ["a", "b", "c"]})
            # And the loader missing-pkgs uses must see the same names.
            self.assertEqual(
                missing_pkgs._load_package_names(out),
                ["a", "b", "c"],
            )


class TestSlicePkgs(unittest.TestCase):
    """slice-pkgs must match the canonical ceil-division chunking."""

    def test_total_one_returns_full_list(self) -> None:
        """A single shard returns the full package list."""
        self.assertEqual(
            slice_pkgs.apply_shard(["a", "b", "c"], 0, 1),
            ["a", "b", "c"],
        )

    def test_partitions_cover_every_name_exactly_once(self) -> None:
        """Shard partitions rebuild every package name exactly once."""
        names = [f"p{i}" for i in range(173)]
        for total in [1, 2, 4, 7]:
            with self.subTest(total=total):
                rebuilt: list[str] = []
                for i in range(total):
                    rebuilt.extend(
                        slice_pkgs.apply_shard(names, i, total),
                    )
                self.assertEqual(rebuilt, names)

    def test_matches_missing_pkgs_apply_shard(self) -> None:
        """Slice-pkgs sharding matches missing-pkgs sharding."""
        names = [f"p{i}" for i in range(50)]
        for total in [1, 2, 3, 7]:
            for idx in range(total):
                with self.subTest(total=total, idx=idx):
                    self.assertEqual(
                        slice_pkgs.apply_shard(names, idx, total),
                        missing_pkgs._apply_shard(names, idx, total),
                    )


if __name__ == "__main__":
    unittest.main()
