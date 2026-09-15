"""Regression tests for configuration selection and Make invalidation."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resolve_cluster_cfg import cluster_config_paths, gemm_config_path, update_stamp


ROOT = Path(__file__).resolve().parents[2]


class ClusterConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hemaia-cluster-cfg-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.snitch = self.directory / "snitch"
        self.cluster_dir = self.snitch / "target/snitch_cluster/cfg"
        self.cluster_dir.mkdir(parents=True)
        self.a = self.write_cluster("generic", [32, 2, 32])
        self.b = self.write_cluster("small", [8, 16, 16])
        self.cfg = self.directory / "soc.hjson"
        self.write_soc(["small", "small"])

    def write_cluster(self, name, shape):
        path = self.cluster_dir / f"{name}.hjson"
        path.write_text(json.dumps({"snax_versacore_core_template": {
            "snax_acc_cfg": [{"snax_versacore_spatial_unrolling": [[shape]]}]
        }}))
        return path

    def write_soc(self, names, path=None):
        (path or self.cfg).write_text(json.dumps({"clusters": names}))

    def test_selection_and_heterogeneity(self):
        self.assertEqual(cluster_config_paths(self.cfg, self.snitch), [self.b, self.b])
        self.assertEqual(gemm_config_path(self.cfg, self.snitch), self.b)
        self.write_soc(["generic", "small"])
        with self.assertRaisesRegex(ValueError, "incompatible GEMM"):
            gemm_config_path(self.cfg, self.snitch)
        self.write_soc(["absent"])
        with self.assertRaisesRegex(ValueError, "does not exist"):
            cluster_config_paths(self.cfg, self.snitch)

    def test_stamp_preserves_mtime_until_selection_or_content_changes(self):
        stamp = self.directory / "stamp"
        self.assertTrue(update_stamp(stamp, [self.cfg, self.a]))
        before = stamp.stat().st_mtime_ns
        self.assertFalse(update_stamp(stamp, [self.cfg, self.a]))
        self.assertEqual(stamp.stat().st_mtime_ns, before)
        self.assertTrue(update_stamp(stamp, [self.cfg, self.b]))
        self.write_cluster("small", [16, 8, 16])
        self.assertTrue(update_stamp(stamp, [self.cfg, self.b]))

    def test_make_rebuilds_for_older_config_switch_and_same_mtime_edit(self):
        makefile = self.directory / "Makefile"
        makefile.write_text(
            f"include {ROOT}/util/cluster_cfg.mk\n"
            ".SECONDEXPANSION:\n"
            "result: $$(CLUSTER_CFG_DEPS)\n"
            "\t@echo rebuild >> builds\n"
            "\t@cp $(GEMM_HWCFG) $@\n"
        )
        older_cfg = self.directory / "older.hjson"
        self.write_soc(["generic"], older_cfg)
        os.utime(older_cfg, (1, 1))
        os.utime(self.a, (1, 1))

        def build(cfg):
            subprocess.run(["make", "--no-print-directory", f"CFG={cfg}",
                            f"SNITCH_ROOT={self.snitch}"], cwd=self.directory,
                           check=True, capture_output=True, text=True)

        build(self.cfg)  # Default goal must remain 'result', not the helper stamp.
        build(self.cfg)
        self.assertEqual((self.directory / "builds").read_text().splitlines(), ["rebuild"])
        build(older_cfg)
        self.assertEqual((self.directory / "result").read_text(), self.a.read_text())
        build(self.cfg)  # Also invalidate when switching back to a previous config.
        previous_mtime = self.b.stat().st_mtime_ns
        self.write_cluster("small", [16, 8, 16])
        os.utime(self.b, ns=(previous_mtime, previous_mtime))
        build(self.cfg)
        build(self.cfg)
        self.assertEqual((self.directory / "result").read_text(), self.b.read_text())
        self.assertEqual(len((self.directory / "builds").read_text().splitlines()), 4)

    def test_clean_does_not_need_a_config_or_checkout(self):
        makefile = self.directory / "Makefile"
        makefile.write_text(
            f"include {ROOT}/util/cluster_cfg.mk\n"
            ".SECONDEXPANSION:\n"
            "result: $$(CLUSTER_CFG_DEPS)\n"
            "\t@touch $@\n"
            ".PHONY: clean clean-data\n"
            "clean clean-data:\n"
            "\t@rm -f result\n"
        )
        subprocess.run(["make", "--no-print-directory", "-n", "clean", "clean-data",
                        "CFG=missing.hjson", "SNITCH_ROOT=missing-checkout"],
                       cwd=self.directory, check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
