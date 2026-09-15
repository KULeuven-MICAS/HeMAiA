#!/usr/bin/env python3
"""Regression checks for selecting the tapeout GEMM geometry."""

import pathlib
import subprocess
import tempfile
import unittest

import hjson

import validate_shapes


ROOT = pathlib.Path(__file__).resolve().parents[6]
HERE = pathlib.Path(__file__).resolve().parent
SNITCH = ROOT / "deps/snitch_cluster"
CFG_DIR = SNITCH / "target/snitch_cluster/cfg"


@unittest.skipUnless(CFG_DIR.is_dir(), "requires the snitch_cluster dependency")
class GemmShapesTest(unittest.TestCase):
    def expected(self, name):
        return validate_shapes.expected_from_hwcfg(
            hjson.loads((CFG_DIR / f"{name}.hjson").read_text()))

    def test_shape_zero_counts_and_masks(self):
        for cfg in ("snax_versacore_to_256KB_cluster",
                    "snax_versacore_to_256KB_simd_cluster"):
            with self.subTest(cfg=cfg):
                globals_, shapes = self.expected(cfg)
                first = shapes[0]
                self.assertEqual(
                    [first[k] for k in ("meshRow", "tileSize", "meshCol")],
                    [8, 16, 16])
                self.assertEqual(first["Ctlbound0"], 4)
                self.assertEqual(first["D32tlbound0"], 4)
                self.assertEqual(first["channel_en_A"], [0xffff])
                self.assertEqual(first["channel_en_B"], [0, 0xffffffff])
                self.assertEqual(validate_shapes.diff(
                    globals_, shapes,
                    *validate_shapes.parse_header((HERE / "gemm_shapes.h").read_text())), [])
        _, shapes = self.expected("snax_versacore_to_cluster")
        self.assertEqual(shapes[0]["D32tlbound0"], 32)
        self.assertEqual(shapes[0]["tileSize"], 2)

    def test_generation_preserves_mtime_and_checks_granularity(self):
        with tempfile.TemporaryDirectory() as temporary:
            header = pathlib.Path(temporary) / "gemm_shapes.h"
            for cfg in ("snax_versacore_to_256KB_cluster",
                        "snax_versacore_to_256KB_simd_cluster",
                        "snax_versacore_to_cluster"):
                with self.subTest(cfg=cfg):
                    globals_, shapes = self.expected(cfg)
                    text = validate_shapes.render_header(globals_, shapes)
                    validate_shapes.write_if_changed(header, text)
                    before = header.stat().st_mtime_ns
                    self.assertFalse(validate_shapes.write_if_changed(header, text))
                    self.assertEqual(header.stat().st_mtime_ns, before)
                    self.assertEqual(validate_shapes.diff(
                        globals_, shapes, *validate_shapes.parse_header(header.read_text())), [])
                    wrong = text.replace("#define BINGO_GRANULARITY_A 2",
                                         "#define BINGO_GRANULARITY_A 1")
                    errors = validate_shapes.diff(
                        globals_, shapes, *validate_shapes.parse_header(wrong))
                    self.assertTrue(any("BINGO_GRANULARITY_A" in error for error in errors))

    def test_make_switches_geometry_and_recovers_deleted_header(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            header = directory / "gemm_shapes.h"
            stamp = directory / "validated.stamp"
            makefile = directory / "Makefile"
            makefile.write_text(
                ".DEFAULT_GOAL := all\n"
                f"SNITCH_ROOT := {SNITCH}\n"
                f"GEMM_SHAPES := {header}\n"
                f"VALIDATE_STAMP := {stamp}\n"
                f"include {HERE / 'gemm_shapes.mk'}\n"
                "all: $(GEMM_SHAPES) $(VALIDATE_STAMP)\n")

            def build(cfg):
                subprocess.run(
                    ["make", "--no-print-directory", "-f", str(makefile),
                     f"CFG={ROOT / 'target/rtl/cfg' / (cfg + '.hjson')}"],
                    cwd=directory, check=True, capture_output=True, text=True,
                    timeout=20)
                return validate_shapes.parse_header(header.read_text())[1][0]

            self.assertEqual(build("hemaia_tapeout_2c")["D32tlbound0"], 4)
            before = (header.stat().st_mtime_ns, stamp.stat().st_mtime_ns)
            build("hemaia_tapeout_2c")
            self.assertEqual((header.stat().st_mtime_ns, stamp.stat().st_mtime_ns), before)
            self.assertEqual(build("hemaia_tapeout_1c")["D32tlbound0"], 32)
            self.assertEqual(build("hemaia_tapeout_2c_simd")["D32tlbound0"], 4)
            before_header = header.stat().st_mtime_ns
            before_stamp = stamp.stat().st_mtime_ns
            build("hemaia_tapeout_2c")
            self.assertEqual(header.stat().st_mtime_ns, before_header)
            self.assertGreater(stamp.stat().st_mtime_ns, before_stamp)
            header.unlink()
            self.assertEqual(build("hemaia_tapeout_2c")["D32tlbound0"], 4)

if __name__ == "__main__":
    unittest.main()
