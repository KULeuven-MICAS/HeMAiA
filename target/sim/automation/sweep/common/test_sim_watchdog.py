#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Tests of sim_watchdog.py. No RTL simulator is involved: the scheduler log is written by the test in the
format snax_bingo_progress_probe prints, and the "simulator" is a copy of `sleep` named occamy_chip.vcs that
runs in the run's bin directory, so the process side (finding it by name and directory, SIGINT, the events,
the evidence, the status) is exercised on a real PID.   python3 -m unittest -v test_sim_watchdog"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim_watchdog as W  # noqa: E402

NOQUOTA = ["min_headroom_gb=-1000", "max_trace_gb=0"]   # the test must not depend on the account's quota


def ev(t, chip, cl, co, e, tid, ty=0):
    return "[BINGO EV] t=%d chip=%s cl=%d co=%d ev=%s id=%d ty=%d" % (t, chip, cl, co, e, tid, ty)


def prog(t, chip, armed, fetch, nlist, rel, start, done, quiet=0, cells="", ep=1):
    return ("[BINGO PROG] chip=%s t=%d ep=%d armed=%d fetch=%d/%d rel=%d start=%d done=%d quiet=%d anom=0 |%s"
            % (chip, t, ep, armed, fetch, nlist, rel, start, done, quiet, cells))


US = 1000000   # ps per us


def write(path, mode, text):
    with open(path, mode) as f:
        f.write(text)


class JudgeTest(unittest.TestCase):
    def judge(self, **kw):
        p = W.load_params("default", ["%s=%s" % kv for kv in kw.items()])
        return W.Judge(p)

    def test_parse_and_totals(self):
        j = self.judge()
        j.feed_bingo(["[BINGO ARM] t=%d chip=00 ep=1 n_task=40" % (100 * US),
                      ev(101 * US, "00", 0, 1, "F", 7), ev(102 * US, "00", 0, 1, "S", 7), ev(130 * US, "00", 0, 1, "D", 7),
                      prog(131 * US, "00", 1, 10, 40, 8, 5, 4, cells=" 0.1:w2@9,r1@8,c3@7,x1")])
        c = j.b.chips["00"]
        self.assertEqual((c.fetched, c.nlist, c.rel, c.start, c.done), (10, 40, 8, 5, 4))
        self.assertEqual(c.cells[0]["wh"], "9")
        self.assertEqual(j.b.longest_task_ps, 28 * US)
        self.assertEqual(len(j.busy()), 1)

    def test_no_hang_while_tasks_progress(self):
        j = self.judge()
        for k in range(1, 30):   # a progress record every 50 us, the counters move every time
            j.feed_bingo([prog(k * 50 * US, "00", 1, k, 100, k, k, k)])
            self.assertIsNone(j.verdict())

    def test_hang_when_counters_freeze(self):
        j = self.judge(stall_floor_us=300)
        j.feed_bingo([prog(100 * US, "00", 1, 5, 100, 5, 5, 4)])
        j.feed_bingo([prog(300 * US, "00", 1, 5, 100, 5, 5, 4)])
        self.assertIsNone(j.verdict())            # 200 us quiet
        j.feed_bingo([prog(420 * US, "00", 1, 5, 100, 5, 5, 4)])
        v = j.verdict()
        self.assertEqual(v[0], "HANG-SCHED")      # 320 us quiet

    def test_other_chip_moving_is_progress(self):
        j = self.judge(stall_floor_us=300)
        for k in range(1, 20):
            j.feed_bingo([prog(k * 100 * US, "00", 1, 5, 100, 5, 5, 4), prog(k * 100 * US, "33", 1, k, 100, k, k, k)])
            self.assertIsNone(j.verdict())

    def test_idle_chips_and_finished_workload_are_not_a_hang(self):
        j = self.judge(stall_floor_us=300)
        j.feed_bingo(["[BINGO ARM] t=%d chip=00 ep=1 n_task=3" % (10 * US),
                      "[BINGO CMPL] chip=00 t=%d ep=1 armed=0 fetch=3/3 rel=3 start=3 done=3 quiet=0 anom=0" % (20 * US),
                      prog(5000 * US, "33", 0, 0, 0, 0, 0, 0)])
        self.assertIsNone(j.verdict())
        self.assertTrue(j.all_complete())

    def test_adaptive_threshold_follows_the_longest_task(self):
        j = self.judge(stall_floor_us=300, stall_factor=6, stall_cap_us=1000)
        j.feed_bingo([ev(10 * US, "00", 0, 1, "S", 1), ev(110 * US, "00", 0, 1, "D", 1)])   # a 100 us task
        self.assertAlmostEqual(j.threshold_us(), 600.0)
        j.feed_bingo([ev(200 * US, "00", 0, 1, "S", 2), ev(900 * US, "00", 0, 1, "D", 2)])  # a 700 us task: capped
        self.assertAlmostEqual(j.threshold_us(), 1000.0)

    def test_dummy_tasks_do_not_stretch_the_threshold(self):
        j = self.judge()
        j.feed_bingo([ev(10 * US, "00", 0, 1, "S", 1, ty=1), ev(900 * US, "00", 0, 1, "D", 1, ty=1)])
        self.assertEqual(j.b.longest_task_ps, 0)

    def test_axi_stall_confirmed_by_quiet_scheduler(self):
        line = ("[AXI-STALL] testharness.i_dut.i_hemaia_0_0.x @237948000 writes out 1, reads out 0 | "
                "aw 1/0 w 0/0 b 0/1 ar 0/1 r 0/1")
        j = self.judge(axi_confirm_us=100)
        j.feed_probes([line])
        j.feed_bingo([prog(100 * US, "00", 1, 5, 100, 5, 5, 4), prog(150 * US, "00", 1, 5, 100, 5, 5, 4)])
        self.assertIsNone(j.verdict())            # stalled link, scheduler quiet only 50 us
        j.feed_bingo([prog(230 * US, "00", 1, 5, 100, 5, 5, 4)])
        self.assertEqual(j.verdict()[0], "HANG-AXI")

    def test_benign_axi_lines_are_ignored(self):
        blocked_read = ("[AXI-STALL] a @1 writes out 0, reads out 1 | aw 0/1 w 0/0 b 0/1 ar 0/1 r 0/1")
        wrapped = ("[AXI-STALL] b @1 writes out 0, reads out 4294967294 | aw 0/1 w 0/0 b 0/1 ar 0/1 r 0/1")
        self.assertFalse(W.real_axi_stall(blocked_read))
        self.assertFalse(W.real_axi_stall(wrapped))
        self.assertTrue(W.real_axi_stall("[AXI-STALL] c @1 writes out 0, reads out 2 | aw 1/0 w 0/0 b 0/1 ar 0/1 r 0/1"))

    def test_bound_arm_deadline_assertion(self):
        j = self.judge(bound_us=500)
        j.feed_bingo([prog(600 * US, "00", 0, 0, 0, 0, 0, 0)])
        self.assertEqual(j.verdict()[0], "BOUND")
        j = self.judge(arm_deadline_us=1000)
        j.feed_bingo([prog(1500 * US, "00", 0, 0, 0, 0, 0, 0)])
        self.assertEqual(j.verdict()[0], "NOT-ARMED")
        j = self.judge()
        j.feed_transcript(['Error: "x.sv", 10: tb: at time 5 ps'])
        self.assertEqual(j.verdict()[0], "ASSERT")

    def test_anomaly_stops_only_when_asked(self):
        a = "[BINGO ANOMALY] t=5 chip=00 cl=0 co=1 what=checkout_push_without_ready_push w=1 r=8 c=7 x=0"
        j = self.judge(); j.feed_bingo([a]); self.assertIsNone(j.verdict())
        j = self.judge(anomaly="stop"); j.feed_bingo([a]); self.assertEqual(j.verdict()[0], "ANOMALY")

    def test_evidence_names_tasks(self):
        d = tempfile.mkdtemp()
        try:
            f = os.path.join(d, "dfg.csv")
            write(f, "w", "ID,Chiplet,Cluster,Core,Type,Kernel,Name\n9,00,0,4,normal,k,gu_c0_i1\n")
            j = W.Judge(W.load_params("default", []), W.Names(f))
            j.feed_bingo([prog(100 * US, "00", 1, 5, 100, 5, 5, 4, cells=" 0.1:w2@9,r0@-,c1@9,x1")])
            self.assertIn("gu_c0_i1", j.evidence())
        finally:
            shutil.rmtree(d)

    def _dfg(self, d):
        f = os.path.join(d, "dfg.csv")
        write(f, "w", "ID,Chiplet,Cluster,Core,Type,Kernel,Name\n"
              "1544,00,0,4,normal,__host_bingo_kernel_exit,exit_00\n"
              "77,00,0,2,normal,__snax_bingo_kernel_gemm,gu_c0_i3\n")
        return f

    def test_chip_whose_only_open_task_is_its_exit_has_drained(self):
        d = tempfile.mkdtemp()
        try:
            j = W.Judge(W.load_params("default", ["stall_floor_us=300"]), W.Names(self._dfg(d)))
            j.feed_bingo(["[BINGO ARM] t=%d chip=00 ep=1 n_task=390" % (300 * US),
                          prog(800 * US, "00", 1, 390, 390, 224, 124, 123, cells=" 0.4:w0@-,r0@-,c1@1544,x1"),
                          prog(2000 * US, "00", 1, 390, 390, 224, 124, 123, cells=" 0.4:w0@-,r0@-,c1@1544,x1")])
            self.assertEqual(j.busy(), [])
            self.assertTrue(j.all_complete())
            self.assertIsNone(j.verdict())             # 1200 us of quiet after the list drained is not a hang
        finally:
            shutil.rmtree(d)

    def test_open_task_that_is_not_the_exit_keeps_the_chip_busy(self):
        d = tempfile.mkdtemp()
        try:
            j = W.Judge(W.load_params("default", ["stall_floor_us=300"]), W.Names(self._dfg(d)))
            j.feed_bingo([prog(800 * US, "00", 1, 390, 390, 224, 124, 123, cells=" 0.2:w0@-,r0@-,c1@77,x1"),
                          prog(1200 * US, "00", 1, 390, 390, 224, 124, 123, cells=" 0.2:w0@-,r0@-,c1@77,x1")])
            self.assertEqual(len(j.busy()), 1)
            self.assertEqual(j.verdict()[0], "HANG-SCHED")   # a GEMM that never finishes is a hang
        finally:
            shutil.rmtree(d)

    def test_without_dfg_only_the_host_lane_may_hold_the_last_task(self):
        j = self.judge()
        j.feed_bingo([prog(800 * US, "00", 1, 10, 10, 9, 5, 4, cells=" 0.4:w0@-,r0@-,c1@99,x1")])
        self.assertEqual(j.busy(), [])
        j = self.judge()
        j.feed_bingo([prog(800 * US, "00", 1, 10, 10, 9, 5, 4, cells=" 0.1:w0@-,r0@-,c1@99,x1 0.4:w0@-,r0@-,c0@-,x0")])
        self.assertEqual(len(j.busy()), 1)
        j = self.judge()
        j.feed_bingo([prog(800 * US, "00", 1, 9, 10, 9, 5, 4, cells=" 0.4:w0@-,r0@-,c1@99,x1")])
        self.assertEqual(len(j.busy()), 1)            # the list is not fully fetched yet

    def test_tail_reads_only_new_complete_lines(self):
        d = tempfile.mkdtemp()
        try:
            f = os.path.join(d, "l.log")
            t = W.Tail(f)
            self.assertEqual(t.read(), [])
            write(f, "w", "a\nb\nc")
            self.assertEqual(t.read(), ["a", "b"])
            write(f, "a", "d\ne\n")
            self.assertEqual(t.read(), ["cd", "e"])
            os.remove(f)
            write(f, "w", "x\n")                             # replaced file
            self.assertEqual(t.read(), ["x"])
        finally:
            shutil.rmtree(d)


class ProcessTest(unittest.TestCase):
    """A real PID named occamy_chip.vcs in <run>/task_0/bin, a scripted log, the watchdog as a subprocess."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.run = os.path.join(self.d, "runs", "t1")
        self.bin = os.path.join(self.run, "task_0_x", "bin")
        os.makedirs(self.bin)
        shutil.copy("/bin/sleep", os.path.join(self.bin, "occamy_chip.vcs"))
        self.sim = None
        self.wd = None

    def tearDown(self):
        for p in (self.sim, self.wd):
            if p and p.poll() is None:
                p.kill()
            if p:
                p.wait()
                if p.stdout:
                    p.stdout.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def start(self, secs=120):
        self.sim = subprocess.Popen([os.path.join(self.bin, "occamy_chip.vcs"), str(secs)], cwd=self.bin)

    def log(self, name, lines):
        with open(os.path.join(self.bin, name), "a") as f:
            f.write("\n".join(lines) + "\n")

    def watch(self, *sets):
        args = [sys.executable, os.path.join(HERE, "sim_watchdog.py"), "watch", "--run-dir", self.run]
        for s in NOQUOTA + ["interval_s=1", "stop_grace_s=5"] + list(sets):
            args += ["--set", s]
        self.wd = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def events(self):
        try:
            with open(self.run + ".events") as f:
                return f.read()
        except OSError:
            return ""

    def wait_for(self, text, secs=30):
        end = time.time() + secs
        while time.time() < end:
            if text in self.events():
                return True
            time.sleep(0.5)
        return False

    def test_hang_is_stopped_and_documented(self):
        self.start()
        self.log("bingo_progress.log", [prog(100 * US, "00", 1, 5, 100, 5, 5, 4, cells=" 0.1:w1@3,r0@-,c1@3,x1"),
                                          prog(900 * US, "00", 1, 5, 100, 5, 5, 4, cells=" 0.1:w1@3,r0@-,c1@3,x1")])
        self.watch("stall_floor_us=300")
        self.assertTrue(self.wait_for("STOP HANG-SCHED"), self.events())
        self.sim.wait(timeout=15)
        self.assertIsNotNone(self.sim.returncode)
        self.assertIn("simulator", self.events())
        with open(self.run + ".hang.txt") as f:
            self.assertIn("chip 00", f.read())
        self.wd.wait(timeout=30)

    def test_healthy_run_is_left_alone_until_the_simulator_exits(self):
        self.start(secs=6)
        self.log("bingo_progress.log", ["[BINGO ARM] t=%d chip=00 ep=1 n_task=3" % (10 * US),
                                         prog(20 * US, "00", 1, 1, 3, 1, 1, 1)])
        self.watch("stall_floor_us=300")
        time.sleep(2)
        self.log("bingo_progress.log", [prog(60 * US, "00", 1, 3, 3, 3, 3, 3),
                                         "[BINGO CMPL] chip=00 t=%d ep=1 armed=0 fetch=3/3 rel=3 start=3 done=3 quiet=0 anom=0" % (70 * US)])
        self.sim.wait(timeout=20)
        self.wd.wait(timeout=30)
        self.assertNotIn("STOP", self.events())
        self.assertIn("gone", self.events())

    def test_assertion_in_the_transcript_stops_the_run(self):
        self.start()
        self.log("bingo_progress.log", [prog(100 * US, "00", 1, 1, 9, 1, 1, 1)])
        self.watch()
        time.sleep(3)
        self.log("sim_live.log", ['Error: "axi_x.sv", 12: tb: at time 123 ps'])
        self.assertTrue(self.wait_for("STOP ASSERT"), self.events())

    def test_wedge_when_nothing_moves(self):
        self.start()
        self.log("bingo_progress.log", [prog(100 * US, "00", 1, 1, 9, 1, 1, 1)])
        self.watch("wedge_min=0.05", "stall_floor_us=100000", "stall_cap_us=0")
        self.assertTrue(self.wait_for("STOP WEDGE", 40), self.events())

    def test_status_line_and_stale_flag(self):
        self.start()
        self.log("bingo_progress.log", [prog(100 * US, "00", 1, 5, 100, 5, 5, 4)])
        self.watch("stall_floor_us=100000", "stall_cap_us=0")
        time.sleep(4)
        out = subprocess.run([sys.executable, os.path.join(HERE, "sim_watchdog.py"), "status", self.run],
                             capture_output=True, text=True).stdout
        self.assertIn("busy 1/1 chips", out)
        self.assertIn("fetched 5/100", out)
        self.assertNotIn("STALE", out)
        self.wd.kill(); self.wd.wait()                                   # the watchdog dies, the simulator lives
        with open(self.run + ".watchdog.json") as f:
            s = json.load(f)
        s["updated"] -= 600
        with open(self.run + ".watchdog.json", "w") as f:
            json.dump(s, f)
        out = subprocess.run([sys.executable, os.path.join(HERE, "sim_watchdog.py"), "status", self.run],
                             capture_output=True, text=True).stdout
        self.assertIn("STALE", out)

    def test_never_touches_a_simulator_in_another_directory(self):
        other = os.path.join(self.d, "other")
        os.makedirs(other)
        shutil.copy("/bin/sleep", os.path.join(other, "occamy_chip.vcs"))
        stranger = subprocess.Popen([os.path.join(other, "occamy_chip.vcs"), "60"], cwd=other)
        try:
            self.start()
            self.log("bingo_progress.log", [prog(100 * US, "00", 1, 5, 100, 5, 5, 4), prog(900 * US, "00", 1, 5, 100, 5, 5, 4)])
            self.watch("stall_floor_us=300")
            self.assertTrue(self.wait_for("STOP HANG-SCHED"), self.events())
            self.sim.wait(timeout=15)
            time.sleep(1)
            self.assertIsNone(stranger.poll(), "the watchdog stopped a simulator of another run")
        finally:
            stranger.kill()
            stranger.wait()


if __name__ == "__main__":
    unittest.main()
