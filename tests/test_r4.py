"""Unit tests for the ``mit_r4`` condition (Hermes d8c3aafb22, verfeinerung r4).

The bridge loads one Hermes commit per process, and test_adapter.py loads
c8341ea748; so every check runs in a fresh subprocess (tests/_r4_probe.py).
Run: venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from p3g_adapter import hermes_bridge  # noqa: E402
from p3g_adapter.gate import CONDITIONS  # noqa: E402

PROBE = ROOT / "tests" / "_r4_probe.py"


def probe(*args):
    out = subprocess.run([sys.executable, str(PROBE), *args], cwd=str(ROOT), capture_output=True, text=True,
                         timeout=600, env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin",
                                           "HOME": str(Path.home())})
    if out.returncode != 0:
        raise AssertionError(out.stderr[-3000:])
    return json.loads(out.stdout.strip().splitlines()[-1])


def wt_status():
    return subprocess.run(["git", "--no-optional-locks", "-C", str(hermes_bridge.WORKTREE), "status", "--porcelain"],
                          capture_output=True, text=True, check=True).stdout


class ConditionTableTest(unittest.TestCase):
    def test_mit_r4_switches(self):
        c = CONDITIONS["mit_r4"]
        self.assertEqual(c["switches"], {"taint_tracking": True, "session_rule": True, "fail_closed": True,
                                         "verfeinerung": "r4"})
        self.assertTrue(c["rule"] and c["mark"] and c["pass_args"])
        self.assertEqual(c["commit"], "d8c3aafb22")
        self.assertIsNone(CONDITIONS["mit"].get("commit"), "mit stays on c8341ea748 (comparison runs)")


class R4LoadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wt_before = wt_status()
        cls.r = probe("load")

    def test_commit_and_sha256(self):
        self.assertTrue(self.r["commit"].startswith("d8c3aafb22"))
        self.assertTrue(self.r["blobs_ok"])
        self.assertIn("agent/session_gateway_refine.py", self.r["files"])

    def test_only_snapshot_modules(self):
        self.assertTrue(self.r["modules_in_snapshot"])
        self.assertIn("agent.session_gateway_refine", self.r["modules"])

    def test_switches_and_level(self):
        self.assertEqual(self.r["level"], 4)
        self.assertEqual(self.r["refinement"], "r4")
        self.assertTrue(self.r["rule"] and self.r["taint"] and self.r["fail_closed"])
        self.assertFalse(self.r["rate_limits"] or self.r["critic"] or self.r["egress_allow"])

    def test_refinement_code_active(self):
        self.assertTrue(self.r["term_curl_egress"])
        self.assertFalse(self.r["term_ls_egress"])
        self.assertEqual(self.r["slack_send_class"], [True, True])

    def test_one_commit_per_process(self):
        self.assertTrue(self.r["switch_commit_raises"])

    def test_worktree_untouched(self):
        self.assertEqual(wt_status(), self.wt_before)


class R4GateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = probe("gate")

    def test_tainted_sensitive_blocks_send(self):
        self.assertEqual(self.r["read_send"], ["executed", "blocked"])
        self.assertIn("BLOCKED: approval required", self.r["blocked_msg"])
        self.assertTrue(self.r["wrapped"])

    def test_args_reach_session_rule(self):
        self.assertEqual(self.r["args_seen"][1], {"tool": "agentdojo.send_direct_message",
                                                  "args": {"recipient": "Alice", "body": "hi"}})

    def test_taint_only_runs_and_bedingt_blocks(self):
        self.assertEqual(self.r["web_send"], ["executed", "executed"])
        self.assertEqual(self.r["bedingt_send"], ["executed", "blocked"])


class R4ReplayEquivalenceTest(unittest.TestCase):
    """r4 refines only terminal/execute_code/read_file/search_files/delegate_task;
    AgentDojo tools keep their axes -> same gate decisions as streng (c8341ea748)."""

    def test_same_decisions_as_streng(self):
        a, b = probe("replay", "mit"), probe("replay", "mit_r4")
        for mode in ("replay-gt", "replay-obedient"):
            self.assertEqual(a[mode]["summary"], b[mode]["summary"], mode)
            self.assertEqual(a[mode]["decisions"], b[mode]["decisions"], mode)
        self.assertGreater(b["replay-obedient"]["summary"]["blocked_calls"], 0)


if __name__ == "__main__":
    unittest.main()
