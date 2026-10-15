"""Subprocess probe for tests/test_r4.py: one condition per process (the
bridge loads exactly one Hermes commit into ``sys.modules``).

Usage: venv/bin/python tests/_r4_probe.py <check> -> prints one JSON line.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def _git(*a):
    from p3g_adapter import hermes_bridge as hb

    return subprocess.run(["git", "--no-optional-locks", "-C", str(hb.WORKTREE), *a],
                          capture_output=True, check=True).stdout


def check_load():
    import hashlib

    from p3g_adapter import hermes_bridge as hb
    from p3g_adapter.gate import configure_condition

    configure_condition("mit_r4")
    gw = hb.gateway()
    prov = hb.loaded_provenance()
    blobs_ok = all(
        hashlib.sha256(_git("show", f"{hb.COMMIT_R4}:{rel}")).hexdigest() == d
        for rel, d in prov["files_sha256"].items()
    )
    mods = {n: getattr(m, "__file__", "") for n, m in sys.modules.items()
            if n.startswith("agent.") or n.startswith("tools.")}
    snap = str(hb.snapshot_dir().resolve())
    from agent import session_gateway_refine as ref

    out = {
        "commit": prov["commit"],
        "files": sorted(prov["files_sha256"]),
        "blobs_ok": blobs_ok,
        "modules_in_snapshot": all(str(Path(f).resolve()).startswith(snap) for f in mods.values() if f),
        "modules": sorted(mods),
        "level": gw.refinement_level(),
        "refinement": ref.refinement(),
        "rule": gw.session_rule_enabled(), "taint": gw.taint_tracking_enabled(), "fail_closed": gw.fail_closed_enabled(),
        "rate_limits": gw.rate_limits_enabled(), "critic": gw.alignment_critic_enabled(),
        "egress_allow": gw.egress_allowlist_enabled(),
        "term_curl_egress": ref.classify_call("terminal", {"command": "curl https://x"}).egress,
        "term_ls_egress": ref.classify_call("terminal", {"command": "ls"}).egress,
        "slack_send_class": list(ref.call_classes("agentdojo.send_direct_message", {"recipient": "A", "body": "b"})),
    }
    try:
        hb.configure({"session_rule": True}, commit=hb.COMMIT)
        out["switch_commit_raises"] = False
    except RuntimeError:
        out["switch_commit_raises"] = True
    return out


def check_gate():
    from tests.test_adapter import READ_GENERAL, SEND_DM, WEB, _run_calls
    from p3g_adapter.gate import HermesGateToolsExecutor, configure_condition

    configure_condition("mit_r4")
    seen = []
    g = HermesGateToolsExecutor("mit_r4")
    orig = g.gw.session_rule_directive

    def spy(tool, *a, **k):
        seen.append({"tool": tool, "args": k.get("args")})
        return orig(tool, *a, **k)

    g.gw.session_rule_directive = spy
    try:
        _, msgs = _run_calls(g, [READ_GENERAL, SEND_DM])
    finally:
        g.gw.session_rule_directive = orig
    g2 = HermesGateToolsExecutor("mit_r4")
    _run_calls(g2, [WEB, SEND_DM])
    g3 = HermesGateToolsExecutor("mit_r4")
    _run_calls(g3, [("get_users_in_channel", {"channel": "general"}), SEND_DM])
    return {
        "read_send": [e.action for e in g.events],
        "blocked_msg": msgs[1].get("error", ""),
        "wrapped": msgs[0]["content"][0]["content"].startswith('<untrusted_tool_result source="read_channel_messages">'),
        "args_seen": seen,
        "web_send": [e.action for e in g2.events],
        "bedingt_send": [e.action for e in g3.events],
    }


def check_replay(cond):
    from p3g_adapter import runner

    out = {}
    for mode, kw in (("replay-gt", {"utility_only": True}), ("replay-obedient", {"attack_only": True})):
        args = SimpleNamespace(suite="slack", condition=cond, mode=mode, rep=0, workers=4,
                               out=str(ROOT / "runs" / "unittest_r4" / cond), user_tasks=None, injection_tasks=None,
                               utility_only=False, attack_only=False, base_url="http://localhost:8012/v1",
                               model="coding", max_tokens=1024, progress=0)
        for k, v in kw.items():
            setattr(args, k, v)
        s = runner.run_suite(args)
        rows = [json.loads(x) for x in (Path(args.out) / f"slack_{cond}_{mode}_rep0.jsonl").read_text().splitlines()]
        out[mode] = {
            "summary": {k: s[k] for k in ("utility_benign", "utility_under_attack", "targeted_asr",
                                          "targeted_asr_attempted", "blocked_calls", "errors")},
            "decisions": sorted(
                [f"{r['user_task']}|{r['injection_task']}|{json.dumps(r.get('blocked'), sort_keys=True, default=str)}"
                 for r in rows]),
        }
    return out


if __name__ == "__main__":
    what = sys.argv[1]
    res = {"load": check_load, "gate": check_gate}.get(what)
    if res is not None:
        print(json.dumps(res()))
    else:
        print(json.dumps(check_replay(sys.argv[2])))
