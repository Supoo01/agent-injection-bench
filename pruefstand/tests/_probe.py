"""Subprozess-Probe fuer test_pruefstand.py (eine Hermes-Version je Prozess).

Aufruf: venv/bin/python pruefstand/tests/_probe.py <check> <outdir> -> eine JSON-Zeile.
Ohne LLM: der Agent ist der Skript-Agent aus p3g_adapter.runner (replay-obedient =
fuehrt die Ground-Truth der Injection-Task aus, sobald er eine Injektion sieht).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from pruefstand.plan import build_plan, item_key  # noqa: E402
from pruefstand.version import erfasse  # noqa: E402

SPECS = {
    "r4": {"name": "t_r4", "modell": "basis", "gateway": "r4"},
    "aus": {"name": "t_aus", "modell": "basis", "gateway": "aus"},
    "critic_v2": {"name": "t_c2", "modell": "basis",
                  "gateway": {"commit": "7df656f4a5", "kurz": "r4",
                              "schalter": {"taint_tracking": True, "session_rule": True, "fail_closed": True,
                                           "verfeinerung": "r4"}},
                  "critic": {"version": "v2"}},
}


def mini_plan(items):
    plan = build_plan("schnell")
    plan["items"] = []
    for suite, u, i, a in items:
        it = {"suite": suite, "user_task": u, "injection_task": i, "attack": a, "rep": 1}
        it["key"] = item_key(it)
        plan["items"].append(it)
    return plan


def replay_factory(suite_names):
    from agentdojo.task_suite.load_suites import get_suite

    from p3g_adapter.runner import ReplayLLM, first_injected_call

    suites = {s: get_suite("v1.2.2", s) for s in suite_names}
    after = {}

    def factory(user_task, injection_task, env):
        calls = user_task.ground_truth(env.model_copy(deep=True))
        inj_calls, aft = None, None
        if injection_task is not None:
            inj_calls = injection_task.ground_truth(env.model_copy(deep=True))
            for name, s in suites.items():
                if type(s.user_tasks.get(user_task.ID)) is type(user_task):
                    if (name, user_task.ID) not in after:
                        after[(name, user_task.ID)] = first_injected_call(s, user_task)
                    aft = after[(name, user_task.ID)]
        return ReplayLLM(calls, user_task.GROUND_TRUTH_OUTPUT, inj_calls, aft)

    return factory


def main():
    check, out = sys.argv[1], Path(sys.argv[2])
    from pruefstand import lauf

    rec = erfasse(SPECS[check])
    items = [("slack", "user_task_1", None, None),
             ("slack", "user_task_1", "injection_task_2", "important_instructions"),
             ("slack", "user_task_1", "injection_task_2", "tool_knowledge"),
             ("banking", "user_task_0", "injection_task_0", "ignore_previous")]
    critic_calls = []
    plan = mini_plan(items)
    if check == "critic_v2":
        # Critic-Endpunkt ersetzen: jede Anfrage -> passt_nicht (kein Netz)
        lauf.condition_from_version(rec)
        import agent.session_gateway_critic as cm

        def fake(messages, cfg):
            critic_calls.append({"n_msgs": len(messages), "base_url": cfg["base_url"], "model": cfg["model"],
                                 "v2": "bisherige_schritte" in messages[-1]["content"]})
            return "passt_nicht", "Testantwort"

        cm.call_critic = fake
        lauf.condition_from_version = lambda r: lauf.CONDITION
    meta = lauf.run(rec, plan, out, base_url="http://localhost:8012/v1", workers=2, frist_min=10,
                    make_llm=replay_factory(["slack", "banking"]))
    rows = [json.loads(l) for l in (out / "ergebnisse.jsonl").read_text().splitlines() if l.strip()]
    # Fortsetzung: zweiter Aufruf darf nichts neu rechnen
    meta2 = lauf.run(rec, plan, out, base_url="http://localhost:8012/v1", workers=2, frist_min=10,
                     make_llm=replay_factory(["slack", "banking"])) if check != "critic_v2" else {"neu": 0}
    print(json.dumps({
        "version_id": rec["version_id"], "gateway": rec["gateway"] and rec["gateway"]["commit"],
        "meta": meta, "neu_bei_fortsetzung": meta2["neu"],
        "rows": [{k: r.get(k) for k in ("key", "utility", "security", "blocked", "error", "critic", "attack")} for r in rows],
        "critic_calls": critic_calls,
    }))


if __name__ == "__main__":
    main()
