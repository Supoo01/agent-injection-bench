"""Subprozess-Probe fuer test_bruecke.py: K5-Fall (Slack user_task_4) am Gate mit/ohne Bruecke url_ziele."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def main(mit: bool):
    from agentdojo.agent_pipeline.agent_pipeline import load_system_message
    from agentdojo.task_suite.load_suites import get_suite

    from p3g_adapter import gate as G
    from pruefstand.lauf import condition_from_version
    from pruefstand.version import erfasse

    rec = erfasse({"name": "t", "modell": "basis", "gateway": "s2s5"}, docker=lambda *a, **k: "img")
    cond = condition_from_version(rec)
    tools = G.url_ziele_aktivieren() if mit else []
    sm = load_system_message(None)
    prompt = get_suite("v1.2.2", "slack").get_user_task_by_id("user_task_4").PROMPT

    class Ja:
        def answer(self, *a, **k):
            return True

    g = G.HermesGateToolsExecutor(cond)
    g.note_turn_start(sm, prompt)
    g.state.mark(tainted=True, sensitive=True, source="agentdojo.read_inbox")
    g.human = Ja()
    out = []
    for url in ("www.eve-blog.com", "www.true-informations.com", "www.eve-blog.com"):
        g._last_approval = None
        d = g._decide("get_webpage", {"url": url})
        out.append({"url": url, "frei": d is None, "freigabe": g._last_approval})
    print(json.dumps({"tools": tools, "aufrufe": out, "liste": sorted(g.state.egress.entries)}))


if __name__ == "__main__":
    main(sys.argv[1] == "mit")
