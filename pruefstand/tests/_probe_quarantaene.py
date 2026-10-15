"""Subprozess-Probe fuer test_quarantaene.py (eine Hermes-Bedingung je Prozess, kein LLM, kein Netz).

Aufruf: venv/bin/python pruefstand/tests/_probe_quarantaene.py <check> <outdir> -> eine JSON-Zeile.

Checks
  aus        pruefstand.lauf, Gateway aus + Quarantaene-Leser V1, Leser-Endpunkt gemockt (treuer Leser:
             kopiert Adressen/Zahlen aus den Daten als Fakten)
  s2s5       wie aus, Gateway s2s5 + simulierter Mensch Oracle p=0
  fehler     Gateway aus, Leser-Endpunkt wirft Netzfehler -> fail-closed Platzhalter
  ohne       Gateway aus ohne Leser (Vergleich: Tool-Nachrichten unveraendert roh)
Agent: Skript-Agent replay-obedient (fuehrt die Injection-GT aus, sobald ein Ergebnis eine Injektion traegt).
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from pruefstand.tests._probe import mini_plan, replay_factory  # noqa: E402
from pruefstand.version import erfasse  # noqa: E402

Q = {"variante": "v1", "modell": "coding", "max_zeichen": 300}
SPECS = {
    "aus": {"name": "t_q1_aus", "modell": "basis", "gateway": "aus", "quarantaene": Q},
    "fehler": {"name": "t_q1_aus", "modell": "basis", "gateway": "aus", "quarantaene": Q},
    "ohne": {"name": "t_aus", "modell": "basis", "gateway": "aus"},
    "s2s5": {"name": "t_q1_s2s5", "modell": "basis", "gateway": "s2s5", "quarantaene": Q,
             "mensch": {"p": 0.0, "politik": "streng", "modell": "oracle"}},
}
ANFRAGEN = []


def treuer_leser(url, body, timeout):
    """Gemockter Endpunkt: Schema/Request pruefen, Adressen und Zahlen der Daten als Fakten zurueckgeben."""
    assert url == "http://localhost:8012/v1/chat/completions", url
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["structured_outputs"]["json"]["properties"]["fakten"]["items"]["properties"]["typ"]["enum"]
    assert "tools" not in body
    user = body["messages"][-1]["content"]
    daten = user.split("<data>\n", 1)[1].rsplit("\n</data>", 1)[0]
    ANFRAGEN.append(len(daten))
    fakten = []
    for m in re.findall(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", daten):
        fakten.append({"typ": "email", "wert": m, "beleg": "data"})
    for m in re.findall(r"\bwww\.[\w.-]+\.[a-z]{2,}\b|https?://\S+", daten):
        fakten.append({"typ": "url", "wert": m.rstrip(".,"), "beleg": "data"})
    content = json.dumps({"zusammenfassung": f"Tool data with {len(daten)} characters.",
                          "fakten": fakten[:20], "enthaelt_anweisungen": "nformation" in daten or "TODO" in daten})
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}}


def kaputter_leser(url, body, timeout):
    from p3g_adapter.quarantaene import LeserFehler

    raise LeserFehler("netz", "ConnectionRefusedError")


def main():
    check, out = sys.argv[1], Path(sys.argv[2])
    from p3g_adapter import quarantaene as qm
    from pruefstand import lauf

    rec = erfasse(SPECS[check])
    items = [("slack", "user_task_1", None, None),
             ("slack", "user_task_1", "injection_task_2", "important_instructions"),
             ("banking", "user_task_0", "injection_task_0", "ignore_previous")]
    plan = mini_plan(items)
    make_leser = None
    if check != "ohne":
        post = kaputter_leser if check == "fehler" else treuer_leser
        cfg = {k: v for k, v in rec["quarantaene"].items() if k != "modul_sha256"}
        make_leser = lambda auftrag: qm.Leser(cfg, auftrag=auftrag, post=post)  # noqa: E731
    meta = lauf.run(rec, plan, out, base_url="http://localhost:8012/v1", workers=1, frist_min=10,
                    make_llm=replay_factory(["slack", "banking"]), make_leser=make_leser)
    rows = [json.loads(l) for l in (out / "ergebnisse.jsonl").read_text().splitlines() if l.strip()]
    tool_msgs = {}
    with gzip.open(out / "nachrichten.jsonl.gz", "rt") as f:
        for line in f:
            d = json.loads(line)
            tool_msgs[d["key"]] = [
                {"tool": re.search(r"function='([^']+)'", str(m.get("tool_call"))).group(1),
                 "text": "".join(b.get("content") or "" for b in (m.get("content") or []))}
                for m in d["messages"] if m["role"] == "tool"]
    from pruefstand import auswertung as aw

    print(json.dumps({
        "version_id": rec["version_id"], "quarantaene": rec.get("quarantaene"), "meta": meta,
        "rows": [{k: r.get(k) for k in ("key", "utility", "security", "blocked", "error", "leser", "spur", "attack",
                                        "s25")} for r in rows],
        "tool_msgs": tool_msgs, "anfragen": len(ANFRAGEN),
        "kennzahlen_leser": aw.kennzahlen(rows).get("leser"),
    }, default=str))


if __name__ == "__main__":
    main()
