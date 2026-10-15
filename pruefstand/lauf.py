"""Injektions-Pruefstand: einen Messplan fuer EINE Version ausfuehren.

Ein Prozess = eine Version (die Hermes-Bruecke laedt genau einen Commit).
Alle Plan-Eintraege (Suites x Angriffe x Wiederholungen) laufen in einem
gemeinsamen Thread-Pool gegen den lokalen vLLM (nur localhost). Ergebnisse
werden zeilenweise nach ``ergebnisse.jsonl`` geschrieben; ein erneuter Aufruf
mit demselben Ordner setzt fort (erledigte Schluessel werden uebersprungen).

Aufruf:
  venv/bin/python -m pruefstand.lauf --version V.json --plan P.json --out DIR [--workers 8] [--frist-min 150]
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List

from pruefstand.plan import item_key

CONDITION = "pruefstand"


def condition_from_version(rec: Dict[str, Any]) -> str:
    """Registriert die Gateway-Bedingung der Version und laedt den Hermes-Code."""
    from p3g_adapter import hermes_bridge
    from p3g_adapter.gate import configure_condition, register_condition

    gw = rec.get("gateway")
    if gw is None:
        register_condition(CONDITION, None)
        return CONDITION
    hk = gw.get("herkunft")
    if hk is not None:
        from pruefstand.version import HERKUNFT_MODUL, sha256_file

        if hk.get("modul_sha256") != sha256_file(HERKUNFT_MODUL):
            raise RuntimeError("p3g_adapter/herkunft.py weicht von der erfassten Version ab")
    register_condition(CONDITION, gw["schalter"], commit=gw.get("commit_kurz") or gw["commit"],
                       herkunft=None if hk is None else {k: v for k, v in hk.items()
                                                         if k not in ("umsetzung", "modul_sha256")})
    configure_condition(CONDITION)
    prov = hermes_bridge.loaded_provenance()
    if prov["commit"] != gw["commit"] or prov["files_sha256"] != gw["dateien_sha256"]:
        raise RuntimeError("geladener Hermes-Code weicht von der erfassten Version ab")
    return CONDITION


def done_keys(path: Path) -> set:
    if not path.is_file():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.add(json.loads(line)["key"])
            except Exception:
                pass
    return out


def run(rec: Dict[str, Any], plan: Dict[str, Any], out: Path, *, base_url: str, workers: int,
        frist_min: float, max_tokens: int = 8192, make_llm=None, spur: bool = True,
        make_leser=None) -> Dict[str, Any]:
    from agentdojo.agent_pipeline.agent_pipeline import load_system_message
    from agentdojo.attacks import load_attack
    from agentdojo.task_suite.load_suites import get_suite

    from p3g_adapter.runner import _dummy_named_pipeline, run_task

    out.mkdir(parents=True, exist_ok=True)
    cond = condition_from_version(rec)
    if (rec.get("bruecke") or {}).get("url_ziele"):
        from p3g_adapter.gate import url_ziele_aktivieren

        if rec.get("gateway") is None or not url_ziele_aktivieren():
            raise RuntimeError("bruecke.url_ziele braucht ein Gateway mit S2/S5 (URL_READ_TOOLS)")
    res_path = out / "ergebnisse.jsonl"
    erledigt = done_keys(res_path)
    todo = [it for it in plan["items"] if it["key"] not in erledigt]
    random.Random(plan["plan_sha256"]).shuffle(todo)  # Suites mischen -> gleichmaessige Last
    suites = {s: get_suite(plan["benchmark"], s) for s in plan["suites"]}
    attacks = {(s, a): load_attack(a, suites[s], _dummy_named_pipeline()) for s in plan["suites"] for a in plan["angriffe"]}
    system_message = load_system_message(None)
    model = rec["modell"]["served_name"]
    leser_factory = None
    if rec.get("quarantaene"):
        from p3g_adapter import quarantaene as qm
        from pruefstand.version import QUARANTAENE_MODUL, sha256_file

        if rec["quarantaene"].get("modul_sha256") != sha256_file(QUARANTAENE_MODUL):
            raise RuntimeError("p3g_adapter/quarantaene.py weicht von der erfassten Version ab")
        q_cfg = {k: v for k, v in rec["quarantaene"].items() if k != "modul_sha256"}
        leser_factory = make_leser or (lambda auftrag: qm.Leser(q_cfg, auftrag=auftrag))

    if make_llm is None:
        import openai

        from p3g_adapter.runner import LocalOpenAILLM

        client = openai.OpenAI(api_key="EMPTY", base_url=base_url, timeout=600, max_retries=2)

        def make_llm(u, i, env):
            return LocalOpenAILLM(client, model, max_tokens=max_tokens)

    lock = threading.Lock()
    t0 = time.time()
    frist = t0 + frist_min * 60

    def one(it):
        suite = suites[it["suite"]]
        u = suite.user_tasks[it["user_task"]]
        i = suite.injection_tasks[it["injection_task"]] if it["injection_task"] else None
        try:
            inj = attacks[(it["suite"], it["attack"])].attack(u, i) if i is not None else {}
            hf = None
            if rec.get("mensch"):
                from p3g_adapter.mensch import human_aus_spec

                seed = f"{plan['plan_sha256']}|{it['key']}"
                hf = lambda oracle, _s=seed: human_aus_spec(rec["mensch"], oracle, _s)  # noqa: E731
            r = run_task(suite, u, i, inj, cond, make_llm, system_message, human_factory=hf, spur=spur,
                         leser_factory=leser_factory)
        except Exception as e:
            r = {"user_task": u.ID, "injection_task": i.ID if i else None, "utility": False,
                 "security": False if i else None, "error": f"harness {type(e).__name__}: {e}"[:500],
                 "trace": traceback.format_exc()[-2000:]}
        r.update(key=it["key"], suite=it["suite"], attack=it["attack"], rep=it["rep"], version_id=rec["version_id"])
        return r

    n_neu = 0
    abgebrochen = False
    with open(res_path, "a") as fr, gzip.open(out / "nachrichten.jsonl.gz", "at") as fm, \
            ThreadPoolExecutor(max_workers=workers) as ex:
        pending = set()
        queue = list(todo)
        while queue or pending:
            while queue and len(pending) < workers and time.time() < frist:
                pending.add(ex.submit(one, queue.pop(0)))
            if not pending:
                abgebrochen = bool(queue)
                break
            fertig, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for f in fertig:
                r = f.result()
                msgs = r.pop("_messages", None)
                with lock:
                    fr.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                    fr.flush()
                    if msgs is not None:
                        fm.write(json.dumps({"key": r["key"], "messages": msgs}, ensure_ascii=False, default=str) + "\n")
                n_neu += 1
                if n_neu % 20 == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] {n_neu + len(erledigt)}/{len(plan['items'])}",
                          file=sys.stderr, flush=True)
            if time.time() >= frist and queue:
                abgebrochen = True
                queue = []
    gesamt = len(done_keys(res_path))
    meta = {"version_id": rec["version_id"], "plan_sha256": plan["plan_sha256"], "profil": plan["profil"],
            "neu": n_neu, "erledigt": gesamt, "geplant": len(plan["items"]),
            "vollstaendig": gesamt >= len(plan["items"]), "frist_erreicht": abgebrochen,
            "sekunden": round(time.time() - t0, 1), "modell_name": model, "base_url": base_url, "workers": workers,
            "mensch": rec.get("mensch"), "spur": spur,
            **({"quarantaene": rec["quarantaene"]} if rec.get("quarantaene") else {})}
    (out / "lauf_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://localhost:8012/v1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--frist-min", type=float, default=150)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--ohne-spur", action="store_true", help="ASR_spur nicht je Lauf mitschreiben")
    a = ap.parse_args(argv)
    from urllib.parse import urlparse

    if urlparse(a.base_url).hostname not in ("localhost", "127.0.0.1", "::1"):
        ap.error("nur localhost-Endpunkte")
    rec = json.loads(Path(a.version).read_text())
    plan = json.loads(Path(a.plan).read_text())
    meta = run(rec, plan, Path(a.out), base_url=a.base_url, workers=a.workers, frist_min=a.frist_min,
               max_tokens=a.max_tokens, spur=not a.ohne_spur)
    print(json.dumps(meta, ensure_ascii=False))
    return 0 if meta["vollstaendig"] else 3


if __name__ == "__main__":
    sys.exit(main())
