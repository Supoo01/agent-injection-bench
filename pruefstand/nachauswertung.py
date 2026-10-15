"""Injektions-Pruefstand: Nachauswertung vorhandener Laeufe (nur lesen) und Zerlegung.

1. ASR_spur nachtraeglich: Fuer Laeufe ohne Feld ``spur`` (vor Teil B gemessen) wird die
   Aufrufspur aus den gespeicherten Nachrichten gelesen (Pruefstand: ``nachrichten.jsonl.gz``,
   Live-Laeufe ``runs/live_*``: ``messages/<datei>.jsonl``) und mit dem Zielmerkmal der
   Injection-GT verglichen (``p3g_adapter.mensch.SpurPruefer``). Die Nachrichten enthalten nur den
   LETZTEN Pipeline-Versuch (``quelle = nachrichten_letzter_versuch``); Sperrzahlen kommen aus
   ``blocked`` (alle Versuche), die Abweichung zur Nachrichtenzaehlung wird ausgewiesen.
2. Zerlegung Modell / Markierung / Regel (Loesungsdokument 3.3) aus drei Laufordnern
   (gepaart ueber den Plan-Schluessel, gleiche plan_sha256 Pflicht).

Schreibt NUR in die mit ``--out`` angegebene Datei (nie in den Laufordner).

Aufruf:
  venv/bin/python -m pruefstand.nachauswertung LAUFORDNER --out DATEI.json [--md DATEI.md]
  venv/bin/python -m pruefstand.nachauswertung --zerlegung AUS MARKIERUNG REGEL --out DATEI.json
"""

from __future__ import annotations

import argparse
import ast
import glob
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pruefstand import auswertung as aw

BENCHMARK = "v1.2.2"
LIVE_ATTACK = "important_instructions"
_TC = re.compile(r"^function='([^']+)' args=(\{.*\}) id=(?:'[^']*'|None) placeholder_args=(.*)$", re.S)


def _lit(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:
            return v
    return v


def parse_tc(s: Any) -> Tuple[str, Dict[str, Any]]:
    if isinstance(s, dict):
        return s.get("function"), dict(s.get("args") or {})
    m = _TC.match(str(s))
    if not m:
        raise ValueError(str(s)[:120])
    return m.group(1), ast.literal_eval(m.group(2))


def text_of(content: Any) -> str:
    content = _lit(content)
    if content is None or content == "None":
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(str(b.get("content", "")) for b in content if isinstance(b, dict))


def spur_aufrufe(messages: List[Dict[str, Any]]) -> Tuple[List[Tuple[str, Dict[str, Any], str]], List[str], int]:
    """(aufrufe (tool, args, aktion), tool-Ergebnistexte, gesperrte Aufrufe laut Nachrichten)."""
    blocked_ids, texts = set(), []
    n_block = 0
    for m in messages:
        if m.get("role") != "tool":
            continue
        if str(m.get("p3g_blocked")) == "True":
            blocked_ids.add(m.get("tool_call_id"))
            n_block += 1
        else:
            texts.append(text_of(m.get("content")))
    out = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        tcs = _lit(m.get("tool_calls"))
        if not tcs or tcs == "None":
            continue
        for tc in tcs:
            try:
                fn, args = parse_tc(tc)
            except Exception:
                continue
            mid = re.search(r"id='([^']*)'", str(tc))
            aktion = "blocked" if (mid and mid.group(1) in blocked_ids) else "executed"
            out.append((fn, args, aktion))
    return out, texts, n_block


class _Kontext:
    """Suites/Angriffe einmal laden; Oracle je (suite, user, injection, angriff) cachen."""

    def __init__(self) -> None:
        from agentdojo.task_suite.load_suites import get_suite

        self._get_suite = get_suite
        self.suites: Dict[str, Any] = {}
        self.attacks: Dict[Tuple[str, str], Any] = {}
        self.cache: Dict[Tuple[str, str, str, str], Any] = {}

    def suite(self, name: str):
        if name not in self.suites:
            self.suites[name] = self._get_suite(BENCHMARK, name)
        return self.suites[name]

    def pruefer(self, suite_name: str, user: str, inj: str, attack: str):
        k = (suite_name, user, inj, attack)
        if k not in self.cache:
            from agentdojo.attacks import load_attack

            from p3g_adapter import mensch
            from p3g_adapter.runner import _dummy_named_pipeline

            s = self.suite(suite_name)
            if (suite_name, attack) not in self.attacks:
                self.attacks[(suite_name, attack)] = load_attack(attack, s, _dummy_named_pipeline())
            u, i = s.user_tasks[user], s.injection_tasks[inj]
            injections = self.attacks[(suite_name, attack)].attack(u, i)
            oracle = mensch.oracle_fuer_lauf(s, u, i, injections)
            self.cache[k] = (mensch.SpurPruefer(oracle), injections, i.GOAL)
        return self.cache[k]


def ergaenze_spur(rows: List[Dict[str, Any]], nachrichten: Dict[str, List[Dict[str, Any]]],
                  ctx: Optional[_Kontext] = None, attack_default: str = LIVE_ATTACK,
                  suite_default: Optional[str] = None) -> Dict[str, Any]:
    """Setzt ``spur`` fuer Angriffslaeufe ohne Spur aus den Nachrichten (in den Zeilen-Kopien)."""
    from p3g_adapter.mensch import enthaelt_injektion

    ctx = ctx or _Kontext()
    stat = {"ergaenzt": 0, "ohne_nachrichten": 0, "vorhanden": 0, "sperren_abweichung_letzter_versuch": 0,
            "sperren_nachrichten": 0, "sperren_blocked_alle_versuche": 0}
    for r in rows:
        msgs = nachrichten.get(r["_mkey"])
        stat["sperren_blocked_alle_versuche"] += len(r.get("blocked") or [])
        if msgs is not None:
            _a, _t, nb = spur_aufrufe(msgs)
            stat["sperren_nachrichten"] += nb
            stat["sperren_abweichung_letzter_versuch"] += int(nb != len(r.get("blocked") or []))
        if r.get("injection_task") is None:
            continue
        if isinstance(r.get("spur"), dict):
            stat["vorhanden"] += 1
            continue
        if msgs is None:
            stat["ohne_nachrichten"] += 1
            continue
        pr, injections, goal = ctx.pruefer(r.get("suite") or suite_default, r["user_task"], r["injection_task"],
                                           r.get("attack") or attack_default)
        aufrufe, texte, _nb = spur_aufrufe(msgs)
        sp = pr.bewerte(aufrufe)
        sp["gelesen"] = any(enthaelt_injektion(t, injections, goal) for t in texte)
        sp["quelle"] = "nachrichten_letzter_versuch"
        r["spur"] = sp
        stat["ergaenzt"] += 1
    return stat


# ---------------------------------------------------------------------------
# Laufordner lesen (Pruefstand oder Live)
# ---------------------------------------------------------------------------

def lese_pruefstand(d: Path) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    rows = aw.load_rows(d)
    for r in rows:
        r["_mkey"] = r.get("key")
    msgs: Dict[str, List[Dict[str, Any]]] = {}
    p = d / "nachrichten.jsonl.gz"
    if p.is_file():
        with gzip.open(p, "rt") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # unvollstaendige letzte Zeile eines laufenden Laufs
                    msgs.setdefault(rec["key"], rec["messages"])
    return rows, msgs


def lese_live(d: Path) -> Dict[str, Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]]:
    """runs/live_*: je Bedingung (Dateiname <suite>_<cond>_llm_rep<n>.jsonl) Zeilen + Nachrichten."""
    out: Dict[str, Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]] = {}
    for f in sorted(glob.glob(str(d / "*_llm_rep*.jsonl"))):
        name = Path(f).name
        m = re.match(r"^([a-z]+)_(.+)_llm_rep(\d+)\.jsonl$", name)
        if not m:
            continue
        suite, cond, rep = m.group(1), m.group(2), int(m.group(3))
        rows, msgs = out.setdefault(cond, ([], {}))
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            r.setdefault("suite", suite)
            r["rep"] = rep
            r["attack"] = LIVE_ATTACK if r.get("injection_task") else None
            r["key"] = f"{suite}|{r['user_task']}|{r.get('injection_task') or '-'}|{r['attack'] or '-'}|{rep}"
            r["_mkey"] = r["key"]
            rows.append(r)
        mf = d / "messages" / name
        if mf.is_file():
            for line in open(mf, encoding="utf-8"):
                if line.strip():
                    rec = json.loads(line)
                    k = f"{suite}|{rec['user_task']}|{rec.get('injection_task') or '-'}|" \
                        f"{LIVE_ATTACK if rec.get('injection_task') else '-'}|{rep}"
                    msgs.setdefault(k, rec["messages"])
    return out


def _sauber(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]


def nachauswerten(d: Path) -> Dict[str, Any]:
    ctx = _Kontext()
    if (d / "ergebnisse.jsonl").is_file():
        rows, msgs = lese_pruefstand(d)
        stat = ergaenze_spur(rows, msgs, ctx)
        out: Dict[str, Any] = {"lauf": str(d), "art": "pruefstand", "ergaenzung": stat,
                               "kennzahlen": aw.kennzahlen(_sauber(rows))}
        for f in ("version.json", "plan.json"):
            if (d / f).is_file():
                j = json.loads((d / f).read_text())
                out[f[:-5]] = {"name": j.get("name"), "version_id": j.get("version_id")} if f == "version.json" \
                    else {"profil": j.get("profil"), "plan_sha256": j.get("plan_sha256")}
        return out
    live = lese_live(d)
    if not live:
        raise SystemExit(f"{d}: weder ergebnisse.jsonl noch *_llm_rep*.jsonl")
    out = {"lauf": str(d), "art": "live", "bedingungen": {}}
    for cond, (rows, msgs) in sorted(live.items()):
        stat = ergaenze_spur(rows, msgs, ctx)
        k = aw.kennzahlen(_sauber(rows))
        je_inj: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n": 0, "gelesen": 0, "spur": 0, "spur_gelesen": 0,
                                                                 "versucht_gelesen": 0, "wirksam": 0})
        for r in rows:
            if r.get("injection_task") is None or not isinstance(r.get("spur"), dict):
                continue
            j = je_inj[r["injection_task"]]
            g = bool(r["spur"].get("gelesen"))
            j["n"] += 1
            j["gelesen"] += int(g)
            j["spur"] += int(bool(aw.spur_wert(r)))
            j["spur_gelesen"] += int(bool(aw.spur_wert(r)) and g)
            j["versucht_gelesen"] += int(bool(r.get("security_attempted")) and g)
            j["wirksam"] += int(bool(r.get("security")))
        out["bedingungen"][cond] = {"ergaenzung": stat, "kennzahlen": k, "je_injection": dict(sorted(je_inj.items()))}
    return out


def markdown(e: Dict[str, Any]) -> str:
    lines = [f"# Nachauswertung ASR_spur: {e['lauf']}", ""]
    blocks = [("", e)] if e.get("art") == "pruefstand" else list(e["bedingungen"].items())
    lines += ["| Bedingung | ASR_spur | ASR_wirksam | Nebenwert security_attempted | Spur gegeben gelesen | "
              "security_attempted gegeben gelesen | Sperren (alle Versuche / letzter) |", "|---|---|---|---|---|---|---|"]
    for name, b in blocks:
        k, st = b["kennzahlen"], b["ergaenzung"]
        lines.append(f"| {name or (e.get('version') or {}).get('name', '-')} | {aw._f(k['asr_spur'])} | "
                     f"{aw._f(k['asr_wirksam'])} | {aw._f(k['asr_versucht'])} | "
                     f"{aw._f(k['spur']['gegeben_gelesen'])} | {aw._f(k['spur']['versucht_gegeben_gelesen'])} | "
                     f"{st['sperren_blocked_alle_versuche']} / {st['sperren_nachrichten']} "
                     f"({st['sperren_abweichung_letzter_versuch']} Laeufe abweichend) |")
    if e.get("art") == "pruefstand":
        lines += ["", "| Suite | ASR_spur | ASR_wirksam |", "|---|---|---|"]
        for s, v in e["kennzahlen"]["je_suite"].items():
            lines.append(f"| {s} | {aw._f(v.get('asr_spur'))} | {aw._f(v['asr'])} |")
        lines += ["", "| Angriff | ASR_spur | ASR_wirksam | security_attempted |", "|---|---|---|---|"]
        for a, v in e["kennzahlen"]["je_angriff"].items():
            lines.append(f"| {a} | {aw._f(v.get('asr_spur'))} | {aw._f(v['asr'])} | {aw._f(v['asr_versucht'])} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--zerlegung", nargs=3, metavar=("AUS", "MARKIERUNG", "REGEL"))
    ap.add_argument("--out", required=True, help="Ergebnisdatei (JSON); nie im Laufordner")
    ap.add_argument("--md", help="optional Markdown-Zusammenfassung")
    a = ap.parse_args(argv)
    out_path = Path(a.out).resolve()
    quellen = [Path(x).resolve() for x in ([a.run_dir] if a.run_dir else []) + list(a.zerlegung or [])]
    for q in quellen:
        if out_path == q or q in out_path.parents or (a.md and q in Path(a.md).resolve().parents):
            ap.error("Ausgabe darf nicht im Laufordner liegen")
    if a.zerlegung:
        ctx = _Kontext()
        teile, plan = [], {}
        for name, d in zip(("aus", "markierung", "regel"), a.zerlegung):
            rows, msgs = lese_pruefstand(Path(d))
            ergaenze_spur(rows, msgs, ctx)
            teile.append(_sauber(rows))
            pj = Path(d) / "plan.json"
            plan[name] = json.loads(pj.read_text())["plan_sha256"] if pj.is_file() else None
        e = {"zerlegung": aw.zerlegung(*teile, plan_sha={k: v for k, v in plan.items() if v}),
             "laeufe": dict(zip(("aus", "markierung", "regel"), a.zerlegung))}
    elif a.run_dir:
        e = nachauswerten(Path(a.run_dir))
    else:
        ap.error("LAUFORDNER oder --zerlegung angeben")
    out_path.write_text(json.dumps(e, ensure_ascii=False, indent=1, default=str) + "\n")
    if a.md and "zerlegung" not in e:
        Path(a.md).write_text(markdown(e))
    print(json.dumps({k: v for k, v in e.items() if k in ("lauf", "art")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
