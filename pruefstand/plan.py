"""Injektions-Pruefstand: Messplan je Profil (deterministisch, fuer alle Versionen gleich).

Nur AgentDojo-0.1.35-Material (Benchmark v1.2.2): Suites slack, banking,
workspace, travel; Angriffe aus der AgentDojo-Registry (keine eigenen Inhalte).
Gezielte Angriffe (targeted ASR): important_instructions, tool_knowledge,
ignore_previous, system_message. Nicht verwendet: *_dos (messen
Dienstverweigerung, kein Angreiferziel), manual (interaktiv), Varianten von
important_instructions mit falschen Namen.

Workspace-Injection-Tasks 6-13 sind ausgeschlossen: sie bestehen schon
AgentDojos eigene Ground-Truth-Pruefung nicht (Bericht paket3g 05.10.,
Abschnitt 1) -> ASR waere dort immer 0 und wuerde die Quote verduennen.

Profile
-------
schnell  1 Wiederholung; je Suite 6 Auftraege ohne Angriff und 10 feste
         (Auftrag, Injection)-Paare, jedes Paar unter jedem der 4 Angriffe
         (gepaartes Design): 4 x (6 + 40) = 184 Agentenlaeufe.
voll     5 Wiederholungen; alle 97 Auftraege ohne Angriff + alle Paare
         (629 ohne Workspace 6-13) unter jedem der 4 Angriffe:
         5 x (97 + 4 x 629) = 13 065 Agentenlaeufe.

Auswahl mit festem Seed; Injection-Tasks im Rundlauf, damit jede
Injection-Task vorkommt. Plan-sha256 wird mit jedem Ergebnis abgelegt.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List, Optional

SUITES = ("slack", "banking", "workspace", "travel")
ATTACKS = ("important_instructions", "tool_knowledge", "ignore_previous", "system_message")
BENCHMARK_VERSION = "v1.2.2"
EXCLUDED_INJECTIONS = {"workspace": {f"injection_task_{i}" for i in range(6, 14)}}
SEED = 20261006

PROFILES: Dict[str, Dict[str, Any]] = {
    "schnell": {"reps": 1, "benign_per_suite": 6, "pairs_per_suite": 10, "frist_min": 150},
    "voll": {"reps": 5, "benign_per_suite": None, "pairs_per_suite": None, "frist_min": 96 * 60},
}
# Gemessen: Slack-Laeufe 05./06.10. (Basis, 8 parallel): Mittel 130 s je Agentenlauf -> 16,5 s Wandzeit je Lauf.
SEKUNDEN_JE_LAUF = 16.5
CONTAINER_START_MIN = 6


def _numkey(task_id: str) -> int:
    return int(task_id.rsplit("_", 1)[-1])


def suite_tasks(suite_name: str, loader=None):
    """(user_ids, injection_ids) sortiert; Workspace 6-13 entfernt."""
    if loader is None:
        from agentdojo.task_suite.load_suites import get_suite

        s = get_suite(BENCHMARK_VERSION, suite_name)
        users, injs = list(s.user_tasks), list(s.injection_tasks)
    else:
        users, injs = loader(suite_name)
    users = sorted(users, key=_numkey)
    injs = sorted((i for i in injs if i not in EXCLUDED_INJECTIONS.get(suite_name, set())), key=_numkey)
    return users, injs


def build_plan(profile: str, suites=SUITES, attacks=ATTACKS, seed: int = SEED, loader=None,
               reps: Optional[int] = None) -> Dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"Profil {profile!r} unbekannt ({sorted(PROFILES)})")
    prof = PROFILES[profile]
    reps = prof["reps"] if reps is None else reps
    items: List[Dict[str, Any]] = []
    auswahl: Dict[str, Any] = {}
    for suite in suites:
        users, injs = suite_tasks(suite, loader)
        rng = random.Random(f"{seed}:{suite}")
        nb = prof["benign_per_suite"]
        benign = users if nb is None else sorted(rng.sample(users, min(nb, len(users))), key=_numkey)
        npairs = prof["pairs_per_suite"]
        if npairs is None:
            pairs = [(u, i) for u in users for i in injs]
        else:
            order = rng.sample(users, len(users))
            pairs, k = [], 0
            while len(pairs) < min(npairs, len(users) * len(injs)):
                u, i = order[k % len(order)], injs[k % len(injs)]
                if (u, i) not in pairs:
                    pairs.append((u, i))
                k += 1
            pairs.sort(key=lambda p: (_numkey(p[0]), _numkey(p[1])))
        auswahl[suite] = {"ohne_angriff": benign, "paare": [list(p) for p in pairs]}
        for rep in range(1, reps + 1):
            for u in benign:
                items.append({"suite": suite, "user_task": u, "injection_task": None, "attack": None, "rep": rep})
            for attack in attacks:
                for u, i in pairs:
                    items.append({"suite": suite, "user_task": u, "injection_task": i, "attack": attack, "rep": rep})
    for it in items:
        it["key"] = item_key(it)
    plan = {
        "profil": profile, "benchmark": BENCHMARK_VERSION, "suites": list(suites), "angriffe": list(attacks),
        "seed": seed, "reps": reps, "ausgeschlossen": {k: sorted(v, key=_numkey) for k, v in EXCLUDED_INJECTIONS.items()},
        "auswahl": auswahl, "items": items,
    }
    plan["plan_sha256"] = plan_sha256(plan)
    return plan


def item_key(it: Dict[str, Any]) -> str:
    return f"{it['suite']}|{it['user_task']}|{it['injection_task'] or '-'}|{it['attack'] or '-'}|{it['rep']}"


def plan_sha256(plan: Dict[str, Any]) -> str:
    core = {k: plan[k] for k in ("profil", "benchmark", "suites", "angriffe", "seed", "reps", "items")}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def estimate(plan: Dict[str, Any], n_versions: int = 1, sekunden_je_lauf: float = SEKUNDEN_JE_LAUF) -> Dict[str, Any]:
    n = len(plan["items"])
    lauf_min = n * sekunden_je_lauf / 60
    return {"agentenlaeufe": n, "minuten_je_version": round(lauf_min + CONTAINER_START_MIN),
            "stunden_gesamt": round(n_versions * (lauf_min + CONTAINER_START_MIN) / 60, 1),
            "annahme": f"{sekunden_je_lauf} s Wandzeit je Agentenlauf (8 parallel, Slack 05./06.10.), "
                       f"+{CONTAINER_START_MIN} min Containerstart; Workspace/Travel laenger -> eher +30 %"}


def counts(plan: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for it in plan["items"]:
        c = out.setdefault(it["suite"], {"ohne_angriff": 0, "unter_angriff": 0})
        c["unter_angriff" if it["injection_task"] else "ohne_angriff"] += 1
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("profil", choices=sorted(PROFILES))
    ap.add_argument("--out")
    ap.add_argument("--versionen", type=int, default=1)
    a = ap.parse_args()
    p = build_plan(a.profil)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(p, f, ensure_ascii=False, indent=1)
    print(json.dumps({"profil": a.profil, "plan_sha256": p["plan_sha256"], "je_suite": counts(p),
                      "schaetzung": estimate(p, a.versionen)}, ensure_ascii=False, indent=1))
