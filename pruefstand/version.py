"""Injektions-Pruefstand: Versionserfassung.

Eine *Version* = Modell (Basis oder LoRA-Adapter) + Gateway-Stand (Hermes-
Commit + Schalter, oder aus) + optional Critic-Version. Jede Version wird
reproduzierbar festgehalten: Adapter-Dateien per sha256, Hermes-Commit voll
aufgeloest + sha256 der exportierten Gateway-Dateien (``git show``, nie die
Arbeitskopie), Critic-Datei + Optionen, dazu vLLM-Image, Generation-Config,
Basismodell-Revision und der Pruefstand-Code (git HEAD + dirty).

``version_id`` = sha256 ueber den kanonischen JSON-Kern (Modell-Hashes,
Gateway-Commit/Schalter/Dateien, Critic) -> gleiche Version = gleiche ID.

Spezifikation (JSON, ``pruefstand/versionen/*.json``)::

    {"name": "p4b+r4",
     "modell": "basis" | "data/lora_adapter_p4b_thinking_b",   # relativ zu PRUEFSTAND_HARNESS
     "gateway": "r4" | "streng" | "aus" | {"commit": "d8c3aafb22", "schalter": {...}},
     "critic": null | {"version": "v1"|"v2", "optionen": {...}},
     # optional (Pruefstand Teil B):
     "herkunft": null | {"modus": "ergaenzung", "lese_egress_ausnahme": true, "zahl_min_stellen": 6,
                         "inhalt_pruefen": true},          # S2 als Adapter-Nachbildung, geht in version_id
     "mensch": null | {"p": 0.0, "politik": "streng"|"wohlwollend", "modell": "oracle"|"karte",
                       "p_nein": 0.0}}                    # Messbedingung, NICHT in version_id

Gateway-Kurzform ``s2s5`` = Hermes e7d8befaaa mit S2 Herkunftsregel + S5-Lockerungen (Hermes-Code).
Gateway-Kurzform ``markierung`` = nur taint_tracking (Huelle, keine Regel; Zerlegung
Modell/Markierung/Regel).

Aufruf: venv/bin/python -m pruefstand.version SPEC.json [--out version.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
# Adapter paths in version specs are relative to PRUEFSTAND_HARNESS (default: <repo>/adapters).
HARNESS = Path(os.environ.get("PRUEFSTAND_HARNESS", str(ROOT / "adapters")))
MODELS = Path(os.environ.get("PRUEFSTAND_MODELS", str(Path.home() / ".cache" / "huggingface")))
GEN_CONFIG = Path(os.environ.get("PRUEFSTAND_GEN_CONFIG",
                                 str(ROOT / "config" / "generation_config.json")))
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
VLLM_IMAGE = os.environ.get("PRUEFSTAND_IMAGE", "vllm/vllm-openai:v0.27.1")
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")
SERVED_BASE, SERVED_LORA = "coding", "hermes"

# Gateway-Kurzformen (Hermes-Branch feat/paket3-gateway)
SCHALTER_3B = {"taint_tracking": True, "session_rule": True, "fail_closed": True}
GATEWAY_PRESETS: Dict[str, Optional[Dict[str, Any]]] = {
    "aus": None,
    "streng": {"commit": "c8341ea748", "schalter": dict(SCHALTER_3B)},
    "r4": {"commit": "d8c3aafb22", "schalter": {**SCHALTER_3B, "verfeinerung": "r4"}},
    # nur Markierung (Huelle der Taint-Quellen), keine Regel: wie gate.CONDITIONS["nur_markierung"]
    "markierung": {"commit": "d8c3aafb22",
                   "schalter": {"taint_tracking": True, "session_rule": False, "fail_closed": False}},
    # Paket 3 S2/S5, Designentscheidungen E2-E7/E11:
    # Hermes' eigene Herkunftsregel + Lockerungen (agent/session_gateway_herkunft.py), keine Nachbildung.
    # critic_hinweis aus: nur Hinweiszeile auf der Karte, ohne Wirkung auf Entscheidungen (Oracle-Mensch).
    # delegate_kein_ausloeser aus: wirkt nur mit child_agents (3f), AgentDojo hat kein delegate_task.
    # command_allowlist_beachten (E4) an, aber ohne terminal/execute_code in AgentDojo wirkungslos.
    "s2s5": {"commit": "e7d8befaaa",
             "schalter": {**SCHALTER_3B, "verfeinerung": "r4", "herkunftsregel": True, "aufgabenliste": True,
                          "buendelung": True, "suchkanaele_frei": True, "command_allowlist_beachten": True,
                          "steer_marker_entschaerfen": True, "critic_hinweis": False,
                          "delegate_kein_ausloeser": False}},
}
MENSCH_MODELLE = ("oracle", "karte")
MENSCH_POLITIKEN = ("streng", "wohlwollend")
HERKUNFT_MODUL = ROOT / "p3g_adapter" / "herkunft.py"
CRITIC_MIN_COMMIT = {"v1": "2a373b94a3", "v2": "7df656f4a5"}
# Critic im Pruefstand: gleicher Testcontainer, Basismodell als Critic (wie Produktion 'coding').
CRITIC_DEFAULTS = {"base_url": "http://localhost:8012/v1", "model": SERVED_BASE}


class VersionFehler(ValueError):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _run(cmd, cwd=None) -> str:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


# ---------------------------------------------------------------------------
# Normalisierung der Spezifikation
# ---------------------------------------------------------------------------

def normalize_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise VersionFehler("Spezifikation muss ein Objekt sein")
    modell = spec.get("modell", "basis")
    if not isinstance(modell, str) or not modell:
        raise VersionFehler("modell fehlt")
    gw = spec.get("gateway", "aus")
    if isinstance(gw, str):
        if gw not in GATEWAY_PRESETS:
            raise VersionFehler(f"unbekannter Gateway-Stand {gw!r} (erlaubt: {sorted(GATEWAY_PRESETS)} oder Objekt)")
        preset = GATEWAY_PRESETS[gw]
        gw_norm = None if preset is None else {"commit": preset["commit"], "schalter": dict(preset["schalter"]),
                                                "kurz": gw}
    elif isinstance(gw, dict):
        if not gw.get("commit") or not isinstance(gw.get("schalter"), dict):
            raise VersionFehler("gateway braucht commit und schalter")
        gw_norm = {"commit": str(gw["commit"]), "schalter": dict(gw["schalter"]), "kurz": gw.get("kurz", "eigen")}
    else:
        raise VersionFehler("gateway: Kurzform oder Objekt")
    critic = spec.get("critic")
    if critic is not None:
        if not isinstance(critic, dict) or critic.get("version") not in CRITIC_MIN_COMMIT:
            raise VersionFehler("critic: {'version': 'v1'|'v2', 'optionen': {...}}")
        if gw_norm is None:
            raise VersionFehler("critic braucht einen Gateway-Stand (Critic-Code liegt im Hermes-Commit)")
        opts = {**CRITIC_DEFAULTS, **dict(critic.get("optionen") or {})}
        from urllib.parse import urlparse

        if urlparse(opts["base_url"]).hostname not in ("localhost", "127.0.0.1", "::1"):
            raise VersionFehler("critic base_url nur localhost")
        critic = {"version": critic["version"], "optionen": opts}
    herkunft = spec.get("herkunft")
    if herkunft is not None:
        from p3g_adapter import herkunft as hk

        if gw_norm is None or not (gw_norm["schalter"].get("taint_tracking") and gw_norm["schalter"].get("session_rule")):
            raise VersionFehler("herkunft braucht ein Gateway mit taint_tracking + session_rule (Modus ergaenzung)")
        if not isinstance(herkunft, dict):
            raise VersionFehler("herkunft: Objekt")
        unbekannt = set(herkunft) - set(hk.DEFAULTS)
        if unbekannt:
            raise VersionFehler(f"herkunft: unbekannte Felder {sorted(unbekannt)}")
        try:
            herkunft = hk.config(herkunft)
        except ValueError as e:
            raise VersionFehler(str(e))
        if herkunft["modus"] != "ergaenzung":
            raise VersionFehler("herkunft.modus: im Pruefstand nur 'ergaenzung' (Designentscheidung E2)")
    mensch = spec.get("mensch")
    if mensch is not None:
        if gw_norm is None:
            raise VersionFehler("mensch braucht ein Gateway (ohne Gateway gibt es keine Freigaben)")
        if not isinstance(mensch, dict):
            raise VersionFehler("mensch: Objekt")
        unbekannt = set(mensch) - {"p", "p_ja", "p_nein", "politik", "modell"}
        if unbekannt:
            raise VersionFehler(f"mensch: unbekannte Felder {sorted(unbekannt)}")
        p_ja = float(mensch.get("p_ja", mensch.get("p", 0.0)))
        p_nein = float(mensch.get("p_nein", 0.0))
        m = {"p": p_ja, "p_ja": p_ja, "p_nein": p_nein, "politik": mensch.get("politik", "streng"),
             "modell": mensch.get("modell", "oracle")}
        if not (0.0 <= p_ja <= 1.0 and 0.0 <= p_nein <= 1.0):
            raise VersionFehler("mensch.p in [0,1]")
        if m["politik"] not in MENSCH_POLITIKEN or m["modell"] not in MENSCH_MODELLE:
            raise VersionFehler(f"mensch: politik {MENSCH_POLITIKEN}, modell {MENSCH_MODELLE}")
        mensch = m
    if herkunft is not None:
        gw_norm["herkunft"] = herkunft
    name = spec.get("name") or f"{'basis' if modell == 'basis' else Path(modell).name}+{(gw_norm or {}).get('kurz', 'aus')}"
    out = {"name": str(name), "modell": modell, "gateway": gw_norm, "critic": critic}
    if mensch is not None:
        out["mensch"] = mensch
    return out


def gateway_switches(norm: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Schalter-Block fuer den Bruecken-Config-Stub (inkl. Critic)."""
    gw = norm.get("gateway")
    if gw is None:
        return None
    sw = dict(gw["schalter"])
    if norm.get("critic"):
        sw["alignment_critic"] = True
        sw["alignment_critic_version"] = norm["critic"]["version"]
        sw["critic"] = dict(norm["critic"]["optionen"])
    return sw


# ---------------------------------------------------------------------------
# Erfassung
# ---------------------------------------------------------------------------

def erfasse_modell(modell: str, harness: Path = HARNESS, models: Path = MODELS) -> Dict[str, Any]:
    rev_file = models / "hub" / f"models--{BASE_MODEL.replace('/', '--')}" / "refs" / "main"
    out: Dict[str, Any] = {
        "basis": BASE_MODEL,
        "basis_revision": rev_file.read_text().strip() if rev_file.is_file() else None,
    }
    if modell == "basis":
        out.update(art="basis", served_name=SERVED_BASE, adapter=None)
        return out
    rel = modell[len(str(harness)) + 1:] if modell.startswith(str(harness) + "/") else modell
    if rel.startswith("/") or ".." in Path(rel).parts:
        raise VersionFehler(f"Adapter {modell} muss im Repo {harness} liegen")
    d = harness / rel
    missing = [f for f in ADAPTER_FILES if not (d / f).is_file()]
    if missing:
        raise VersionFehler(f"Adapter {d}: fehlt {missing}")
    cfg = json.loads((d / "adapter_config.json").read_text())
    out.update(
        art="lora", served_name=SERVED_LORA,
        adapter={"pfad": rel, "sha256": {f: sha256_file(d / f) for f in ADAPTER_FILES},
                 "r": cfg.get("r"), "lora_alpha": cfg.get("lora_alpha"),
                 "base_model_name_or_path": cfg.get("base_model_name_or_path")},
    )
    if out["adapter"]["r"] is not None and int(out["adapter"]["r"]) > 16:
        raise VersionFehler("Adapter-Rang > 16 (Container: --max-lora-rank 16)")
    return out


def erfasse_gateway(norm: Dict[str, Any], snapshot: Optional[Callable[[str], Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    gw = norm.get("gateway")
    if gw is None:
        return None
    if snapshot is None:
        from p3g_adapter import hermes_bridge

        snapshot = hermes_bridge.ensure_snapshot
    prov = snapshot(gw["commit"])
    files = prov["files_sha256"]
    out = {"commit": prov["commit"], "commit_kurz": gw["commit"], "kurz": gw.get("kurz"),
           "schalter": gateway_switches(norm), "dateien_sha256": files, "worktree": prov.get("worktree")}
    if gw.get("herkunft") is not None:
        out["herkunft"] = {**gw["herkunft"], "umsetzung": "adapter-rechenmodell",
                           "modul_sha256": sha256_file(HERKUNFT_MODUL)}
    if norm.get("critic"):
        crit = "agent/session_gateway_critic.py"
        if crit not in files:
            raise VersionFehler(f"Commit {gw['commit']} enthaelt keinen Critic")
        out["critic"] = {"version": norm["critic"]["version"], "datei_sha256": files[crit],
                         "optionen": norm["critic"]["optionen"]}
    return out


def erfasse_umgebung(docker: Callable[..., str] = _run) -> Dict[str, Any]:
    out: Dict[str, Any] = {"vllm_image": VLLM_IMAGE}
    try:
        out["vllm_image_id"] = docker(["docker", "image", "inspect", VLLM_IMAGE, "--format", "{{.Id}}"])
    except Exception:
        out["vllm_image_id"] = None
    out["generation_config_sha256"] = sha256_file(GEN_CONFIG) if GEN_CONFIG.is_file() else None
    try:
        head = _run(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
        dirty = bool(_run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no",
                           "--", "p3g_adapter", "pruefstand"]))
    except Exception:
        head, dirty = None, None
    try:
        from importlib.metadata import version as _v

        ad = _v("agentdojo")
    except Exception:
        ad = None
    out["pruefstand"] = {"git_head": head, "dirty": dirty, "agentdojo": ad, "benchmark": "v1.2.2"}
    return out


def kern(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Was die Version ausmacht (Basis fuer version_id)."""
    m = rec["modell"]
    gw = rec["gateway"]
    return {
        "modell": {"basis": m["basis"], "basis_revision": m.get("basis_revision"),
                   "adapter_sha256": (m.get("adapter") or {}).get("sha256")},
        "gateway": None if gw is None else {"commit": gw["commit"], "schalter": gw["schalter"],
                                            "dateien_sha256": gw["dateien_sha256"],
                                            # nur wenn vorhanden -> bisherige version_id unveraendert
                                            **({"herkunft": gw["herkunft"]} if gw.get("herkunft") else {})},
    }


def erfasse(spec: Dict[str, Any], *, harness: Path = HARNESS, models: Path = MODELS,
            snapshot=None, docker=_run) -> Dict[str, Any]:
    norm = normalize_spec(spec)
    rec = {
        "name": norm["name"],
        "spec": norm,
        "modell": erfasse_modell(norm["modell"], harness, models),
        "gateway": erfasse_gateway(norm, snapshot),
        "critic": None,
        "umgebung": erfasse_umgebung(docker),
    }
    if norm.get("mensch") is not None:
        rec["mensch"] = norm["mensch"]  # Messbedingung, nicht Teil der version_id
    if rec["gateway"] and rec["gateway"].get("critic"):
        rec["critic"] = rec["gateway"]["critic"]
    rec["version_id"] = hashlib.sha256(canonical(kern(rec)).encode()).hexdigest()[:16]
    return rec


# --- E12 (07.10.2026): optionale Chat-Vorlage je Version ------------------------------------------------------
# Spezifikationsfeld ``"chat_template": "vorlagen/e12/chat_template.jinja"`` (relativ zu PRUEFSTAND_HARNESS): run.sh
# reicht es als PRUEFSTAND_CHAT_TEMPLATE an start_vllm.sh weiter (vLLM --chat-template, serverweit). Die Vorlage
# (sha256) geht in die version_id ein; ohne das Feld bleiben Spezifikation, Datensatz und version_id unveraendert.
# Als Huelle um normalize_spec/kern/erfasse/kurzbeschreibung angelegt (eigener Block, keine Aenderung der Funktionen).
_normalize_spec_ohne_vorlage = normalize_spec
_kern_ohne_vorlage = kern
_erfasse_ohne_vorlage = erfasse


def normalize_spec(spec: Dict[str, Any]) -> Dict[str, Any]:  # noqa: F811
    out = _normalize_spec_ohne_vorlage(spec)
    vorlage = spec.get("chat_template") if isinstance(spec, dict) else None
    if vorlage is not None:
        if not isinstance(vorlage, str) or not vorlage.endswith(".jinja") or vorlage.startswith("/") \
                or ".." in Path(vorlage).parts:
            raise VersionFehler("chat_template: relativer Pfad einer .jinja-Datei im Harness-Repo")
        out["chat_template"] = vorlage
    return out


def kern(rec: Dict[str, Any]) -> Dict[str, Any]:  # noqa: F811
    out = _kern_ohne_vorlage(rec)
    if rec.get("chat_template"):  # nur wenn gesetzt -> bisherige version_ids unveraendert
        out["chat_template_sha256"] = rec["chat_template"]["sha256"]
    return out


def erfasse(spec: Dict[str, Any], *, harness: Path = HARNESS, **kw) -> Dict[str, Any]:  # noqa: F811
    rec = _erfasse_ohne_vorlage(spec, harness=harness, **kw)
    vorlage = rec["spec"].get("chat_template")
    if vorlage:
        f = harness / vorlage
        if not f.is_file():
            raise VersionFehler(f"Chat-Vorlage {f} fehlt")
        rec["chat_template"] = {"pfad": vorlage, "sha256": sha256_file(f)}
        rec["version_id"] = hashlib.sha256(canonical(kern(rec)).encode()).hexdigest()[:16]
    return rec


def kurzbeschreibung(rec: Dict[str, Any]) -> str:
    m = rec["modell"]
    mod = "Basis" if m["art"] == "basis" else f"{Path(m['adapter']['pfad']).name} ({m['adapter']['sha256']['adapter_model.safetensors'][:12]})"
    gw = rec["gateway"]
    g = "Gateway aus" if gw is None else f"Gateway {gw.get('kurz')} @{gw['commit'][:10]}"
    c = "" if not rec.get("critic") else f", Critic {rec['critic']['version']}"
    h = "" if not (gw or {}).get("herkunft") else f", S2 Herkunft {gw['herkunft']['modus']} (Adapter)"
    m = rec.get("mensch")
    mm = "" if not m else f", Mensch {m['modell']}/{m['politik']} p={m['p']:g}"
    return f"{mod}, {g}{c}{h}{mm}"


# E12: Kurzbeschreibung um die Chat-Vorlage ergaenzen (siehe Block oben)
_kurzbeschreibung_ohne_vorlage = kurzbeschreibung


def kurzbeschreibung(rec: Dict[str, Any]) -> str:  # noqa: F811
    v = rec.get("chat_template")
    return _kurzbeschreibung_ohne_vorlage(rec) + ("" if not v else f", Vorlage {v['pfad']} ({v['sha256'][:12]})")


# --- Quarantaene-Leser V1 (Entwurf: docs/findings.md, Abschnitt 3) ------------------------------------------------
# Spezifikationsfeld ``"quarantaene": {"variante": "v1", "modell": "coding", "max_zeichen": 600, ...}`` (Felder siehe
# ``p3g_adapter/quarantaene.DEFAULTS``). Jedes Ergebnis einer Taint-Quelle erreicht das Hauptmodell nur als
# Leser-Ausgabe. Konfiguration + sha256 von ``p3g_adapter/quarantaene.py`` gehen in die version_id ein; ohne das Feld
# bleiben Spezifikation, Datensatz und version_id unveraendert. Wieder als Huelle (eigener Block).
QUARANTAENE_MODUL = ROOT / "p3g_adapter" / "quarantaene.py"
_normalize_spec_ohne_q = normalize_spec
_kern_ohne_q = kern
_erfasse_ohne_q = erfasse
_kurzbeschreibung_ohne_q = kurzbeschreibung


def normalize_spec(spec: Dict[str, Any]) -> Dict[str, Any]:  # noqa: F811
    out = _normalize_spec_ohne_q(spec)
    q = spec.get("quarantaene") if isinstance(spec, dict) else None
    if q is not None:
        from p3g_adapter import quarantaene as qm

        if not isinstance(q, dict):
            raise VersionFehler("quarantaene: Objekt")
        try:
            out["quarantaene"] = qm.config(q)
        except ValueError as e:
            raise VersionFehler(str(e))
    return out


def kern(rec: Dict[str, Any]) -> Dict[str, Any]:  # noqa: F811
    out = _kern_ohne_q(rec)
    if rec.get("quarantaene"):  # nur wenn gesetzt -> bisherige version_ids unveraendert
        out["quarantaene"] = rec["quarantaene"]
    return out


def erfasse(spec: Dict[str, Any], **kw) -> Dict[str, Any]:  # noqa: F811
    rec = _erfasse_ohne_q(spec, **kw)
    q = rec["spec"].get("quarantaene")
    if q:
        rec["quarantaene"] = {**q, "modul_sha256": sha256_file(QUARANTAENE_MODUL)}
        rec["version_id"] = hashlib.sha256(canonical(kern(rec)).encode()).hexdigest()[:16]
    return rec


def kurzbeschreibung(rec: Dict[str, Any]) -> str:  # noqa: F811
    q = rec.get("quarantaene")
    return _kurzbeschreibung_ohne_q(rec) + ("" if not q else
                                             f", Quarantaene-Leser {q['variante']}"
                                             f"{'' if q.get('anweisungen_inhalt', True) else '-strikt'}"
                                             f" ({q['modell']}, <= {q['max_zeichen']} Z.)")


# --- Bruecke v2 (14.10.2026): Spezifikationsfeld ``"bruecke": {"url_ziele": true}`` -------------------------------
# URL-Lesetools der AgentDojo-Bruecke (get_webpage) bekommen ihr url-Argument als echtes Hermes-Ziel statt des
# Werkzeug-Kanals (p3g_adapter.gate.url_ziele_aktivieren). Nur wenn gesetzt -> bisherige version_ids unveraendert.
BRUECKE_SCHLUESSEL = ("url_ziele",)
_normalize_spec_ohne_b = normalize_spec
_kern_ohne_b = kern
_erfasse_ohne_b = erfasse
_kurzbeschreibung_ohne_b = kurzbeschreibung


def normalize_spec(spec: Dict[str, Any]) -> Dict[str, Any]:  # noqa: F811
    out = _normalize_spec_ohne_b(spec)
    b = spec.get("bruecke") if isinstance(spec, dict) else None
    if b is not None:
        if not isinstance(b, dict) or set(b) - set(BRUECKE_SCHLUESSEL) or not all(isinstance(v, bool) for v in b.values()):
            raise VersionFehler(f"bruecke: Objekt mit bool-Feldern aus {BRUECKE_SCHLUESSEL}")
        if any(b.values()):
            out["bruecke"] = {k: b[k] for k in BRUECKE_SCHLUESSEL if b.get(k)}
    return out


def kern(rec: Dict[str, Any]) -> Dict[str, Any]:  # noqa: F811
    out = _kern_ohne_b(rec)
    if rec.get("bruecke"):
        out["bruecke"] = rec["bruecke"]
    return out


def erfasse(spec: Dict[str, Any], **kw) -> Dict[str, Any]:  # noqa: F811
    rec = _erfasse_ohne_b(spec, **kw)
    b = rec["spec"].get("bruecke")
    if b:
        rec["bruecke"] = dict(b)
        rec["version_id"] = hashlib.sha256(canonical(kern(rec)).encode()).hexdigest()[:16]
    return rec


def kurzbeschreibung(rec: Dict[str, Any]) -> str:  # noqa: F811
    return _kurzbeschreibung_ohne_b(rec) + (", Bruecke url_ziele" if (rec.get("bruecke") or {}).get("url_ziele") else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spec", help="Versions-Spezifikation (JSON-Datei)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    try:
        rec = erfasse(json.loads(Path(a.spec).read_text()))
    except VersionFehler as e:
        print(f"Versionsfehler: {e}", file=sys.stderr)
        return 2
    text = json.dumps(rec, ensure_ascii=False, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
