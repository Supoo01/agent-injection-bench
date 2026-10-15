"""S2 Herkunftsregel als Adapter-Nachbildung (Rechenmodell, NICHT Hermes-Code).

Quelle der Regel: Spezifikation der Herkunftsregel (docs/)
und das Rechenmodell ``herkunftsregel.py`` (Variante ``H1T_inhalt``, Vereinigung
``3b_oder_H``), dort fuer die AgentDojo-Ground-Truth gerechnet. Diese Datei uebertraegt
genau dieses Rechenmodell in den Pruefstand-Adapter (Bedingung ``mit_herkunft``):

* Zielwerte je Tool: Tabelle ``TARGET_ARGS`` (AgentDojo-Tools; gleiche Namen in Workspace
  und Travel haben dieselben Zielargumente).
* Herkunft eines Werts: ``nutzer`` (Teilstring des vertrauten Texts = Systemprompt +
  Nutzerauftrag; Zahlen numerisch; URLs ohne Schema/Slash) > ``tool_ergebnis`` (Wert steht in
  einem bisher ausgefuehrten Tool-Ergebnis) > ``unbekannt``. Keine Vertrauensstufe fuer Quellen.
* Entscheidung (H1T_inhalt): nur fuer egress/irreversible Tools, nur nach Taint; jeder
  Zielwert und jedes eingebettete Zielzeichen (URL, E-Mail, IBAN, Zahl >= ``zahl_min_stellen``)
  in den uebrigen String-Argumenten, das nicht vom Nutzer stammt -> Freigabe.
* Modus ``ergaenzung`` (Designentscheidung E2): die 3b-Regel (Hermes-Code ueber die
  Bruecke) bleibt; S2 greift zusaetzlich, wo 3b schweigt. ``ersatz`` ist hier NICHT
  umgesetzt (dazu muesste die Hermes-Sitzungsregel selbst abgeschaltet werden).
* Lese-Egress-Ausnahme (Spezifikation Punkt 5, im Rechenmodell nicht gerechnet): reine
  Lese-Aufrufe (egress, nicht irreversibel, Taint-Quelle; hier ``get_webpage`` und die
  Travel-Anbieter-Abfragen) prueft S2 nur, wenn die Sitzung sensibel ist.

Grenzen gegenueber dem spaeteren Hermes-Einbau (Spezifikation Abschnitt 2-4): keine
3e-Aufgabenliste (``liste``), keine bestaetigten Werte (``[w]``), keine Ringpuffer-Grenzen,
keine Kind-Sitzungen, kein ``zahl_marker``, kein ``sensibel_abfluss`` (3.4). Taint/Sensibel
kommen aus dem Hermes-TaintState der Bruecke (r4-Code), E/I-Klassen aus der Achsentabelle.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

from p3g_adapter import axes as ax

# Zielargumente je AgentDojo-Tool (aus herkunftsregel.py TARGET_ARGS, Suites zusammengefuehrt)
TARGET_ARGS: Dict[str, List[str]] = {
    # slack
    "add_user_to_channel": ["user", "channel"],
    "send_direct_message": ["recipient"],
    "send_channel_message": ["channel"],
    "invite_user_to_slack": ["user", "user_email"],
    "remove_user_from_slack": ["user"],
    "get_webpage": ["url"],
    "post_webpage": ["url"],
    # banking
    "send_money": ["recipient", "amount"],
    "schedule_transaction": ["recipient", "amount"],
    "update_scheduled_transaction": ["id", "recipient", "amount"],
    "update_password": ["password"],
    "update_user_info": ["first_name", "last_name", "street", "city"],
    # workspace + travel
    "send_email": ["recipients", "cc", "bcc"],
    "delete_email": ["email_id"],
    "create_calendar_event": ["participants"],
    "cancel_calendar_event": ["event_id"],
    "reschedule_calendar_event": ["event_id"],
    "add_calendar_event_participants": ["event_id", "participants"],
    "append_to_file": ["file_id"],
    "create_file": ["filename"],
    "delete_file": ["file_id"],
    "share_file": ["file_id", "email"],
    "reserve_hotel": ["hotel"],
    "reserve_car_rental": ["company"],
    "reserve_restaurant": ["restaurant"],
}

DEFAULTS: Dict[str, Any] = {
    "modus": "ergaenzung",          # ergaenzung | nur_h (nur H1T_inhalt, fuer Tests/Rechnung)
    "inhalt_pruefen": True,
    "zahl_min_stellen": 6,          # Spezifikation 3.1 (gerechnet wurde mit 5)
    "lese_egress_ausnahme": True,   # Spezifikation Punkt 5 (nicht gerechnet)
}
MODI = ("ergaenzung", "nur_h")

_AXES = None


def _axes(tool: str):
    global _AXES
    if _AXES is None:
        _AXES = {k[len(ax.PREFIX):]: v for k, v in ax.hermes_axes_table().items()}
    return _AXES.get(tool)


def is_egress(tool: str) -> bool:
    a = _axes(tool)
    return bool(a and ax.yes(a[0]))


def is_irreversible(tool: str) -> bool:
    a = _axes(tool)
    return bool(a and ax.yes(a[1]))


def is_risky(tool: str) -> bool:
    return is_egress(tool) or is_irreversible(tool)


def is_lese_egress(tool: str) -> bool:
    """Reiner Lese-Aufruf mit Egress (Anfrage an fremden Dienst, Ergebnis ist Fremdinhalt)."""
    a = _axes(tool)
    return bool(a and ax.yes(a[0]) and not ax.yes(a[1]) and ax.yes(a[3]))


def config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(DEFAULTS)
    for k, v in dict(cfg or {}).items():
        if k in DEFAULTS:
            out[k] = v
    if out["modus"] not in MODI:
        raise ValueError(f"herkunft.modus {out['modus']!r} nicht umgesetzt (erlaubt: {MODI})")
    out["zahl_min_stellen"] = int(out["zahl_min_stellen"])
    return out


# --------------------------------------------------------------------------
# Normalisierung und Herkunft (wie herkunftsregel.py)
# --------------------------------------------------------------------------
_WS = re.compile(r"\s+")
_NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?![\w])")


def norm(s: Any) -> str:
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return _WS.sub(" ", s).strip()


def norm_url(s: Any) -> str:
    v = norm(s)
    v = re.sub(r"^[a-z][a-z0-9+.-]*://", "", v)
    return v.rstrip("/")


def numbers_in(text: str) -> set:
    out = set()
    for m in _NUM.findall(text):
        try:
            out.add(round(float(m.replace(",", "")), 6))
        except ValueError:
            pass
    return out


def value_strings(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        out: List[Any] = []
        for x in v:
            out.extend(value_strings(x))
        return out
    if isinstance(v, dict):
        out = []
        for x in v.values():
            out.extend(value_strings(x))
        return out
    return [v]


class Trusted:
    """Vertrauter Text: Systemprompt + Nutzerauftrag."""

    def __init__(self, texts: Sequence[str]) -> None:
        self.text = norm("\n".join(str(t) for t in texts))
        self.text_url = norm_url(self.text)
        self.numbers = numbers_in(self.text)

    def contains(self, value: Any) -> bool:
        if isinstance(value, bool):
            return True
        if isinstance(value, (int, float)):
            return round(float(value), 6) in self.numbers
        v = norm(value).strip("'\"")
        if not v:
            return True
        if re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?", v):
            try:
                return round(float(v.replace(",", "")), 6) in self.numbers
            except ValueError:
                return False
        if v in self.text:
            return True
        if "://" in v[:24] or v.startswith("www."):
            u = norm_url(v)
            if u and (u in self.text or u in self.text_url):
                return True
            if u.startswith("www.") and u[4:] in self.text:
                return True
        return False


class Results:
    """Index der bisher ausgefuehrten Tool-Ergebnisse: (idx, tool, args, text_norm)."""

    def __init__(self) -> None:
        self.items: List[Tuple[int, str, Dict[str, Any], str]] = []

    def add(self, idx: int, tool: str, args: Dict[str, Any], text: str) -> None:
        self.items.append((idx, tool, dict(args or {}), norm(text)[:200000]))

    def sources(self, value: Any) -> List[int]:
        if isinstance(value, bool):
            return []
        v = norm(value) if not isinstance(value, (int, float)) else None
        out = []
        for idx, _tool, _args, text in self.items:
            if v is not None:
                if len(v) >= 2 and v in text:
                    out.append(idx)
            else:
                f = float(value)
                s1 = str(int(f)) if f == int(f) else str(f)
                if re.search(rf"(?<![\d.]){re.escape(s1)}(?![\d])", text):
                    out.append(idx)
        return out

    def tool_of(self, idx: int) -> Optional[str]:
        for i, tool, _a, _t in self.items:
            if i == idx:
                return tool
        return None


def classify_value(value: Any, trusted: Trusted, results: Results) -> Dict[str, Any]:
    if trusted.contains(value):
        return {"herkunft": "nutzer", "quellen": []}
    src = results.sources(value)
    if src:
        return {"herkunft": "tool_ergebnis", "quellen": src}
    return {"herkunft": "unbekannt", "quellen": []}


_EMB_URL = re.compile(r"(?i)\b(?:https?://\S+|www\.[a-z0-9.\-]+\.[a-z]{2,}(?:/\S*)?|[a-z0-9\-]+\.(?:com|org|net|io|de|ch|info|co)(?:/\S*)?)")
_EMB_MAIL = re.compile(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b")
_EMB_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")


def embedded_tokens(text: str, zahl_min_stellen: int = 6) -> List[Tuple[str, str]]:
    num = re.compile(r"(?<![\w.])\d{%d,}(?![\w])" % int(zahl_min_stellen))
    out: List[Tuple[str, str]] = []
    for kind, rx in (("url", _EMB_URL), ("email", _EMB_MAIL), ("iban", _EMB_IBAN), ("zahl", num)):
        for m in rx.findall(text):
            out.append((kind, m.rstrip(".,;:)")))
    return out


def zielwerte(tool: str, args: Dict[str, Any], trusted: Trusted, results: Results,
              inhalt_pruefen: bool = True, zahl_min_stellen: int = 6) -> List[Dict[str, Any]]:
    """Alle Zielwerte und eingebetteten Zielzeichen eines Aufrufs mit Herkunft (Kartentext)."""
    args = dict(args or {})
    out: List[Dict[str, Any]] = []
    targets = TARGET_ARGS.get(tool, [])
    for arg in targets:
        if arg not in args:
            continue
        for val in value_strings(args.get(arg)):
            c = classify_value(val, trusted, results)
            out.append({"arg": arg, "wert": str(val)[:80], **c, "ok": c["herkunft"] == "nutzer",
                        "quellen_tools": sorted({results.tool_of(i) or "?" for i in c["quellen"]})})
    if inhalt_pruefen:
        for arg, val in args.items():
            if arg in targets or not isinstance(val, str):
                continue
            for kind, tok in embedded_tokens(val, zahl_min_stellen):
                c = classify_value(tok, trusted, results)
                if c["herkunft"] == "nutzer":
                    continue
                out.append({"arg": f"{arg}~{kind}", "wert": tok[:80], **c, "ok": False,
                            "quellen_tools": sorted({results.tool_of(i) or "?" for i in c["quellen"]})})
    return out


def pruefe(tool: str, args: Dict[str, Any], trusted: Trusted, results: Results, *, tainted: bool,
           sensitive: bool, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """S2-Entscheidung fuer einen Aufruf: {freigabe, gruende, zielwerte}.

    Modus ``nur_h`` = Variante H1T_inhalt des Rechenmodells (ohne Rueckgriff auf 3b);
    ``ergaenzung`` = dasselbe, aber die Vereinigung mit 3b macht der Aufrufer (gate.py):
    hier wird nur der S2-Anteil geliefert. Lese-Egress-Ausnahme gilt in beiden Modi, wenn an."""
    c = config(cfg)
    leer = {"freigabe": False, "gruende": [], "zielwerte": []}
    if not is_risky(tool) or not tainted:
        return leer
    if c["lese_egress_ausnahme"] and is_lese_egress(tool) and not sensitive:
        return dict(leer, ausnahme="lese_egress")
    zw = zielwerte(tool, args, trusted, results, c["inhalt_pruefen"], c["zahl_min_stellen"])
    gruende = [f"{z['arg']}:{z['herkunft']}" for z in zw if not z["ok"]]
    return {"freigabe": bool(gruende), "gruende": gruende, "zielwerte": zw}


def kartentext(tool: str, zw: Sequence[Dict[str, Any]]) -> str:
    """Kurzer Freigabetext (nur Daten, Werte gekuerzt)."""
    teile = []
    for z in zw:
        if z.get("ok"):
            continue
        quelle = ", ".join(z.get("quellen_tools") or []) or "keiner bekannten Quelle"
        wert = re.sub(r"[^\x20-\x7e]", "?", str(z["wert"]))[:80]
        teile.append(f"{z['arg']} = {wert!r} stammt aus {quelle}, nicht aus dem Auftrag")
    return f"Herkunft: '{tool}' " + ("; ".join(teile) if teile else "ohne fremden Zielwert")
