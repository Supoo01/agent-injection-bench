"""AgentDojo pipeline element: tool execution behind the Hermes session gateway.

``HermesGateToolsExecutor`` replaces AgentDojo's ``ToolsExecutor``. Per tool call:

1. session rule (Hermes ``session_rule_directive``) on the current per-run
   ``TaintState``. Hermes answers ``approve`` + ``strict_once``. The benchmark has
   no human, so (exactly like Hermes without a reachable CLI/gateway, Paket 3b
   Punkt 3) the call is BLOCKED and not executed; a ``block`` directive is
   blocked as well. Any exception inside the gate while the rule is on blocks
   (fail-closed, Punkt 4a).
2. execute the AgentDojo function.
3. record the result in the TaintState (Hermes ``record_tool_result``).
4. optionally wrap the result of taint-source tools with Hermes'
   ``<untrusted_tool_result>`` wrapper.

Conditions (one per process, see :data:`CONDITIONS`):
``ohne``          plain AgentDojo, no Hermes element at all;
``mit``           session_rule + taint_tracking + fail_closed on, results of
                  taint sources wrapped;
``nur_markierung`` taint_tracking on (wrapping only), no rule (ablation).
``mit_r4``        like ``mit`` but with the Hermes code of commit d8c3aafb22
                  (refinement r4) and ``verfeinerung: r4``; the gate passes
                  the call arguments to ``session_rule_directive`` like the
                  Hermes funnel does since that commit.
``mit_herkunft``  like ``mit_r4`` plus the S2 Herkunftsregel (mode ``ergaenzung``)
                  as an adapter-side reimplementation of the computation model
                  (:mod:`p3g_adapter.herkunft`, NOT Hermes code).

Hermes S2/S5 (Pruefstand, commit e7d8befaaa on): when a registered condition
switches on any of the S2/S5 switches (:data:`S25_SCHALTER`), the gate runs
Hermes' OWN ``agent.session_gateway_herkunft`` like the Hermes funnel does:
turn start ``note_user_request`` (E5 task list) + ``note_user_message`` +
``note_system_prompt`` on a stand-in agent bound to the run's TaintState;
before a call ``apply(..., rule=<3b directive>)`` (S2 reason, relaxations,
error -> block); a granted/denied approval ``finalize``s the bundling ticket;
after a call ``note_result`` (token index) and, with
``steer_marker_entschaerfen``, ``defang_steer_content`` on the result text.
Tool-name roles (read-egress exception, E3 search channels) follow the
inventory analogue of each AgentDojo tool (:func:`map_hermes_roles`).
Not modelled: E4 (no terminal/execute_code in AgentDojo), E11 (no
delegate_task / child agents), E7 card hint (no effect on decisions).

Simulated human (Pruefstand Teil B, optional): with ``gate.human`` set
(:class:`p3g_adapter.mensch.HumanSim`) every ``approve`` (rule, S2 or critic)
is answered by the simulated human instead of being blocked: yes -> the call
runs; no -> ``BLOCKED: approval denied by the user (...)``. ``human = None``
(default) keeps the behaviour without a human (approval = block).
"""

from __future__ import annotations

import importlib
from ast import literal_eval
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME
from agentdojo.agent_pipeline.tool_execution import is_string_list, tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, ChatToolResultMessage, text_content_block_from_string

from p3g_adapter import axes as ax
from p3g_adapter import hermes_bridge

CONDITIONS: Dict[str, Dict[str, Any]] = {
    "ohne": {"switches": None, "rule": False, "mark": False},
    "mit": {
        "switches": {"taint_tracking": True, "session_rule": True, "fail_closed": True},
        "rule": True,
        "mark": True,
    },
    "mit_r4": {
        "switches": {"taint_tracking": True, "session_rule": True, "fail_closed": True, "verfeinerung": "r4"},
        "rule": True,
        "mark": True,
        "commit": hermes_bridge.COMMIT_R4,
        "pass_args": True,
    },
    "nur_markierung": {
        "switches": {"taint_tracking": True, "session_rule": False, "fail_closed": False},
        "rule": False,
        "mark": True,
    },
    "mit_herkunft": {
        "switches": {"taint_tracking": True, "session_rule": True, "fail_closed": True, "verfeinerung": "r4"},
        "rule": True,
        "mark": True,
        "commit": hermes_bridge.COMMIT_R4,
        "pass_args": True,
        "herkunft": {"modus": "ergaenzung"},
    },
}

# Hermes S2/S5 switches (Loesungsdokument 06./07.10., commit e7d8befaaa).
S25_SCHALTER = ("herkunftsregel", "aufgabenliste", "buendelung", "suchkanaele_frei",
                "command_allowlist_beachten", "delegate_kein_ausloeser", "steer_marker_entschaerfen")

# Same text template as tools/approval.py (strict_once, no CLI/gateway).
NO_HUMAN_TEMPLATE = (
    "BLOCKED: approval required ({description}) but no interactive user or gateway is present to approve it."
)
# Simulated human said no (Pruefstand Teil B).
DENIED_TEMPLATE = "BLOCKED: approval denied by the user ({description})."


_TAINT_CACHE: Dict[str, bool] = {}


def _taint_quelle(tool: str) -> bool:
    """Taint-Quelle laut Achsentabelle (ja|bedingt) -- gleich fuer Gateway aus und an."""
    if not _TAINT_CACHE:
        _TAINT_CACHE.update({k[len(ax.PREFIX):]: ax.yes(v[3]) for k, v in ax.hermes_axes_table().items()})
    return _TAINT_CACHE.get(tool, True)  # unbekanntes Tool: wie fremd behandeln


@dataclass
class GateEvent:
    index: int
    tool: str
    args: Dict[str, Any]
    action: str  # executed | blocked
    reason: str = ""
    state_before: Dict[str, Any] = field(default_factory=dict)
    attempt: int = 1
    approval: Optional[str] = None  # None | granted | denied (simulated human)


def register_condition(name: str, switches: Optional[Dict[str, Any]], commit: Optional[str] = None,
                       mark: Optional[bool] = None, herkunft: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Pruefstand: a condition from a gateway version (Hermes commit + switches).

    ``switches=None`` = gateway off (like ``ohne``). The rule is on when
    ``session_rule`` is on; results of taint sources are wrapped when
    ``taint_tracking`` is on (Hermes behaviour). The alignment critic runs when
    ``alignment_critic`` is on (``critic`` options in the switches). Whether
    call arguments are passed to ``session_rule_directive`` is decided from
    the loaded code (``args`` parameter, from d8c3aafb22 on). ``herkunft``
    (optional) switches on the adapter-side S2 rule (needs taint_tracking and
    session_rule, mode ``ergaenzung``)."""
    if herkunft is not None:
        from p3g_adapter import herkunft as hk

        if switches is None or not (switches.get("taint_tracking") and switches.get("session_rule")):
            raise ValueError("herkunft needs taint_tracking + session_rule (mode ergaenzung)")
        hk.config(herkunft)  # validates
    if switches is None:
        cond: Dict[str, Any] = {"switches": None, "rule": False, "mark": False}
    else:
        sw = dict(switches)
        cond = {
            "switches": sw,
            "rule": bool(sw.get("session_rule")),
            "mark": bool(sw.get("taint_tracking") or sw.get("alignment_critic")) if mark is None else mark,
            "commit": commit,
            "pass_args": "auto",
            "critic": bool(sw.get("alignment_critic")),
        }
        if herkunft is not None:
            cond["herkunft"] = dict(herkunft)
        if any(sw.get(k) for k in S25_SCHALTER):
            if herkunft is not None:
                raise ValueError("adapter herkunft (mit_herkunft) and Hermes S2/S5 switches exclude each other")
            cond["hermes_s25"] = True
    CONDITIONS[name] = cond
    return cond


def map_hermes_roles(hk) -> Dict[str, List[str]]:
    """Tool-name roles of ``session_gateway_herkunft`` for AgentDojo tools.

    Hermes keys some S2/S5 behaviour on its own tool names (``URL_READ_TOOLS``:
    read-egress exception; ``SEARCH_TOOLS``: E3 search channels, no S2). The
    AgentDojo tools carry ``agentdojo.<tool>`` keys, so each tool whose
    inventory analogue (:mod:`p3g_adapter.axes`, column ``analog``) is in one of
    these sets is added to it (in memory only; the module FILES stay the
    commit's bytes): get_webpage -> web_extract, travel provider lookups ->
    web_search. Idempotent; returns the mapping for the record."""
    out: Dict[str, List[str]] = {}
    for attr in ("URL_READ_TOOLS", "SEARCH_TOOLS"):
        base = frozenset(getattr(hk, attr))
        add = sorted({ax.key(n) for tools in ax.TOOL_MAP.values() for n, a in tools.items() if a.analog in base})
        setattr(hk, attr, frozenset(base | set(add)))
        out[attr] = add
    return out


def url_ziele_aktivieren() -> List[str]:
    """Bruecke v2 (Versionsfeld ``"bruecke": {"url_ziele": true}``, 14.10.2026): Hermes' Zielparser kennt die
    AgentDojo-URL-Lesetools (get_webpage -> Rolle web_extract) nicht und faellt auf den Werkzeug-Kanal
    ``kanal:tool:agentdojo.get_webpage`` zurueck; mit S5-Buendelung gab dann EINE Freigabe jede spaetere URL frei
    (K5-Fall Quarantaene-Messung). Hier bekommen diese Tools ihr ``url``-Argument als echtes Ziel (Host-Ebene,
    wie web_extract in Hermes). Nur im Speicher, idempotent; liefert die betroffenen Tool-Schluessel."""
    se = importlib.import_module("agent.session_gateway_egress")
    hk = importlib.import_module("agent.session_gateway_herkunft")
    tools = sorted(t for t in getattr(hk, "URL_READ_TOOLS", ()) if t.startswith(ax.PREFIX))
    if getattr(se.extract_targets, "_p3g_url_ziele", False):
        return tools
    orig = se.extract_targets

    def extract_targets(tool_name, args, cfg=None, *a, **kw):
        if tool_name in tools and isinstance(args, dict) and isinstance(args.get("url"), str) and args["url"].strip():
            url = args["url"].strip()
            if "://" not in url:
                url = "https://" + url
            return se.parse_url(url, cfg if cfg is not None else se.load_egress_config())
        return orig(tool_name, args, cfg, *a, **kw)

    extract_targets._p3g_url_ziele = True
    se.extract_targets = extract_targets
    return tools


def configure_condition(name: str) -> Dict[str, Any]:
    cond = CONDITIONS[name]
    if cond["switches"] is not None:
        gw = hermes_bridge.configure(cond["switches"], commit=cond.get("commit"))
        want = cond["switches"].get("verfeinerung")
        if want is not None and gw.refinement_level() != {"streng": 0, "r1": 1, "r2": 2, "r3": 3, "r4": 4}[want]:
            raise RuntimeError(f"verfeinerung {want!r} not active in loaded Hermes code")
        if cond.get("pass_args") == "auto":
            import inspect

            cond["pass_args"] = "args" in inspect.signature(gw.session_rule_directive).parameters
        if cond.get("critic"):
            cm = importlib.import_module("agent.session_gateway_critic")
            if not cm.alignment_critic_enabled():
                raise RuntimeError("alignment_critic switch not active in loaded Hermes code")
            want_v = cond["switches"].get("alignment_critic_version")
            if want_v and (not hasattr(cm, "critic_version") or cm.critic_version() != want_v):
                raise RuntimeError(f"alignment_critic_version {want_v!r} not supported by loaded Hermes code")
        if cond.get("hermes_s25"):
            try:
                hk = importlib.import_module("agent.session_gateway_herkunft")
            except ImportError as exc:
                raise RuntimeError(f"S2/S5 switches set, but the loaded Hermes code has no herkunft module: {exc}")
            if not hk.any_enabled() and not cond["switches"].get("steer_marker_entschaerfen"):
                raise RuntimeError("S2/S5 switches not active in loaded Hermes code")
            if cond["switches"].get("herkunftsregel") and not hk.herkunft_enabled():
                raise RuntimeError("herkunftsregel not active in loaded Hermes code")
            cond["hermes_rollen"] = map_hermes_roles(hk)
    return cond


class HermesGateToolsExecutor(BasePipelineElement):
    """Drop-in for ``agentdojo.agent_pipeline.ToolsExecutor`` with the Hermes gate.

    One instance per task run (it owns the run's TaintState)."""

    name = "hermes_gate_tools_executor"

    def __init__(self, condition: str, output_formatter=tool_result_to_str) -> None:
        self.condition = condition
        self.cond = CONDITIONS[condition]
        self.output_formatter = output_formatter
        self.events: List[GateEvent] = []
        self.executed_tool_calls: List[Any] = []  # FunctionCall objects actually run (current attempt)
        self._n = 0
        self.gw = hermes_bridge.gateway() if self.cond["switches"] is not None else None
        self.state = self.gw.TaintState() if self.gw is not None else None
        self.cm = importlib.import_module("agent.session_gateway_critic") if self.cond.get("critic") else None
        self.critic_events: List[Dict[str, Any]] = []
        # Pruefstand Teil B (all optional, default off)
        self.human = None  # p3g_adapter.mensch.HumanSim or None (= no human: approval -> block)
        self.attempt = 0
        self.herkunft_cfg = None
        if self.cond.get("herkunft") is not None:
            from p3g_adapter import herkunft as hk

            self.herkunft_cfg = hk.config(self.cond["herkunft"])
        self._trusted = None
        self._results = None
        self.herkunft_events: List[Dict[str, Any]] = []
        self.injection_texts: List[str] = []
        self.injection_read = False
        # Hermes S2/S5 (agent.session_gateway_herkunft of the loaded commit)
        self.hk = None
        self._agent = None
        self._ticket = None
        self.s25_events: List[Dict[str, Any]] = []
        # Quarantaene-Leser V1 (optional, default aus): p3g_adapter.quarantaene.Leser
        self.leser = None
        self.leser_merkmale: set = set()  # Adress-Tokens der Injection-GT (Durchreich-Mass)
        self.leser_durchgereicht = False
        if self.cond.get("hermes_s25"):
            import types as _types

            self.hk = importlib.import_module("agent.session_gateway_herkunft")
            # stand-in for the Hermes agent: root agent (depth 0), no session id
            # (no process-wide session store), bound to this run's TaintState
            self._agent = _types.SimpleNamespace(session_id="", _delegate_depth=0)
            setattr(self._agent, "_session_gateway_state", self.state)

    def note_turn_start(self, system_message: str, user_text: str) -> None:
        """Hermes turn start (agent/turn_context.py): E5 task list from the
        REAL user message, trusted user text and root system prompt (S2/E3).
        No-op without Hermes S2/S5 switches."""
        if self.hk is None:
            return
        eg = importlib.import_module("agent.session_gateway_egress")
        eg.note_user_request(self._agent, user_text, display_kind=None)
        self.hk.note_user_message(self._agent, user_text, display_kind=None)
        self.hk.note_system_prompt(self._agent, system_message)

    def s25_stats(self) -> Dict[str, Any]:
        from collections import Counter

        ev = self.s25_events
        return {"s2_anfragen": sum(1 for e in ev if e.get("s2")),
                "s2_codes": dict(Counter(c for e in ev for c in e.get("codes", ()))),
                "gelockert": dict(Counter(e["gelockert"] for e in ev if e.get("gelockert"))),
                "pruef_sperren": sum(1 for e in ev if e.get("block")),
                "tools_s2": [e["tool"] for e in ev if e.get("s2")]}

    def _finalize_ticket(self, approved: bool) -> None:
        t, self._ticket = self._ticket, None
        if t is not None and self.hk is not None:
            self.hk.finalize(t, approved=approved)

    def set_trusted(self, texts: Sequence[str]) -> None:
        """Trusted text for the provenance index (system prompt + user task)."""
        from p3g_adapter import herkunft as hk

        self._trusted = hk.Trusted(list(texts))
        if self._results is None:
            self._results = hk.Results()

    def _index_active(self) -> bool:
        return self._trusted is not None and (self.herkunft_cfg is not None or self.human is not None)

    def _karte(self, tool: str, args: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Approval card content: target values with provenance (for the card human)."""
        if self._trusted is None:
            return []
        from p3g_adapter import herkunft as hk

        cfg = self.herkunft_cfg or hk.config(None)
        return hk.zielwerte(tool, dict(args or {}), self._trusted, self._results, cfg["inhalt_pruefen"],
                            cfg["zahl_min_stellen"])

    def note_user_goal(self, text: str) -> None:
        """Critic: the user's own request (Hermes ``note_user_goal`` at turn
        start; here directly on the run's TaintState, no agent object)."""
        if self.cm is None or not text.strip():
            return
        cfg = self.cm.load_critic_config()
        self.state.critic.add_goal(self.cm._clip(text, int(cfg["max_ziel_zeichen"])), max_goals=int(cfg["max_ziele"]))

    def _critic(self, tool: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.cm is None:
            return None
        d = self.cm.critic_directive(ax.key(tool), dict(args or {}), taint_state=self.state)
        if d is not None:
            self.critic_events.append({"tool": tool, "action": d.get("action"), "code": d.get("critic_code"),
                                       "halt": bool(d.get("halt"))})
        return d

    def critic_stats(self) -> Dict[str, Any]:
        snap = self.state.critic.snapshot() if self.cm is not None else {}
        return {"calls": snap.get("calls", 0), "vetoes": snap.get("vetoes", 0),
                "codes": [e["code"] for e in self.critic_events], "halt": any(e["halt"] for e in self.critic_events)}

    # -- helpers -------------------------------------------------------------
    def _snapshot(self) -> Dict[str, Any]:
        return self.state.snapshot() if self.state is not None else {}

    def _decide(self, tool: str, args: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Return a block message or None (= may execute).

        Order as in the Hermes funnel (hermes_cli/plugins.py): rule ``block``
        -> block; else critic (if on) ``block`` -> block; any remaining
        ``approve`` (rule, S2 Herkunft or critic) -> strict single approval ->
        blocked (no human) or answered by the simulated human (``self.human``)."""
        self._last_approval = None
        if not self.cond["rule"] and self.cm is None and self.herkunft_cfg is None and self.hk is None:
            return None
        self._last_rule_directive = None
        self._ticket = None
        kind, rule_msg = self._decide_rule(tool, args) if self.cond["rule"] else (None, None)
        if kind == "block":
            return rule_msg
        desc = self._last_desc if kind == "approve" else None
        if self.hk is not None:
            kind, desc, rule_msg = self._decide_s25(tool, args, kind, desc, rule_msg)
            if kind == "block":
                return rule_msg
        if kind is None and self.herkunft_cfg is not None:
            kind, desc = self._decide_herkunft(tool, args)
            if kind == "block":
                return desc
            if kind == "approve":
                rule_msg = NO_HUMAN_TEMPLATE.format(description=desc)
        if self.cm is not None:
            try:
                d = self._critic(tool, args or {})
            except Exception as exc:  # critic_directive catches itself; fail-closed anyway
                return f"BLOCKED: alignment critic failed for {tool} (fail-closed): {exc}"
            if d is not None and d.get("action") == "block":
                if d.get("halt"):
                    self.halt_requested = True
                return d.get("message") or f"BLOCKED: {tool}"
            if d is not None and rule_msg is None:
                kind, desc = "approve", d.get("message", tool)
                rule_msg = NO_HUMAN_TEMPLATE.format(description=desc)
        if kind == "approve" and self.human is not None:
            if self.human.answer(tool, dict(args or {}), karte=self._karte(tool, args)):
                self._last_approval = "granted"
                self._finalize_ticket(True)
                return None
            self._last_approval = "denied"
            self._finalize_ticket(False)
            return DENIED_TEMPLATE.format(description=desc or tool)
        self._finalize_ticket(False)
        return rule_msg

    _last_rule_directive: Optional[Dict[str, Any]] = None

    def _decide_s25(self, tool: str, args, kind, desc, rule_msg):
        """Hermes funnel step (hermes_cli/plugins.py): S2 reason + S5
        relaxations of the 3b reason via ``session_gateway_herkunft.apply``.
        Returns (kind, desc, message); kind 'block' = S2 check error."""
        rule = self._last_rule_directive if kind == "approve" else None
        s25 = self.hk.apply(ax.key(tool), dict(args or {}), session_id="", taint_state=self.state, rule=rule)
        if s25 is None:
            return kind, desc, rule_msg
        ev: Dict[str, Any] = {"tool": tool}
        self.s25_events.append(ev)
        if s25.block is not None:
            ev["block"] = True
            return "block", None, s25.block.get("message") or f"BLOCKED: {tool}"
        self._ticket = s25.ticket
        if s25.gelockert:
            ev["gelockert"] = s25.gelockert
        if s25.herkunft is not None:
            ev["s2"] = True
            ev["codes"] = list(s25.herkunft.get("codes") or ())
        parts = [d.get("message") for d in (s25.rule, s25.herkunft) if d and d.get("message")]
        if s25.rule is None and s25.herkunft is None:
            return None, None, None
        d = " ".join(parts) or tool
        return "approve", d, NO_HUMAN_TEMPLATE.format(description=d)

    _last_desc: Optional[str] = None
    _last_approval: Optional[str] = None

    def _decide_herkunft(self, tool: str, args: Optional[Dict[str, Any]] = None):
        """S2 (mode ergaenzung): only reached when the 3b rule is silent.
        (kind, description): kind None | 'approve' | 'block' (check error -> fail-closed)."""
        from p3g_adapter import herkunft as hk

        try:
            if self._trusted is None:
                raise RuntimeError("provenance index without trusted text (set_trusted not called)")
            snap = self.state.snapshot()
            d = hk.pruefe(tool, dict(args or {}), self._trusted, self._results, tainted=bool(snap["tainted"]),
                          sensitive=bool(snap["sensitive"]), cfg=self.herkunft_cfg)
        except Exception as exc:
            return "block", f"BLOCKED: provenance check failed for {tool} (fail-closed): {exc}"
        if not d["freigabe"]:
            return None, None
        self.herkunft_events.append({"tool": tool, "gruende": d["gruende"]})
        return "approve", hk.kartentext(tool, d["zielwerte"])

    halt_requested = False

    def _decide_rule(self, tool: str, args: Optional[Dict[str, Any]] = None):
        """(kind, message): kind None (allow) | 'block' | 'approve'."""
        try:
            if self.cond.get("pass_args"):
                directive = self.gw.session_rule_directive(ax.key(tool), state=self.state, args=dict(args or {}))
            else:
                directive = self.gw.session_rule_directive(ax.key(tool), state=self.state)
        except Exception as exc:  # pragma: no cover - session_rule_directive catches itself
            return "block", f"BLOCKED: session security check failed for {tool} (fail-closed): {exc}"
        if directive is None:
            return None, None
        if directive.get("action") == "block":
            return "block", directive.get("message") or f"BLOCKED: {tool}"
        if directive.get("action") == "approve":
            # strict_once without a reachable human -> blocked (Paket 3b, Punkt 3).
            self._last_rule_directive = dict(directive)
            self._last_desc = directive.get("message", tool)
            return "approve", NO_HUMAN_TEMPLATE.format(description=directive.get("message", tool))
        return None, None

    def _record(self, tool: str, args: Dict[str, Any]) -> None:
        if self.gw is None:
            return
        try:
            self.gw.record_tool_result(ax.key(tool), args, state=self.state)
        except Exception:
            # Hermes: failed taint recording -> session counts as tainted+sensitive.
            self.state.mark(tainted=True, sensitive=True, source=f"{tool}:record_failed")

    def _format(self, tool: str, result: Any, args: Optional[Dict[str, Any]] = None) -> str:
        text = self.output_formatter(result)
        if self.hk is not None and (self.cond["switches"] or {}).get("steer_marker_entschaerfen"):
            # E6: Hermes make_tool_result_message defangs before wrapping
            st = importlib.import_module("agent.session_gateway_sondertoken")
            text = st.defang_steer_content(text)
        if self.leser is not None and _taint_quelle(tool):
            text = self._lies(tool, text, args)
        if self.cond["mark"] and self.gw is not None and self.gw.is_taint_source(ax.key(tool)):
            text = hermes_bridge.wrap_untrusted(tool, text)
        return text

    def _lies(self, tool: str, text: str, args: Optional[Dict[str, Any]]) -> str:
        """Quarantaene-Leser: getaintetes Ergebnis -> typisierte Leser-Ausgabe (fail-closed im Leser).
        Die Leser-Ausgabe bleibt fremd: sie wird zusaetzlich als Ergebnis desselben Tools in die
        Herkunfts-Indizes eingetragen (Hermes S2 ``note_result`` und Adapter-S2), damit auch umgeformte
        Werte nicht als 'unbekannt' durchrutschen."""
        out = self.leser.lies(tool, text)
        if out == text:
            return text
        if self.hk is not None:
            self.hk.note_result(ax.key(tool), dict(args or {}), out, state=self.state)
        if self._index_active():
            self._results.add(self._n, tool, dict(args or {}), out)
        if self.leser_merkmale and not self.leser_durchgereicht:
            from p3g_adapter.mensch import tokens

            self.leser_durchgereicht = bool(tokens(out) & self.leser_merkmale)
        return out

    # -- pipeline ------------------------------------------------------------
    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ):
        if len(messages) == 0 or messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        if not messages[-1]["tool_calls"]:
            return query, runtime, env, messages, extra_args

        results = []
        for tool_call in messages[-1]["tool_calls"]:
            if tool_call.function == EMPTY_FUNCTION_NAME:
                results.append(self._error(tool_call, "Empty function name provided. Provide a valid function name."))
                continue
            if tool_call.function not in (t.name for t in runtime.functions.values()):
                results.append(self._error(tool_call, f"Invalid tool {tool_call.function} provided."))
                continue
            for arg_k, arg_v in tool_call.args.items():
                if isinstance(arg_v, str) and is_string_list(arg_v):
                    tool_call.args[arg_k] = literal_eval(arg_v)

            before = self._snapshot()
            try:
                block = self._decide(tool_call.function, tool_call.args)
            except Exception as exc:
                block = f"BLOCKED: pre_tool_call security check failed for {tool_call.function} (fail-closed): {exc}"
            self._n += 1
            if block is not None:
                self.events.append(
                    GateEvent(self._n, tool_call.function, dict(tool_call.args), "blocked", block, before,
                              attempt=self.attempt, approval=self._last_approval)
                )
                results.append(self._error(tool_call, block, blocked=True))
                continue

            result, error = runtime.run_function(env, tool_call.function, tool_call.args)
            self._record(tool_call.function, tool_call.args)
            if self.hk is not None:
                self.hk.note_result(ax.key(tool_call.function), dict(tool_call.args), self.output_formatter(result),
                                    state=self.state)
            if self._index_active() or self.injection_texts:
                raw = self.output_formatter(result)
                if self._index_active():
                    self._results.add(self._n, tool_call.function, dict(tool_call.args), raw)
                if self.injection_texts and not self.injection_read:
                    from p3g_adapter.mensch import enthaelt_injektion

                    self.injection_read = enthaelt_injektion(raw, dict(enumerate(self.injection_texts)))
            if self.cm is not None and hasattr(self.cm, "note_step"):
                self.cm.note_step(None, ax.key(tool_call.function), dict(tool_call.args),
                                  self.output_formatter(result), state=self.state)
            self.executed_tool_calls.append(tool_call)
            self.events.append(GateEvent(self._n, tool_call.function, dict(tool_call.args), "executed", "", before,
                                         attempt=self.attempt, approval=self._last_approval))
            results.append(
                ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string(self._format(tool_call.function, result,
                                                                         dict(tool_call.args)))],
                    tool_call_id=tool_call.id,
                    tool_call=tool_call,
                    error=error,
                )
            )
        if self.halt_requested:
            # Hermes halts the turn from the N-th critic veto on (halt_decision_for).
            from agentdojo.agent_pipeline.errors import AbortAgentError

            raise AbortAgentError("alignment critic: repeated vetoes, turn halted", [*messages, *results], env)
        return query, runtime, env, [*messages, *results], extra_args

    @staticmethod
    def _error(tool_call, text: str, blocked: bool = False) -> ChatToolResultMessage:
        msg = ChatToolResultMessage(
            role="tool",
            content=[text_content_block_from_string("")],
            tool_call_id=tool_call.id,
            tool_call=tool_call,
            error=text,
        )
        if blocked:
            msg["p3g_blocked"] = True  # type: ignore[typeddict-unknown-key]
        return msg

    def new_attempt(self) -> None:
        """AgentDojo retries a run whose last message has no output; the
        session (TaintState) survives, the trace of executed calls restarts."""
        self.executed_tool_calls = []
        self.attempt += 1

    # -- reporting -----------------------------------------------------------
    def executed_calls(self) -> List[GateEvent]:
        return [e for e in self.events if e.action == "executed"]

    def blocked_calls(self) -> List[GateEvent]:
        return [e for e in self.events if e.action == "blocked"]
