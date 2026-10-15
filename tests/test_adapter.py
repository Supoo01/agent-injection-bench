"""Unit tests for the AgentDojo <-> Hermes session gateway adapter (stdlib unittest).

Run: venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402
from agentdojo.types import ChatAssistantMessage, text_content_block_from_string  # noqa: E402

from p3g_adapter import axes as ax  # noqa: E402
from p3g_adapter import hermes_bridge  # noqa: E402
from p3g_adapter.gate import CONDITIONS, HermesGateToolsExecutor, configure_condition  # noqa: E402
from p3g_adapter import runner  # noqa: E402

SLACK = get_suite(runner.BENCHMARK_VERSION, "slack")


def _calls(*specs):
    return [FunctionCall(function=f, args=a, id=f"c{i}") for i, (f, a) in enumerate(specs)]


def _run_calls(gate, specs, env=None):
    """Feed calls one per assistant turn through the gate; return (env, tool messages)."""
    env = env if env is not None else SLACK.load_and_inject_default_environment({})
    runtime = FunctionsRuntime(SLACK.tools)
    out = []
    for call in _calls(*specs):
        msg = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("")], tool_calls=[call])
        _, _, env, msgs, _ = gate.query("q", runtime, env, [msg], {})
        out.append(msgs[-1])
    return env, out


READ_GENERAL = ("read_channel_messages", {"channel": "general"})
SEND_DM = ("send_direct_message", {"recipient": "Alice", "body": "hi"})
WEB = ("get_webpage", {"url": "www.informations.com"})


class AxesTest(unittest.TestCase):
    def test_every_tool_of_every_suite_is_mapped(self):
        for name in ("slack", "banking", "workspace", "travel"):
            suite = get_suite(runner.BENCHMARK_VERSION, name)
            tools = {t.name for t in suite.tools}
            self.assertEqual(tools, set(ax.TOOL_MAP[name]), name)

    def test_table_consistent_and_values_valid(self):
        table = ax.hermes_axes_table()
        self.assertEqual(len(table), len({t for s in ax.TOOL_MAP.values() for t in s}))
        for k, v in table.items():
            self.assertTrue(k.startswith(ax.PREFIX))
            self.assertTrue(all(x in ("ja", "bedingt", "nein") for x in v), k)

    def test_send_tools_are_egress(self):
        for name in ("send_direct_message", "send_channel_message", "post_webpage", "invite_user_to_slack"):
            self.assertTrue(ax.classes(ax.TOOL_MAP["slack"][name])["egress_oder_irreversibel"], name)


class BridgeTest(unittest.TestCase):
    def setUp(self):
        configure_condition("mit")

    def test_loads_committed_snapshot_not_worktree(self):
        gw = hermes_bridge.gateway()
        self.assertTrue(Path(gw.__file__).resolve().is_relative_to(hermes_bridge.snapshot_dir().resolve()))
        prov = hermes_bridge.loaded_provenance()
        self.assertTrue(prov["commit"].startswith("c8341ea748"))
        self.assertEqual(len(prov["files_sha256"]), len(hermes_bridge.SNAPSHOT_FILES))

    def test_switches_reach_hermes_code(self):
        gw = hermes_bridge.gateway()
        configure_condition("mit")
        self.assertTrue(gw.session_rule_enabled() and gw.taint_tracking_enabled() and gw.fail_closed_enabled())
        configure_condition("nur_markierung")
        self.assertFalse(gw.session_rule_enabled())
        self.assertTrue(gw.taint_tracking_enabled())

    def test_hermes_uses_agentdojo_axes(self):
        gw = hermes_bridge.gateway()
        self.assertTrue(gw.is_taint_source(ax.key("read_channel_messages")))
        self.assertTrue(gw.reads_sensitive(ax.key("read_channel_messages")))
        self.assertTrue(gw.is_egress_or_irreversible(ax.key("send_direct_message")))
        self.assertFalse(gw.is_egress_or_irreversible(ax.key("read_inbox")))


class GateOhneTest(unittest.TestCase):
    def test_no_hermes_no_block_no_wrap(self):
        gate = HermesGateToolsExecutor("ohne")
        self.assertIsNone(gate.gw)
        env, msgs = _run_calls(gate, [READ_GENERAL, SEND_DM])
        self.assertEqual([e.action for e in gate.events], ["executed", "executed"])
        self.assertNotIn("untrusted_tool_result", msgs[0]["content"][0]["content"])
        self.assertEqual(len(env.slack.user_inbox["Alice"]), len(SLACK.load_and_inject_default_environment({}).slack.user_inbox["Alice"]) + 1)


class GateMitTest(unittest.TestCase):
    def setUp(self):
        configure_condition("mit")

    def test_tainted_and_sensitive_blocks_egress(self):
        gate = HermesGateToolsExecutor("mit")
        pre = SLACK.load_and_inject_default_environment({})
        n_alice = len(pre.slack.user_inbox["Alice"])
        env, msgs = _run_calls(gate, [READ_GENERAL, SEND_DM], env=pre)
        self.assertEqual([e.action for e in gate.events], ["executed", "blocked"])
        self.assertIn("BLOCKED: approval required", msgs[1]["error"])
        self.assertIn("Session rule", msgs[1]["error"])
        self.assertTrue(msgs[1].get("p3g_blocked"))
        self.assertEqual(len(env.slack.user_inbox["Alice"]), n_alice, "blocked call must not run")
        self.assertEqual([c.function for c in gate.executed_tool_calls], ["read_channel_messages"])

    def test_taint_only_does_not_block(self):
        gate = HermesGateToolsExecutor("mit")
        _run_calls(gate, [WEB, SEND_DM])
        self.assertEqual([e.action for e in gate.events], ["executed", "executed"])
        snap = gate.state.snapshot()
        self.assertTrue(snap["tainted"])
        self.assertFalse(snap["sensitive"])

    def test_non_egress_read_still_runs_after_taint(self):
        gate = HermesGateToolsExecutor("mit")
        _run_calls(gate, [READ_GENERAL, ("read_inbox", {"user": "Alice"})])
        self.assertEqual([e.action for e in gate.events], ["executed", "executed"])

    def test_bedingt_counts_as_yes(self):
        # get_users_in_channel: S=ja, T=bedingt -> session tainted and sensitive
        gate = HermesGateToolsExecutor("mit")
        _run_calls(gate, [("get_users_in_channel", {"channel": "general"}), SEND_DM])
        self.assertEqual(gate.events[-1].action, "blocked")

    def test_state_is_per_run(self):
        g1 = HermesGateToolsExecutor("mit")
        _run_calls(g1, [READ_GENERAL])
        g2 = HermesGateToolsExecutor("mit")
        _run_calls(g2, [SEND_DM])
        self.assertEqual(g2.events[-1].action, "executed")

    def test_taint_source_results_wrapped(self):
        gate = HermesGateToolsExecutor("mit")
        _, msgs = _run_calls(gate, [READ_GENERAL, ("get_channels", {})])
        text = msgs[0]["content"][0]["content"]
        self.assertTrue(text.startswith('<untrusted_tool_result source="read_channel_messages">'))
        self.assertTrue(text.rstrip().endswith("</untrusted_tool_result>"))

    def test_fail_closed_when_state_breaks(self):
        gate = HermesGateToolsExecutor("mit")

        class Broken:
            def snapshot(self):
                raise RuntimeError("boom")

        gate.state = Broken()
        gate._snapshot = lambda: {}
        _, msgs = _run_calls(gate, [SEND_DM])
        self.assertEqual(gate.events[-1].action, "blocked")
        self.assertIn("fail-closed", msgs[-1]["error"])


class GateNurMarkierungTest(unittest.TestCase):
    def test_wraps_but_never_blocks(self):
        configure_condition("nur_markierung")
        gate = HermesGateToolsExecutor("nur_markierung")
        _, msgs = _run_calls(gate, [READ_GENERAL, SEND_DM])
        self.assertEqual([e.action for e in gate.events], ["executed", "executed"])
        self.assertIn("<untrusted_tool_result", msgs[0]["content"][0]["content"])
        self.assertTrue(gate.state.snapshot()["tainted"])


class RunnerTest(unittest.TestCase):
    def _args(self, cond, mode, **kw):
        base = dict(suite="slack", condition=cond, mode=mode, rep=0, workers=2, out=str(ROOT / "runs" / "unittest"),
                    user_tasks=None, injection_tasks=None, utility_only=False, attack_only=False,
                    base_url="http://localhost:8012/v1", model="coding", max_tokens=1024, progress=0)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_replay_gt_ohne_full_utility(self):
        s = runner.run_suite(self._args("ohne", "replay-gt", utility_only=True))
        self.assertEqual(s["utility_benign"], 1.0)
        self.assertEqual(s["blocked_calls"], 0)

    def test_replay_obedient_ohne_asr_one(self):
        s = runner.run_suite(self._args("ohne", "replay-obedient", attack_only=True))
        self.assertEqual(s["targeted_asr"], 1.0)

    def test_replay_obedient_mit_lowers_asr_and_attempted_ge_executed(self):
        configure_condition("mit")
        s = runner.run_suite(self._args("mit", "replay-obedient", attack_only=True))
        self.assertLess(s["targeted_asr"], 1.0)
        self.assertGreaterEqual(s["targeted_asr_attempted"], s["targeted_asr"])
        self.assertGreater(s["blocked_calls"], 0)

    def test_attack_text_comes_from_benchmark(self):
        pairs = runner.plan_pairs(SLACK, ["user_task_0"], ["injection_task_1"], False, True)
        (_, inj_task, injections), = pairs
        self.assertTrue(injections)
        self.assertTrue(all(inj_task.GOAL in v for v in injections.values()))


class FakeCompletions:
    def __init__(self, message):
        self.message = message
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=self.message)],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class LocalLLMTest(unittest.TestCase):
    def _llm(self, message):
        comp = FakeCompletions(message)
        client = SimpleNamespace(chat=SimpleNamespace(completions=comp))
        return runner.LocalOpenAILLM(client, "coding", max_tokens=64), comp

    def _msgs(self):
        return [
            {"role": "system", "content": [text_content_block_from_string("sys")]},
            {"role": "user", "content": [text_content_block_from_string("hi")]},
        ]

    def test_system_role_and_tool_parsing(self):
        tc = SimpleNamespace(id="t1", function=SimpleNamespace(name="get_channels", arguments="{}"))
        llm, comp = self._llm(SimpleNamespace(content=None, tool_calls=[tc]))
        _, _, _, msgs, _ = llm.query("hi", FunctionsRuntime(SLACK.tools), messages=self._msgs())
        self.assertEqual(comp.kwargs["messages"][0]["role"], "system")
        self.assertEqual(comp.kwargs["tool_choice"], "auto")
        self.assertNotIn("temperature", comp.kwargs)
        self.assertEqual(msgs[-1]["tool_calls"][0].function, "get_channels")
        self.assertEqual(llm.stats()["llm_calls"], 1)

    def test_malformed_arguments_counted(self):
        tc = SimpleNamespace(id="t1", function=SimpleNamespace(name="read_inbox", arguments="{not json"))
        llm, _ = self._llm(SimpleNamespace(content="x", tool_calls=[tc]))
        _, _, _, msgs, _ = llm.query("hi", FunctionsRuntime(SLACK.tools), messages=self._msgs())
        self.assertEqual(msgs[-1]["tool_calls"][0].args, {})
        self.assertEqual(llm.stats()["arg_parse_errors"], 1)

    def test_only_localhost_endpoint(self):
        with self.assertRaises(SystemExit):
            runner.main(["--condition", "ohne", "--mode", "llm", "--out", "/nonexistent",
                         "--base-url", "http://example.org/v1"])


if __name__ == "__main__":
    unittest.main()
