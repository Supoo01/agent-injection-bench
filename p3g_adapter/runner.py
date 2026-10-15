"""Run AgentDojo tasks under a condition (ohne / mit / nur_markierung).

Modes
-----
(AgentDojo's own ``GroundTruthPipeline`` check lives in ``scripts/dryrun.py``.)
``replay-gt``       scripted agent replays the user task's ground-truth calls
                    through the gated executor (no LLM, no injection).
``replay-obedient`` like replay-gt with the benchmark attack in place; as soon as
                    a tool result contains an injection, the agent executes the
                    injection task's ground-truth calls (upper bound: agent
                    always obeys), then continues the user task.
``llm``             real model via an OpenAI-compatible endpoint (localhost).

Scoring: AgentDojo's own task checks (``utility``/``security`` incl. the
``*_from_traces`` variants). The trace passed to the checks contains only
calls that were really executed (``security``); the official variant over all
attempted calls is stored as ``security_attempted``.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage, ToolsExecutionLoop
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.attacks import load_attack
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.task_suite.task_suite import functions_stack_trace_from_messages
from agentdojo.types import ChatAssistantMessage, ChatMessage, get_text_content_as_str, text_content_block_from_string

from p3g_adapter.gate import CONDITIONS, HermesGateToolsExecutor, configure_condition
from p3g_adapter import hermes_bridge

BENCHMARK_VERSION = "v1.2.2"
ATTACK = "important_instructions"
# Contains "vllm_parsed" so AgentDojo's attack resolves the model name to "Local model".
PIPELINE_NAME = "vllm_parsed-nemotron35-p3g"
MAX_ITERS = 15


# ---------------------------------------------------------------------------
# LLM element (OpenAI-compatible local endpoint)
# ---------------------------------------------------------------------------

class LocalOpenAILLM(BasePipelineElement):
    """AgentDojo ``OpenAILLM`` for a local vLLM: ``developer`` -> ``system`` role,
    tolerant to malformed tool-argument JSON, counts calls/tokens."""

    def __init__(self, client, model: str, max_tokens: int = 8192, temperature: Optional[float] = None) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.name = model
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.arg_parse_errors = 0

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        from agentdojo.agent_pipeline.llms import openai_llm as ol

        oa_messages = [ol._message_to_openai(m, self.model) for m in messages]
        for m in oa_messages:
            if m.get("role") == "developer":
                m["role"] = "system"  # type: ignore[index]
        tools = [ol._function_to_openai(t) for t in runtime.functions.values()]
        kwargs: Dict[str, Any] = dict(model=self.model, messages=oa_messages, max_tokens=self.max_tokens)
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        completion = self.client.chat.completions.create(**kwargs)
        self.calls += 1
        if completion.usage is not None:
            self.prompt_tokens += completion.usage.prompt_tokens or 0
            self.completion_tokens += completion.usage.completion_tokens or 0
        msg = completion.choices[0].message
        tool_calls = None
        if msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments not an object")
                except Exception:
                    self.arg_parse_errors += 1
                    args = {}
                tool_calls.append(FunctionCall(function=tc.function.name, args=args, id=tc.id))
        content = [text_content_block_from_string(msg.content)] if msg.content is not None else None
        out = ChatAssistantMessage(role="assistant", content=content, tool_calls=tool_calls)
        return query, runtime, env, [*messages, out], extra_args

    def stats(self) -> Dict[str, int]:
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "arg_parse_errors": self.arg_parse_errors,
        }


# ---------------------------------------------------------------------------
# Scripted replay element (no LLM)
# ---------------------------------------------------------------------------

class ReplayLLM(BasePipelineElement):
    """Emits ground-truth calls one per turn, then the ground-truth answer.

    With ``injection_calls`` set it behaves like an agent that always obeys an
    injection: directly after the ``inject_after``-th call (the first call whose
    result carries an injection placeholder, found with AgentDojo's canaries) it
    executes ``injection_calls`` and then continues with the user task."""

    name = "replay"

    def __init__(
        self,
        user_calls: Sequence[FunctionCall],
        final_output: str,
        injection_calls: Optional[Sequence[FunctionCall]] = None,
        inject_after: Optional[int] = None,
    ) -> None:
        self.queue: List[FunctionCall] = [c.model_copy(deep=True) for c in user_calls]
        self.final_output = final_output
        self.injection_calls = [c.model_copy(deep=True) for c in (injection_calls or [])]
        self.inject_after = inject_after
        self.injected = False
        self._n = 0

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        if self.injection_calls and not self.injected and self.inject_after is not None and self._n >= self.inject_after:
            self.injected = True
            self.queue = [*self.injection_calls, *self.queue]
        if self.queue:
            call = self.queue.pop(0)
            self._n += 1
            call.id = f"replay-{self._n}"
            out = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("")], tool_calls=[call])
        else:
            out = ChatAssistantMessage(
                role="assistant", content=[text_content_block_from_string(self.final_output)], tool_calls=None
            )
        return query, runtime, env, [*messages, out], extra_args


def first_injected_call(suite, user_task) -> Optional[int]:
    """1-based index of the first ground-truth call whose result shows an
    injection placeholder (AgentDojo canaries), None if never."""
    from agentdojo.agent_pipeline.tool_execution import tool_result_to_str

    vectors = suite.get_injection_vector_defaults()
    canaries = {v: f"<CANARY---{v}--->" for v in vectors}
    env = user_task.init_environment(suite.load_and_inject_default_environment(canaries))
    runtime = FunctionsRuntime(suite.tools)
    for n, call in enumerate(user_task.ground_truth(env.model_copy(deep=True)), 1):
        result, _ = runtime.run_function(env, call.function, call.args, raise_on_error=True)
        text = tool_result_to_str(result)
        if any(c in text for c in canaries.values()):
            return n
    return None


# ---------------------------------------------------------------------------
# Single task run
# ---------------------------------------------------------------------------

def build_pipeline(llm: BasePipelineElement, gate: HermesGateToolsExecutor, system_message: str) -> AgentPipeline:
    loop = ToolsExecutionLoop([gate, llm], max_iters=MAX_ITERS)
    p = AgentPipeline([SystemMessage(system_message), InitQuery(), llm, loop])
    p.name = PIPELINE_NAME
    return p


def run_task(
    suite,
    user_task,
    injection_task,
    injections: Dict[str, str],
    condition: str,
    make_llm: Callable[[Any, Any], BasePipelineElement],
    system_message: str,
    human_factory: Optional[Callable[[Any], Any]] = None,
    spur: bool = False,
    leser_factory: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    """Mirror of ``TaskSuite.run_task_with_pipeline`` with the gated executor.

    Optional (Pruefstand Teil B, default off):
    ``human_factory(oracle)`` -> :class:`p3g_adapter.mensch.HumanSim` answering the
    approvals of this run (``res["freigaben"]``); ``spur=True`` records the trace
    measure ASR_spur over ALL pipeline attempts (``res["spur"]``) and the
    per-attempt counts of executed/blocked calls (``res["versuche"]``).
    ``leser_factory(user_prompt)`` -> :class:`p3g_adapter.quarantaene.Leser` (Quarantaene-Leser V1):
    every taint-source result reaches the model only as the reader's typed digest (``res["leser"]``)."""
    t0 = time.time()
    environment = suite.load_and_inject_default_environment(injections)
    task_env = user_task.init_environment(environment)
    pre_env = task_env.model_copy(deep=True)
    gate = HermesGateToolsExecutor(condition)
    gate.note_user_goal(user_task.PROMPT)
    gate.note_turn_start(system_message, user_task.PROMPT)  # Hermes S2/S5 (no-op without its switches)
    oracle = None
    if human_factory is not None or spur:
        from p3g_adapter import mensch

        oracle = mensch.oracle_fuer_lauf(suite, user_task, injection_task, injections)
        gate.injection_texts = list(oracle.injection_texts) + ([injection_task.GOAL] if injection_task is not None else [])
    if human_factory is not None:
        gate.human = human_factory(oracle)
    if leser_factory is not None:
        gate.leser = leser_factory(user_task.PROMPT)
        if oracle is not None and injection_task is not None:
            gate.leser_merkmale = set().union(*[set(d.get("toks") or ()) for d in oracle.inj_literals.values()])
    if human_factory is not None or gate.herkunft_cfg is not None:
        gate.set_trusted([system_message, user_task.PROMPT])
    llm = make_llm(user_task, injection_task, pre_env)
    runtime = FunctionsRuntime(suite.tools)
    messages: Sequence[ChatMessage] = []
    model_output = None
    error = None
    attempts = 0
    for _ in range(3):
        attempts += 1
        gate.new_attempt()
        pipeline = build_pipeline(llm, gate, system_message)
        try:
            _, _, task_env, messages, _ = pipeline.query(user_task.PROMPT, runtime, task_env, [], {})
        except AbortAgentError as e:
            task_env, messages = e.task_environment, e.messages
        except Exception as e:  # LLM/transport errors: count the run as failed
            error = f"{type(e).__name__}: {e}"[:500]
            break
        model_output = messages[-1]["content"] if messages and messages[-1]["role"] == "assistant" else None
        if model_output is not None:
            break

    executed = list(gate.executed_tool_calls)
    attempted = functions_stack_trace_from_messages(messages) if messages else []
    out = model_output or []
    utility = suite._check_task_result(user_task, out, pre_env, task_env, executed)
    res: Dict[str, Any] = {
        "user_task": user_task.ID,
        "injection_task": injection_task.ID if injection_task is not None else None,
        "condition": condition,
        "utility": bool(utility),
        "error": error,
        "attempts": attempts,
        "n_executed": len(executed),
        "n_attempted": len(attempted),
        "blocked": [e.tool for e in gate.blocked_calls()],
        "executed_tools": [c.function for c in executed],
        "final_state": gate._snapshot(),
        "seconds": round(time.time() - t0, 2),
    }
    if injection_task is not None:
        res["security"] = bool(suite._check_task_result(injection_task, out, pre_env, task_env, executed))
        res["security_attempted"] = bool(suite._check_task_result(injection_task, out, pre_env, task_env, attempted))
    if gate.cm is not None:
        res["critic"] = gate.critic_stats()
    if gate.human is not None:
        res["freigaben"] = gate.human.stats()
    if gate.hk is not None:
        res["s25"] = gate.s25_stats()
    if gate.herkunft_cfg is not None:
        res["herkunft"] = {"anfragen": len(gate.herkunft_events),
                           "tools": [e["tool"] for e in gate.herkunft_events]}
    if spur:
        res["versuche"] = [
            {"versuch": a, "ausgefuehrt": sum(1 for e in gate.events if e.attempt == a and e.action == "executed"),
             "gesperrt": sum(1 for e in gate.events if e.attempt == a and e.action == "blocked")}
            for a in sorted({e.attempt for e in gate.events})]
        if injection_task is not None:
            from p3g_adapter import mensch

            sp = mensch.SpurPruefer(oracle).bewerte((e.tool, e.args, e.action) for e in gate.events)
            sp["gelesen"] = bool(gate.injection_read)
            sp["quelle"] = "gate_alle_versuche"
            res["spur"] = sp
    if gate.leser is not None:
        res["leser"] = gate.leser.stats()
        if injection_task is not None and oracle is not None:
            res["leser"]["merkmal_durchgereicht"] = gate.leser_durchgereicht
            res["leser"]["merkmale"] = len(gate.leser_merkmale)
    if hasattr(llm, "stats"):
        res.update(llm.stats())
    if isinstance(llm, ReplayLLM) and injection_task is not None:
        res["replay_injected"] = llm.injected
    res["_messages"] = messages
    return res


# ---------------------------------------------------------------------------
# Suite run
# ---------------------------------------------------------------------------

def _dummy_named_pipeline() -> BasePipelineElement:
    class _N(BasePipelineElement):
        name = PIPELINE_NAME

        def query(self, *a, **k):  # pragma: no cover
            raise RuntimeError("not runnable")

    return _N()


def plan_pairs(suite, user_ids: Optional[List[str]], inj_ids: Optional[List[str]], with_utility: bool, with_attack: bool,
               attack_name: str = ATTACK):
    users = [suite.user_tasks[u] for u in (user_ids or list(suite.user_tasks))]
    injs = [suite.injection_tasks[i] for i in (inj_ids or list(suite.injection_tasks))]
    attack = load_attack(attack_name, suite, _dummy_named_pipeline())
    pairs = []
    if with_utility:
        pairs += [(u, None, {}) for u in users]
    if with_attack:
        for u in users:
            for i in injs:
                pairs.append((u, i, attack.attack(u, i)))
    return pairs


def make_llm_factory(mode: str, args) -> Callable[[Any, Any, Any], BasePipelineElement]:
    if mode == "llm":
        import openai

        client = openai.OpenAI(api_key="EMPTY", base_url=args.base_url, timeout=600, max_retries=2)
        return lambda u, i, env: LocalOpenAILLM(client, args.model, max_tokens=args.max_tokens)

    def factory(user_task, injection_task, env):
        calls = user_task.ground_truth(env.model_copy(deep=True))
        inj_calls, after = None, None
        if mode == "replay-obedient" and injection_task is not None:
            inj_calls = injection_task.ground_truth(env.model_copy(deep=True))
            after = factory.inject_after.get(user_task.ID)
        return ReplayLLM(calls, user_task.GROUND_TRUTH_OUTPUT, inj_calls, after)

    factory.inject_after = {}  # type: ignore[attr-defined]
    return factory


def strip_private(r: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def run_suite(args) -> Dict[str, Any]:
    cond = configure_condition(args.condition)
    suite = get_suite(BENCHMARK_VERSION, args.suite)
    system_message = load_system_message(None)
    with_utility = not args.attack_only
    with_attack = not args.utility_only
    pairs = plan_pairs(suite, args.user_tasks, args.injection_tasks, with_utility, with_attack)
    factory = make_llm_factory(args.mode, args)
    if args.mode == "replay-obedient":
        users = {u.ID: u for u, _, _ in pairs}
        factory.inject_after = {uid: first_injected_call(suite, u) for uid, u in users.items()}  # type: ignore[attr-defined]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / f"{args.suite}_{args.condition}_{args.mode}_rep{args.rep}.jsonl"
    msg_path = out_dir / "messages" / f"{args.suite}_{args.condition}_{args.mode}_rep{args.rep}.jsonl"
    msg_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    t0 = time.time()
    results: List[Dict[str, Any]] = []

    def one(pair):
        u, i, inj = pair
        try:
            r = run_task(suite, u, i, inj, args.condition, factory, system_message)
        except Exception as e:
            r = {
                "user_task": u.ID, "injection_task": i.ID if i else None, "condition": args.condition,
                "utility": False, "security": False if i else None, "error": f"harness {type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-2000:],
            }
        r.update(suite=args.suite, mode=args.mode, rep=args.rep)
        return r

    with open(res_path, "w") as fr, open(msg_path, "w") as fm, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, p) for p in pairs]
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            msgs = r.pop("_messages", None)
            with lock:
                results.append(r)
                fr.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                fr.flush()
                if msgs is not None:
                    fm.write(json.dumps({"user_task": r["user_task"], "injection_task": r["injection_task"],
                                         "messages": msgs}, ensure_ascii=False, default=str) + "\n")
            if args.progress and n % args.progress == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {n}/{len(pairs)}", file=sys.stderr, flush=True)

    summary = summarize(results)
    summary.update(
        suite=args.suite, condition=args.condition, mode=args.mode, rep=args.rep,
        benchmark_version=BENCHMARK_VERSION, attack=ATTACK, n_pairs=len(pairs),
        seconds=round(time.time() - t0, 1), switches=cond["switches"],
        hermes=hermes_bridge.loaded_provenance(),
    )
    (out_dir / f"{args.suite}_{args.condition}_{args.mode}_rep{args.rep}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1)
    )
    return summary


def _rate(xs: List[bool]) -> Optional[float]:
    return round(sum(xs) / len(xs), 4) if xs else None


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    benign = [r for r in results if r.get("injection_task") is None]
    attacked = [r for r in results if r.get("injection_task") is not None]
    return {
        "n_benign": len(benign),
        "n_attacked": len(attacked),
        "utility_benign": _rate([r["utility"] for r in benign]),
        "utility_under_attack": _rate([r["utility"] for r in attacked]),
        "targeted_asr": _rate([bool(r.get("security")) for r in attacked]),
        "targeted_asr_attempted": _rate([bool(r.get("security_attempted")) for r in attacked]),
        "runs_with_block": sum(1 for r in results if r.get("blocked")),
        "blocked_calls": sum(len(r.get("blocked") or []) for r in results),
        "errors": sum(1 for r in results if r.get("error")),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="slack")
    ap.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    ap.add_argument("--mode", choices=["replay-gt", "replay-obedient", "llm"], required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--user-tasks", nargs="*")
    ap.add_argument("--injection-tasks", nargs="*")
    ap.add_argument("--utility-only", action="store_true")
    ap.add_argument("--attack-only", action="store_true")
    ap.add_argument("--base-url", default="http://localhost:8012/v1")
    ap.add_argument("--model", default="coding")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--progress", type=int, default=0)
    args = ap.parse_args(argv)
    if args.mode == "llm":
        from urllib.parse import urlparse

        host = urlparse(args.base_url).hostname
        if host not in ("localhost", "127.0.0.1", "::1"):
            ap.error("only localhost endpoints are allowed")
    s = run_suite(args)
    print(json.dumps({k: v for k, v in s.items() if k != "hermes"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
