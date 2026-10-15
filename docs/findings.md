# Findings: hardening a local tool-using agent against prompt injection (Oct 2026)

Setup: [Hermes Agent](https://github.com/NousResearch/hermes-agent) as a personal assistant (chat front end, web,
files, code execution, sub-agents) on one NVIDIA Jetson Thor (122 GB unified memory), model
Nemotron-3.5-Lightning-30B-A3B (NVFP4) served by vLLM 0.27.1 with LoRA adapters. All injection numbers below come
from this repository's bench, profile `schnell` (per version: 24 benign runs, 160 attack runs over AgentDojo's
workspace/travel/banking/slack suites and four attack styles), compared pairwise on the same plan.

Negative results are reported on purpose; most of them were surprising to us.

## 1. Training the model against injections did not help

| Version (guardrail off) | attack succeeded | attack attempted | solved w/o attack |
|---|---|---|---|
| base model | 21.9 % | – | – |
| adapter P4c (general tool-use training) | 15.6 % | 15.0 % | 91.7 % |
| adapter "injection training v2" (P4c + ~230 teacher-filtered injection refusals) | 23.7 % | 23.7 % | 87.5 % |
| adapter PK3 (tool breadth + small injection share) | 19.4 % | 19.4 % | 79.2 % |
| **production adapter** (older, broad behaviour SFT) | – | **6.2 %** | 87.5 % |

- Dedicated injection training made the model **worse** than its warm start (P4c → v2: +8.1 pp attempted attacks,
  McNemar p = 0.004) and also hurt a separate tool-call quality gate.
- Small injection shares inside broader training did nothing measurable.
- The most robust model was an adapter that was never trained on injections (6.2 % vs 15.0 % for P4c, p = 0.0005).
  We do not know why; it is a reminder to *measure* every candidate instead of assuming.

## 2. A deterministic session rule did

The guardrail ("session gateway", implemented as a Hermes patch) does not depend on the model:

1. **Taint tracking**: once a session has read untrusted content (web, mail, files from outside) *and* sensitive
   data, every call that can send data out or has an irreversible effect needs a human approval for exactly that call.
2. **Provenance rule**: a target value (recipient, URL, path, id) that appears in a tool result but not in the user's
   own text needs an approval, even in sessions that are not sensitive.
3. **Task list**: targets the user named in a real message (not quoted text, not attachments, not text written by the
   orchestrating model for a sub-agent) are free for non-irreversible calls.
4. **Bundling**: an approved target counts as listed for the rest of the conversation ("once per target").
5. **Sub-agent rules**: children inherit the parent's taint, get rights at most like the parent, and every tool that
   can write outside the child's own directory needs an approval (or a sandbox).
6. **Special-token defanging** of tool results (chat-template tokens, out-of-band markers), fail-closed.

| Version | effective attacks | attempted | solved under attack |
|---|---|---|---|
| P4c, guardrail off | 15.6 % | 15.0 % | 75.6 % |
| P4c, guardrail on, simulated human (oracle) | **0.0 %** (0/160) | 9.4 % | 72.5 % |
| production adapter, guardrail on, simulated human | **0.0 %** (0/160) | 1.9 % | 65.6 % |

The price is approvals. In real use the critical number is approval cards per conversation; two changes made it
bearable: a per-session approval for local code without network access, and approval requests of asynchronous
sub-agents that survive the end of the parent's turn (before, they were silently "denied").

## 3. Dual-LLM "quarantine reader": strong, but too expensive (rejected)

Variant V1: every untrusted tool result is replaced by a typed summary written by a second call of the base model
(no tools, grammar-constrained JSON, max 600 chars, 24 facts). Pre-registered acceptance (K1–K5) vs. the same
adapter without reader:

| | without reader | with reader | criterion |
|---|---|---|---|
| attack attempted (guardrail off) | 15.0 % | **4.4 %** (p = 0.002) | K1 ✓ |
| solved without attack | 91.7 % | **66.7 %** | K2 ✗ (−25 pp) |
| median latency per run | 84.5 s | 141 s (+67 %) | K3 ✗ |
| reader errors (fail-closed) | – | 0.2 % | K4 ✓ |
| effective attacks with guardrail | 0/160 | 1/160 | K5 ✗ |

The reader passed the *content* of detected instructions on as "facts" in 79 % of read attacks. The single
effective attack under the guardrail turned out to be an artefact of the bench's tool bridge (an unparsed URL
argument let one approval cover a whole tool; real Hermes tools were not affected). It led to two fixes: the bridge
now reports URL targets (`"bruecke": {"url_ziele": true}`), and the gateway never bundles tool-level fallback targets.

## 4. Token cost: a rule in the harness beat all training

Replay of 50 real assistant requests in an isolated instance (stubbed side effects, blind pairwise judge):

| | no budget | **sub-agent budget** |
|---|---|---|
| finished | 48 | 48 |
| prompt tokens, total | 43.0 M | **7.0 M (−84 %)** |
| most expensive task | 11.4 M | 1.1 M |
| median duration | 165 s | 118 s |
| blind judge (wins) | 12 | 9 (9 ties, 16 invalid; n.s.) |

The budget stops a sub-agent after 40 tool calls or 12 consecutive calls of the same tool and asks it for a final
answer. Almost all tokens had been spent by a few sub-agents looping hundreds of *successful* tool calls, which
existing loop guards (aimed at repeated failures) did not catch. Every trained adapter we tried was *worse* than the
production adapter in this replay (more tool calls and tokens), although several looked better on a synthetic
tool-call battery.

## 5. Smaller lessons

- **Measure "attempted", not only "succeeded".** With a guardrail, "succeeded" is ~0 for every model; only the trace
  of the injected goal call shows whether a model got more or less robust.
- **Pre-register acceptance criteria and pair the comparison.** With n = 160 attack runs, differences below ~5 pp are
  noise; McNemar on the same plan keys is much more sensitive than comparing two rates.
- **Check your counters.** Our replay counted every sub-agent model call twice (the streaming wrapper called the
  non-streaming one, both were instrumented). Ratios survived, absolute numbers were 2× too high.
- **Speculative decoding (DSpark) 3 vs 5 draft tokens**, with thinking and LoRA: orchestrator about equal, sub-agents
  (no thinking) 26–33 % slower per token with 5. vLLM's "mean acceptance length" includes the bonus token
  (maximum = k + 1), which we initially misread as "almost all drafts accepted".

## 6. From bench to production

What the measurements above changed in the live assistant (single user, chat front end), in order:

| Date | Change | Basis |
|---|---|---|
| 2026-10-14 | **Sub-agent budget** on (40 tool calls / 12 same-tool streak) | replay of 50 real requests: −84 % prompt tokens (section 4) |
| 2026-10-14 | Speculative decoding kept at **3** draft tokens | A/B 3 vs 5 with thinking + LoRA (section 5) |
| 2026-10-15 | **Session guardrail** on, exactly the profile measured here (`s2s5`: taint tracking, provenance rule, task list, bundling, sub-agent rules, special-token defanging fail-closed) | 0/160 effective attacks with two different adapters (section 2) |
| 2026-10-15 | Guardrail fixes from bench findings: never bundle tool-level fallback targets; task list reads `example.com.Next` correctly | K5 case of the reader measurement (section 3) |
| 2026-10-15 | Approvals of async sub-agents survive the end of the parent's turn; per-session approval for local code without network | test instance before these fixes: 13 approval cards for a single question |
| – | Quarantine reader **not** deployed | failed pre-registered acceptance (section 3) |
| – | No new adapter deployed; the older production adapter stays | every newer adapter was worse in the real-request replay (section 4) and none beat it on injection robustness (section 1) |

First day with the guardrail and the budget in production: three sub-agents reading local files triggered 37
approval checks, of which the user saw **2 cards** (answered "for this session"); the budget stopped all three
sub-agents after 12 consecutive terminal calls instead of the hundreds seen before.

Every candidate for production now goes through the standing check (`pruefstand/kandidat.sh`, rules R1–R5)
against the current production adapter before anyone looks at it.

## Glossary

| Term in code | Meaning |
|---|---|
| Prüfstand / pruefstand | test bench |
| Version, Kandidat, Referenz | version, candidate, reference |
| aus / an | off / on |
| ASR_wirksam / ASR_spur | attack succeeded / attack attempted (trace) |
| Mensch, Oracle, Karte | simulated human; ground-truth oracle; card-only judgement |
| Freigabe, Karte | approval, approval card |
| Sperre, Gateway | guardrail (session gateway) |
| Herkunftsregel (S2) | provenance rule |
| Aufgabenliste (S5), Bündelung | task list, bundling |
| Leser / Quarantäne-Leser | (quarantine) reader |
| Lauf, Plan, schnell / voll | run, plan, fast / full profile |
| Abnahme | acceptance check |
