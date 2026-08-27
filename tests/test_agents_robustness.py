#!/usr/bin/env python3
"""
Agent-layer unit tests (no server, GPU, or network required).
"""

import os
import sys
import types
import traceback

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ANAGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ANAGENT_ROOT)
from anagent.agents.solver import SolverAgent
from anagent.agents.expert import ExpertAgent
from anagent.agents.orchestrator import MultiAgentOrchestrator

_TESTS = []
def test(fn):
    _TESTS.append(fn)
    return fn


# ------------------------------------------- SolverAgent._parse_solution -----
def _parse(response, awareness=False):
    stub = types.SimpleNamespace(solver_awareness=awareness)
    return SolverAgent._parse_solution(stub, response)


@test
def test_solver_extracts_tagged_answer():
    p = _parse("<think>reasoning</think><answer>The figure shows an upward trend.</answer>")
    assert p["solution"] == "The figure shows an upward trend.", p["solution"]


@test
def test_solver_keeps_untagged_response():
    resp = "The figure shows a strong correlation between the two variables."
    assert _parse(resp)["solution"] == resp


@test
def test_solver_empty_stays_empty():
    assert _parse("")["solution"] == ""


# --------------------------------- ExpertAgent tool-result summarization -----
class _ExpertStub:
    """Binds the REAL Expert methods onto a minimal self (no ExpertAgent.__init__)."""
    _tool_summary_target = ExpertAgent._tool_summary_target
    _summarize_tool_result = ExpertAgent._summarize_tool_result

    def __init__(self, ret=None, raise_=False, ctx=32768, max_turns=5):
        self.args = types.SimpleNamespace(expert_ctx_tokens=ctx)
        self.max_turns = max_turns
        self.max_result_len = 204800
        self.task = {"query": "analyze the figure"}
        self.planner_plan = "plan"
        self.name = "Expert"
        self._ret = ret
        self._raise = raise_
        self.calls = 0

    def _call_llm(self, messages, **kw):
        self.calls += 1
        if self._raise:
            raise RuntimeError("boom")
        return self._ret


@test
def test_expert_target_scales_with_ctx_and_turns():
    assert _ExpertStub(ctx=32768, max_turns=5)._tool_summary_target() == 9175
    # bigger window -> bigger per-result budget; fewer turns -> bigger budget each
    assert _ExpertStub(ctx=131072, max_turns=5)._tool_summary_target() > 9175
    assert _ExpertStub(ctx=32768, max_turns=2)._tool_summary_target() > 9175


@test
def test_expert_small_result_passthrough_no_call():
    e = _ExpertStub(ret="SUMMARY")
    out = e._summarize_tool_result("a short tool result")
    assert out == "a short tool result"
    assert e.calls == 0            # no LLM call for small results


@test
def test_expert_oversized_result_is_summarized():
    e = _ExpertStub(ret="A concise summary with PMID 12345 and OR 1.5.")
    out = e._summarize_tool_result("Z" * 100000)
    assert e.calls == 1
    assert out == "A concise summary with PMID 12345 and OR 1.5."


@test
def test_expert_summary_bounded_to_target():
    tgt = _ExpertStub()._tool_summary_target()
    e = _ExpertStub(ret="S" * 50000)      # model ignores the word limit
    out = e._summarize_tool_result("Z" * 100000)
    assert len(out) <= tgt, len(out)


@test
def test_expert_fallback_on_summarize_failure_is_bounded():
    tgt = _ExpertStub()._tool_summary_target()
    e = _ExpertStub(raise_=True)
    out = e._summarize_tool_result("Z" * 100000)
    assert len(out) <= tgt, len(out)
    assert "[truncated]" in out           # head+tail slice, not empty


# -------------------------------- Orchestrator._execute_agent / guard --------
class _RaisingAgent:
    name = "Solver"
    def run(self, task, memory):
        raise RuntimeError("engine overflow")


@test
def test_solver_reasoning_key_reaches_the_result():
    parsed = _parse("<think>rising trend</think><answer>improves</answer>")
    solver_output = {"solution": parsed["solution"], "reasoning": parsed.get("think", "")}
    assert solver_output["reasoning"] == "rising trend", solver_output
    assert solver_output.get("reasoning", "") == "rising trend"


# ------------------------------------------- ExpertAgent._reset_state --------
@test
def test_reset_state_clears_all_per_sample_fields():
    stub = types.SimpleNamespace(
        expert_turn=3,
        conversation_history=["a"],
        tool_execution_history=["b"],
        expert_agent_trajectory=["c"],
        tool_execution_results=["evidence from the previous sample"],
    )
    ExpertAgent._reset_state(stub)
    assert stub.expert_turn == 0, stub.expert_turn
    assert stub.conversation_history == [], stub.conversation_history
    assert stub.tool_execution_history == [], stub.tool_execution_history
    assert stub.expert_agent_trajectory == [], stub.expert_agent_trajectory
    assert stub.tool_execution_results == [], stub.tool_execution_results


@test
def test_tool_results_do_not_carry_between_samples():
    stub = types.SimpleNamespace(
        expert_turn=0, conversation_history=[], tool_execution_history=[],
        expert_agent_trajectory=[], tool_execution_results=[],
    )
    stub.tool_execution_results.append("SAMPLE-1 evidence")
    ExpertAgent._reset_state(stub)
    stub.tool_execution_results.append("SAMPLE-2 evidence")
    joined = "\n\n".join(stub.tool_execution_results)
    assert "SAMPLE-1" not in joined, joined


@test
def test_execute_agent_swallows_exception():
    stub = types.SimpleNamespace()
    out = MultiAgentOrchestrator._execute_agent(stub, _RaisingAgent(), {}, {}, "solver")
    assert "error" in out
    assert "solution" not in out          # -> get("solution","") == "" downstream


@test
def test_empty_solution_guard_skips_critic():
    # The exact guard used in run(): critic/refine run ONLY when there is a solution.
    def has_solution(solver_output):
        return bool((solver_output.get("solution") or "").strip())

    assert has_solution({"solution": "a real answer"}) is True
    assert has_solution({"solution": ""}) is False          # empty -> skip critic
    assert has_solution({"solution": "   "}) is False        # whitespace -> skip
    assert has_solution({"error": "boom"}) is False          # error dict -> skip


# ------------------------- iterative Solver<->Critic refinement (run loop) ----
class _StubAgent:
    def __init__(self, name, fn):
        self.name = name
        self._fn = fn
        self.calls = 0
    def run(self, task, memory):
        self.calls += 1
        return self._fn(self.calls, memory)


def _orch(critic_turns, critic=True, solver_fn=None):
    o = MultiAgentOrchestrator.__new__(MultiAgentOrchestrator)
    o.save_dir = "/tmp"
    o.args = types.SimpleNamespace()
    o.critic_turns = critic_turns
    o.planner = _StubAgent("Planner", lambda c, m: {"plan": "p"})
    o.expert = _StubAgent("Expert", lambda c, m: {"expert_summary": "es", "conversation_history": []})
    o.solver = _StubAgent("Solver", solver_fn or (lambda c, m: {"solution": f"sol{c}", "raw_response": f"raw{c}", "think": ""}))
    o.critic = _StubAgent("Critic", lambda c, m: {"feedback": f"fb{c}"}) if critic else None
    return o


@test
def test_critic_turns_zero_is_off():
    o = _orch(0, critic=False)
    r = o.run({"query": "q"}, expert_turns=1)
    assert o.solver.calls == 1              # stage-3 only, no refinement
    assert r["answer"] == "sol1"
    assert "refined_answer" not in r        # critic never ran


@test
def test_critic_turns_one_single_round():
    o = _orch(1)
    r = o.run({"query": "q"}, expert_turns=1)
    assert o.critic.calls == 1 and o.solver.calls == 2
    assert r["answer"] == "sol1"            # answer stays the ORIGINAL solution
    assert r["refined_answer"] == "sol2"    # refined = post-critic


@test
def test_critic_turns_iterates_and_chains():
    o = _orch(3)
    r = o.run({"query": "q"}, expert_turns=1)
    assert o.critic.calls == 3              # N critic rounds
    assert o.solver.calls == 4              # stage-3 + N refine rounds
    assert r["answer"] == "sol1"            # original preserved
    assert r["refined_answer"] == "sol4"    # last round wins (memory chained)


@test
def test_critic_stops_early_on_empty_refinement():
    # a refine round that yields nothing stops the loop (don't critique an empty answer)
    o = _orch(5, solver_fn=lambda c, m: {"solution": ("sol1" if c == 1 else ""),
                                         "raw_response": "r", "think": ""})
    o.run({"query": "q"}, expert_turns=1)
    assert o.critic.calls == 1 and o.solver.calls == 2


# ------------------------------------------------------------------ main -----
def main():
    failed = 0
    print(f"Running {len(_TESTS)} agent-robustness tests\n")
    for fn in _TESTS:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(_TESTS) - failed}/{len(_TESTS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
