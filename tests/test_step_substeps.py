import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.flow.step_executor import StepExecutor
from rag.flow.types import PlanStep, StepComplexity


class QueueLLM:
    """LLM stub that returns queued responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen_prompts = []

    def predict(self, prompt: str):
        self.seen_prompts.append(prompt)
        if not self.responses:
            raise RuntimeError("LLM stub out of responses")
        return self.responses.pop(0)


class DummySerena:
    def __init__(self):
        self.used = False


def test_substeps_pipe_previous_results_and_use_serena():
    # Responses: sub1 result, sub1 completeness json, sub2 result, sub2 completeness json
    llm = QueueLLM([
        "sub1-result",
        '{"is_complete": true, "confidence": 1.0, "reason": "ok", "missing_info": null}',
        "sub2-result target symbol",
        '{"is_complete": true, "confidence": 1.0, "reason": "ok", "missing_info": null}',
    ])

    serena = DummySerena()
    executor = StepExecutor(llm=llm, neo4j_retriever=None, mcp_client=None, serena_client=serena)

    # Monkeypatch Serena fetch to mark utilization and return context
    def fake_fetch(_, context_description: str):
        serena.used = True
        return f"SerenaCTX for {context_description}"

    executor._fetch_serena_context = fake_fetch.__get__(executor, StepExecutor)
    executor.knowledge_sources["serena_ide"] = executor._fetch_serena_context

    sub1 = PlanStep(
        step="Graph lookup",
        needs_context=False,
        complexity=StepComplexity.LOW,
    )
    sub2 = PlanStep(
        step="Serena fetch",
        needs_context=True,
        complexity=StepComplexity.LOW,
        required_context_description="target symbol",
        knowledge_source="serena_ide",
    )
    parent = PlanStep(
        step="Parent composed",
        needs_context=False,
        complexity=StepComplexity.LOW,
        sub_steps=[sub1, sub2],
    )

    result = executor._execute_step(parent, accumulated_context=[])

    assert result.success
    assert "sub1-result" in result.message and "sub2-result" in result.message
    assert serena.used, "Serena path should be invoked in sub-step"
    # Second sub-step should see the first sub-step result
    assert sub2.previous_step_result == "sub1-result"
    assert result.context is not None and "SerenaCTX" in result.context
