
import logging
from typing import List, Dict

class PlanDrafter:
    """
    Drafts a multi-step reasoning plan for a user question, marking which steps require code context.
    Uses the LLM for plan drafting with prompt engineering.
    """
    @staticmethod
    def draft_plan(user_question: str, llm=None) -> List[Dict]:
        """
        Calls the LLM to draft a plan. The LLM should be instructed to:
        - Break the question into a step-by-step plan.
        - For each step, decide if code context from the knowledge graph is required.
        - If context is required, set 'context_required': true and provide a 'context_nl' field describing the needed context in natural language.
        - Output a JSON list of steps, each with 'description', 'context_required', and (if needed) 'context_nl'.
        """
        prompt = f"""
You are an expert AI assistant helping a developer answer questions about a codebase using a knowledge graph.
Given the following user question, draft a step-by-step plan to answer it.
For each step:
  - Write a clear description of the step.
  - Decide if code context from the code knowledge graph (e.g., call hierarchy, type hierarchy, etc.) is required.
  - If context is required, set 'context_required': true and provide a 'context_nl' field describing the needed context in natural language (e.g., 'Get the call hierarchy for method X').
  - If not, set 'context_required': false.
Output a JSON list of steps, each with 'description', 'context_required', and (if needed) 'context_nl'.

User question: {user_question}
"""
        if llm is None:
            raise ValueError("An LLM instance must be provided to draft the plan.")
        logging.info(f"Drafting plan with LLM for question: {user_question}")
        plan_json = llm(prompt)
        # Assume the LLM returns a valid JSON list of steps
        import json
        plan = json.loads(plan_json)
        logging.info(f"Drafted plan: {plan}")
        return plan
