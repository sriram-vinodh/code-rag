
import logging
import json
from typing import Dict
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .task_execution_plan import TaskExecutionPlan
from .types import StepComplexity

logger = logging.getLogger(__name__)

class PromptProcessor:
    def __init__(self, llm: BaseLanguageModel):
        """Initialize the prompt processor with LLM and load templates."""
        self.llm = llm
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates from the configured file."""
        try:
            with open("config.json") as f:
                config = json.load(f)
            templates_file = config.get("pipeline_settings", {}).get("templates_file", "prompt_templates.json")
            
            with open(templates_file) as f:
                templates = json.load(f)
            logger.info(f"Loaded prompt templates from {templates_file}")
            return templates.get("rag", {})
        except Exception as e:
            logger.error(f"Failed to load prompt templates: {e}")
            return {}

    def get_execution_plan(self, question: str) -> TaskExecutionPlan:
        """Generate an execution plan for the given question."""
        planner_template = self.templates.get("planner", "")
        if not planner_template:
            logger.error("Planner template not found")
            return self._get_fallback_plan()

        # Log the augmented prompt
        augmented_prompt = planner_template.format(question=question)
        logger.info("\n=== 📝 Augmented Prompt ===\n%s\n========================", augmented_prompt)

        planner_prompt = ChatPromptTemplate.from_template(planner_template)
        planner_chain = planner_prompt | self.llm | JsonOutputParser()

        try:
            plan_data = planner_chain.invoke({"question": question})
            logger.info("\n=== 📋 Generated Execution Plan ===\n%s\n=========================", 
                       json.dumps(plan_data, indent=2))
            return TaskExecutionPlan.from_dict(plan_data)
        except Exception as e:
            logger.error(f"❌ Failed to generate execution plan: {e}")
            return self._get_fallback_plan()

    def _get_fallback_plan(self) -> TaskExecutionPlan:
        """Return a basic single-step plan as fallback."""
        return TaskExecutionPlan.from_dict({
            "total_steps": 1,
            "steps": [{
                "step": "Answer the user's question with available information",
                "needs_context": False,
                "complexity": "low"
            }]
        })
