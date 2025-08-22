
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator
import logging
import json
from .types import PlanStep, StepComplexity, ExecutionResult

logger = logging.getLogger(__name__)

@dataclass
class TaskExecutionPlan:
    total_steps: int
    steps: List[PlanStep] = field(default_factory=list)
    current_step: int = 0

    @classmethod
    def from_dict(cls, plan_data: Dict[str, Any]) -> 'TaskExecutionPlan':
        steps = []
        step_data_list = plan_data.get("steps", [])
        
        if not step_data_list:
            raise ValueError("Plan data must contain at least one step")
            
        for step_data in step_data_list:
            # Log the step data for debugging
            logging.getLogger(__name__).debug("Processing step data: %s", json.dumps(step_data, indent=2))
            
            try:
                complexity = StepComplexity(step_data.get("complexity", "low").lower())
                needs_context = step_data.get("needs_context", False)
                context_desc = step_data.get("required_context_description") if needs_context else None
                knowledge_source = step_data.get("knowledge_source") if needs_context else None
                
                step = PlanStep(
                    step=step_data.get("step", "Unknown step"),
                    needs_context=needs_context,
                    complexity=complexity,
                    required_context_description=context_desc,
                    knowledge_source=knowledge_source
                )
                steps.append(step)
                
            except Exception as e:
                logging.getLogger(__name__).error(f"Error creating step from data: {e}")
                raise
        
        return cls(
            total_steps=plan_data.get("total_steps", len(steps)),
            steps=steps
        )

    def __iter__(self) -> Iterator[PlanStep]:
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def get_next_step(self) -> Optional[PlanStep]:
        if self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            if self.current_step > 0:
                step.previous_step_result = self.steps[self.current_step - 1].result
            return step
        return None

    def advance(self) -> bool:
        if self.current_step < len(self.steps):
            self.current_step += 1
            return True
        return False

    def update_step_result(self, result: ExecutionResult) -> None:
        if self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            step.result = result.message
            step.fetched_context = result.context

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "steps": [step.to_dict() for step in self.steps]
        }
