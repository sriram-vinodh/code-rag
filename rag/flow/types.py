from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class StepComplexity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class PlanStep:
    step: str
    needs_context: bool
    complexity: StepComplexity
    required_context_description: Optional[str] = None
    fetched_context: Optional[str] = None
    knowledge_source: Optional[str] = None
    result: Optional[str] = None
    previous_step_result: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "needs_context": self.needs_context,
            "complexity": self.complexity.value,
            "required_context_description": self.required_context_description,
            "knowledge_source": self.knowledge_source,
            "result": self.result
        }

@dataclass
class ExecutionResult:
    success: bool
    message: str
    context: Optional[str] = None
    alternative_plan: Optional[List[PlanStep]] = None
