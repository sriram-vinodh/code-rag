
import logging
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .flow.prompt_processor import PromptProcessor
from .flow.step_executor import StepExecutor

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, llm: BaseLanguageModel, neo4j_retriever, prompt_template: str):
        self.llm = llm
        self.neo4j_retriever = neo4j_retriever
        self.prompt_template = prompt_template
        self.prompt_processor = PromptProcessor(llm)
        self.step_executor = StepExecutor(neo4j_retriever)

    def ask(self, question: str) -> str:
        logger.info(f"📝 Question received: '{question}'")

        # 1. Generate execution plan
        plan = self.prompt_processor.get_execution_plan(question)
        
        # 2. Execute each step and gather context
        context = self.step_executor.execute_plan(plan)
        
        # 3. Construct the RAG chain with all gathered context
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        rag_chain = prompt | self.llm | StrOutputParser()
        
        # 4. Generate answer using all gathered context
        answer = rag_chain.invoke({
            "context": context,
            "question": question,
            "execution_plan": [s.step for s in plan.steps]
        })
        
        logger.info("✅ Generated answer")
        return answer
