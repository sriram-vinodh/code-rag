from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import logging

class RAGPipeline:
    """
    A RAG pipeline that orchestrates a retriever, a language model, and a prompt.
    """

    def __init__(self, llm: BaseLanguageModel, retriever: BaseRetriever, prompt_template: str):
        """
        Initializes the RAG pipeline with its core components.

        Args:
            llm (BaseLanguageModel): The language model to use for generation.
            retriever (BaseRetriever): The retriever to fetch relevant documents.
            prompt_template (str): The template for the prompt to the LLM.
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Build the RAG chain
        self.chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logging.getLogger(__name__).info("RAG pipeline built successfully.")

    def ask(self, question: str) -> str:
        """Queries the RAG chain with a question and returns the answer."""
        if not self.chain:
            return "The RAG pipeline is not available."
        return self.chain.invoke(question)