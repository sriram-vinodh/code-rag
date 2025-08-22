# Implementation migrated from cypher_query_helper.py
import re
from typing import Optional, List

from langchain_core.prompts import PromptTemplate

class CypherQueryHelper:
    """
    Constructs Cypher queries for the code knowledge graph based on user questions.
    """

    GRAPH_SCHEMA = """
    Node properties are the following:
    Class {name: string}
    Method {name: string}

    Relationship properties are the following:
    HAS_METHOD
    CALLS
    """

    CYPHER_GENERATION_TEMPLATE = """
    You are an expert Neo4j Cypher translator.
    You are given a question, and you need to convert it to a Cypher query.
    You must use the following schema:
    {schema}

    Question: {question}
    Cypher Query:
    """

    @staticmethod
    def generate_cypher_with_llm(llm, question: str) -> str:
        """
        Generates a Cypher query using an LLM.
        """
        prompt = PromptTemplate(
            template=CypherQueryHelper.CYPHER_GENERATION_TEMPLATE,
            input_variables=["schema", "question"],
        )
        chain = prompt | llm
        response = chain.invoke({{"schema": CypherQueryHelper.GRAPH_SCHEMA, "question": question}})
        return response.content


    @staticmethod
    def extract_identifiers(question: str) -> List[str]:
        return re.findall(r"[\w_]+", question)

    @staticmethod
    def build_flexible_query(question: str) -> Optional[str]:
        identifiers = CypherQueryHelper.extract_identifiers(question)
        question_lower = question.lower()
        if "class" in question_lower:
            for i, ident in enumerate(identifiers):
                if ident.lower() == "class" and i + 1 < len(identifiers):
                    class_name = identifiers[i + 1]
                    return (
                        f"MATCH (c:Class {{name: '{class_name}'}}) "
                        f"OPTIONAL MATCH (c)-[r:HAS_METHOD]->(m:Method) "
                        f"RETURN c, collect(r), collect(m) LIMIT 5"
                    )
        if "method" in question_lower:
            for i, ident in enumerate(identifiers):
                if ident.lower() == "method" and i + 1 < len(identifiers):
                    method_name = identifiers[i + 1]
                    return (
                        f"MATCH (m:Method {{name: '{method_name}'}}) "
                        f"OPTIONAL MATCH (m)-[r:CALLS]->(callee:Method) "
                        f"OPTIONAL MATCH (m)<-[rc:CALLS]-(caller:Method) "
                        f"RETURN m, collect(r), collect(callee), collect(rc), collect(caller) LIMIT 5"
                    )
        for ident in identifiers:
            method_query = (
                f"MATCH (m:Method {{name: '{ident}'}}) "
                f"OPTIONAL MATCH (m)-[r:CALLS]->(callee:Method) "
                f"OPTIONAL MATCH (m)<-[rc:CALLS]-(caller:Method) "
                f"RETURN m, collect(r), collect(callee), collect(rc), collect(caller) LIMIT 5"
            )
            class_query = (
                f"MATCH (c:Class {{name: '{ident}'}}) "
                f"OPTIONAL MATCH (c)-[r:HAS_METHOD]->(m:Method) "
                f"RETURN c, collect(r), collect(m) LIMIT 5"
            )
            if "class" in question_lower:
                return class_query
            else:
                return method_query
        fallback_query = "MATCH (m:Method) RETURN m LIMIT 3"
        return fallback_query
# Moved from cypher_query_helper.py
# CypherQueryHelper implementation
