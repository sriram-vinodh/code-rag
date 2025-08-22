import re
import logging
from typing import Optional, List
from helper_models.nl_to_cypher import generate_cypher, train_and_save

class CypherQueryHelper:
    """
    Constructs Cypher queries for the code knowledge graph based on user questions.
    Uses a trained RNN model for NL to Cypher translation.
    """
    
    GRAPH_SCHEMA = """
    Node Labels:
    - Class: Properties {name, file_path}
    - Method: Properties {name, signature}
    - Field: Properties {name}
    - Interface: Properties {name}
    - Package: Properties {name}
    
    Relationships:
    - HAS_METHOD: Class -> Method
    - HAS_FIELD: Class -> Field
    - CALLS: Method -> Method
    - IMPLEMENTS_INTERFACE: Class -> Interface
    - CONTAINS: Package -> Class/Interface
    """

    @staticmethod
    def generate_cypher_with_llm(llm, question: str) -> str:
        """
        Generates a Cypher query using the trained RNN model.
        Falls back to template-based generation if model fails.
        """
        try:
            cypher_query = generate_cypher(question)
            if cypher_query and len(cypher_query) > 10:  # Basic validation
                return cypher_query
        except Exception as e:
            logging.error(f"RNN model error: {e}. Falling back to template.")
        
        # Fallback to template-based approach
        return CypherQueryHelper.build_flexible_query(question)

    @staticmethod
    def extract_identifiers(question: str) -> List[str]:
        """Extract all words that could be identifiers (alphanumeric, underscores)"""
        return re.findall(r"[\w_]+", question)

    @staticmethod
    def build_flexible_query(question: str) -> Optional[str]:
        """Template-based fallback for query generation"""
        identifiers = CypherQueryHelper.extract_identifiers(question)
        question_lower = question.lower()
        
        # Prioritize context
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
        
        # Try both queries for each identifier
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
            # In a real system, you would run both and use the first with results
            # Here, just return the class query first if the question mentions class
            if "class" in question_lower:
                return class_query
            else:
                return method_query
        
        # Fallback: return a sample of methods/classes if no identifier found
        fallback_query = "MATCH (m:Method) RETURN m LIMIT 3"
        return fallback_query
