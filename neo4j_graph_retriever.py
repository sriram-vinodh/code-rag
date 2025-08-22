import logging
from neo4j import GraphDatabase
from typing import Any, Dict, List
import json
import os
class Neo4jGraphRetriever:
    """
    Handles connection to Neo4j and executes Cypher queries for code knowledge graph retrieval.
    """
    def __init__(self, config_path: str = "config.json"):
        self.logger = logging.getLogger(__name__)
        self._load_config(config_path)
        self._connect()
    def _load_config(self, config_path: str):
        with open(config_path, "r") as f:
            config = json.load(f)
        graph_cfg = config.get("neo4j", {})
        self.uri = graph_cfg.get("uri", "neo4j://localhost:7687")
        self.user = graph_cfg.get("user", "neo4j")
        self.password = graph_cfg.get("password", "12345678")
        self.database = graph_cfg.get("database", "neo4j")

    def _connect(self):
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        if hasattr(self, "driver"):
            self.driver.close()

    def query(self, cypher: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        params = params or {}
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, **params)
            return [record.data() for record in result]

# Example usage:
# retriever = Neo4jGraphRetriever()
# results = retriever.query("MATCH (m:Method)-[:CALLS]->(c:Class) RETURN m, c LIMIT 10")
# retriever.close()
import logging
from neo4j import GraphDatabase
from typing import Any, Dict, List
import json
import os

class Neo4jGraphRetriever:
    """
    Handles connection to Neo4j and executes Cypher queries for code knowledge graph retrieval.
    """
    def __init__(self, config_path: str = "config.json"):
        self.logger = logging.getLogger(__name__)
        self._load_config(config_path)
        self._connect()

    def _load_config(self, config_path: str):
        with open(config_path, "r") as f:
            config = json.load(f)
        graph_cfg = config.get("neo4j", {})
        self.uri = graph_cfg.get("uri", "neo4j://localhost:7687")
        self.user = graph_cfg.get("user", "neo4j")
        self.password = graph_cfg.get("password", "12345678")
        self.database = graph_cfg.get("database", "neo4j")

    def _connect(self):
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        if hasattr(self, "driver"):
            self.driver.close()

    def query(self, cypher: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        params = params or {}
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, **params)
            return [record.data() for record in result]

# Example usage:
# retriever = Neo4jGraphRetriever()
# results = retriever.query("MATCH (m:Method)-[:CALLS]->(c:Class) RETURN m, c LIMIT 10")
# retriever.close()
