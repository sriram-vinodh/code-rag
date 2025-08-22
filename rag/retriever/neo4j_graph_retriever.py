import logging
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from typing import Any, Dict, List, Optional
import json
import os

class Neo4jGraphRetriever:
    """
    Handles connection to Neo4j and executes Cypher queries for code knowledge graph retrieval.
    """
    def __init__(self, config_path: str = "config.json"):
        """Initialize the Neo4j graph retriever with configuration."""
        self.logger = logging.getLogger(__name__)
        self.driver: Optional[GraphDatabase.driver] = None
        self.is_connected = False
        self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """Load Neo4j configuration from file."""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            graph_cfg = config.get("neo4j", {})
            self.uri = graph_cfg.get("uri", "neo4j://localhost:7687")
            self.user = graph_cfg.get("user", "neo4j")
            self.password = graph_cfg.get("password", "12345678")
            self.database = graph_cfg.get("database", "neo4j")
            self.logger.info(f"Loaded Neo4j configuration from {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load Neo4j configuration: {e}")
            raise

    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        if self.is_connected:
            return

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Verify connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            self.is_connected = True
            self.logger.info("Successfully connected to Neo4j")
        except ServiceUnavailable:
            self.logger.error("Neo4j service is not available. Check if the database is running.")
            raise
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self) -> None:
        """Close the Neo4j connection."""
        if self.driver:
            try:
                self.driver.close()
                self.is_connected = False
                self.logger.info("Neo4j connection closed")
            except Exception as e:
                self.logger.error(f"Error closing Neo4j connection: {e}")

    def query(self, cypher: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results."""
        if not self.is_connected:
            self.connect()

        params = params or {}
        try:
            self.logger.info("\n=== 🔍 Executing Neo4j Query ===\n%s\nParameters: %s\n===========================", 
                           cypher, json.dumps(params, indent=2))
            
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, **params)
                records = [record.data() for record in result]
                
                self.logger.info("\n=== 📊 Neo4j Query Results ===\n%s\n===========================", 
                               json.dumps(records, indent=2))
                return records
        except Exception as e:
            self.logger.error(f"Failed to execute Cypher query: {e}")
            raise
# Moved from neo4j_graph_retriever.py
# Neo4jGraphRetriever implementation
