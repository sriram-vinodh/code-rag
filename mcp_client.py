"""
MCP Client for Neo4j Cypher query generation and execution.
This client communicates with the Neo4j MCP server to generate and execute Cypher queries.
"""

import logging
import json
import subprocess
import sys
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class MCPNeo4jClient:
    """
    Client for interacting with the Neo4j MCP server.
    Uses stdio transport to communicate with the MCP server process.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the MCP client with configuration."""
        self.config = self._load_config(config_path)
        self.mcp_process = None
        self.request_id = 0
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get("mcp", {})
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            return {}
    
    def connect(self):
        """Start the MCP server process."""
        try:
            server_config = self.config.get("neo4j_server", {})
            command = server_config.get("command", "uvx")
            args = server_config.get("args", ["mcp-neo4j-cypher@0.5.1", "--transport", "stdio"])
            env = server_config.get("env", {})
            
            # Merge environment variables
            import os
            full_env = os.environ.copy()
            full_env.update(env)
            
            logger.info(f"Starting MCP server: {command} {' '.join(args)}")
            
            self.mcp_process = subprocess.Popen(
                [command] + args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=full_env,
                text=True,
                bufsize=1
            )
            
            logger.info("MCP server process started")
            
            # Initialize the connection
            self._send_initialize()
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise
    
    def _send_initialize(self):
        """Send initialize request to MCP server."""
        try:
            init_request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "code-rag",
                        "version": "1.0.0"
                    }
                }
            }
            
            response = self._send_request(init_request)
            logger.info("MCP server initialized")
            
            # Send initialized notification
            initialized = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            self._send_notification(initialized)
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP server: {e}")
            raise
    
    def _next_id(self) -> int:
        """Get next request ID."""
        self.request_id += 1
        return self.request_id
    
    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request and get response."""
        if not self.mcp_process:
            raise RuntimeError("MCP server not connected")
        
        try:
            # Send request
            request_line = json.dumps(request) + "\n"
            logger.debug(f"Sending MCP request: {request_line.strip()}")
            self.mcp_process.stdin.write(request_line)
            self.mcp_process.stdin.flush()
            
            # Read response
            response_line = self.mcp_process.stdout.readline()
            logger.debug(f"Received MCP response: {response_line.strip()}")
            
            if not response_line:
                raise RuntimeError("No response from MCP server")
            
            response = json.loads(response_line)
            
            if "error" in response:
                error_msg = response["error"].get("message", "Unknown error")
                logger.error(f"MCP server error: {error_msg}")
                raise RuntimeError(f"MCP error: {error_msg}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error communicating with MCP server: {e}")
            raise
    
    def _send_notification(self, notification: Dict[str, Any]):
        """Send a JSON-RPC notification (no response expected)."""
        if not self.mcp_process:
            raise RuntimeError("MCP server not connected")
        
        try:
            notification_line = json.dumps(notification) + "\n"
            logger.debug(f"Sending MCP notification: {notification_line.strip()}")
            self.mcp_process.stdin.write(notification_line)
            self.mcp_process.stdin.flush()
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    def list_tools(self) -> Optional[List[Dict[str, Any]]]:
        """
        List all available tools from the MCP server.
        
        Returns:
            List of available tools with their descriptions
        """
        try:
            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/list",
                "params": {}
            }
            
            response = self._send_request(request)
            
            if "result" in response:
                tools = response["result"].get("tools", [])
                logger.info(f"Found {len(tools)} available MCP tools")
                for tool in tools:
                    logger.info(f"  - {tool.get('name')}: {tool.get('description', 'No description')}")
                return tools
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return None
    
    def get_schema(self, sample_size: int = 1000) -> Optional[Dict[str, Any]]:
        """
        Get the Neo4j schema using MCP.
        
        Args:
            sample_size: Sample size for schema inference
            
        Returns:
            Schema information as a dictionary
        """
        try:
            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {
                    "name": "get_neo4j_schema",
                    "arguments": {
                        "sample_size": sample_size
                    }
                }
            }
            
            response = self._send_request(request)
            
            if "result" in response:
                schema_data = response["result"]
                logger.info("Retrieved Neo4j schema via MCP")
                return schema_data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get schema: {e}")
            return None
    
    def execute_read_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a read-only Cypher query using MCP.
        
        Args:
            query: Cypher query string
            params: Query parameters
            
        Returns:
            Query results as a list of dictionaries
        """
        try:
            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {
                    "name": "read_neo4j_cypher",
                    "arguments": {
                        "query": query,
                        "params": params or {}
                    }
                }
            }
            
            logger.info(f"Executing read query via MCP: {query}")
            response = self._send_request(request)
            
            if "result" in response:
                result_data = response["result"]
                # MCP returns content as a list with text content
                if isinstance(result_data, dict) and "content" in result_data:
                    content = result_data["content"]
                    if isinstance(content, list) and len(content) > 0:
                        text_content = content[0].get("text", "")
                        # Parse the JSON result
                        try:
                            results = json.loads(text_content)
                            logger.info(f"Query executed successfully, got {len(results)} results")
                            return results
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse MCP result as JSON: {text_content}")
                            return [{"raw_result": text_content}]
                
                # Fallback: return raw result
                return [result_data]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to execute read query: {e}")
            return None
    
    def execute_write_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a write Cypher query using MCP.
        
        Args:
            query: Cypher query string
            params: Query parameters
            
        Returns:
            Query results as a list of dictionaries
        """
        try:
            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {
                    "name": "write_neo4j_cypher",
                    "arguments": {
                        "query": query,
                        "params": params or {}
                    }
                }
            }
            
            logger.info(f"Executing write query via MCP: {query}")
            response = self._send_request(request)
            
            if "result" in response:
                result_data = response["result"]
                # Parse content similar to read_query
                if isinstance(result_data, dict) and "content" in result_data:
                    content = result_data["content"]
                    if isinstance(content, list) and len(content) > 0:
                        text_content = content[0].get("text", "")
                        try:
                            results = json.loads(text_content)
                            logger.info("Write query executed successfully")
                            return results
                        except json.JSONDecodeError:
                            return [{"raw_result": text_content}]
                
                return [result_data]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to execute write query: {e}")
            return None
    
    def query_with_natural_language(self, question: str, llm=None) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a natural language query by translating to Cypher and executing via MCP.
        
        Process:
        1. Translate NL → Cypher using LLM (MCP doesn't provide NL translation)
        2. Validate Cypher syntax via MCP EXPLAIN
        3. Execute via MCP read_neo4j_cypher tool
        
        Args:
            question: Natural language question about code structure
            llm: Language model for NL to Cypher translation (required)
            
        Returns:
            Query results as a list of dictionaries
        
        Note: The Neo4j MCP server (mcp-neo4j-cypher@0.5.1) provides 3 tools:
              - get_neo4j_schema: Get database schema
              - read_neo4j_cypher: Execute read Cypher queries
              - write_neo4j_cypher: Execute write Cypher queries
              It does NOT provide natural language query translation.
        """
        if not llm:
            logger.error("LLM is required for natural language queries")
            return None
            
        try:
            logger.info(f"Processing NL query via MCP: {question}")
            
            # Step 1: Generate Cypher using LLM (MCP doesn't do NL translation)
            cypher_query = self._generate_cypher_from_nl(question, llm)
            if not cypher_query:
                logger.error("Failed to generate Cypher query")
                return None
            
            # Step 2: Validate query syntax via MCP
            if not self._validate_cypher_via_mcp(cypher_query):
                logger.error("Generated query failed validation")
                return None
            
            # Step 3: Execute via MCP's read_neo4j_cypher tool
            logger.info(f"Executing via MCP: {cypher_query}")
            return self.execute_read_query(cypher_query)
            
        except Exception as e:
            logger.error(f"Failed to process natural language query: {e}")
            return None
    
    def _generate_cypher_from_nl(self, question: str, llm) -> Optional[str]:
        """
        Generate Cypher query from natural language using LLM.
        
        Note: This is done client-side because MCP server doesn't provide NL translation.
        """
        try:
            # Load cypher generation template
            import json
            with open("config.json") as f:
                config = json.load(f)
            templates_file = config.get("pipeline_settings", {}).get("templates_file", "prompt_templates.json")
            
            with open(templates_file) as f:
                templates = json.load(f)
            cypher_template = templates.get("rag", {}).get("cypher_generation", "")
            
            if not cypher_template:
                logger.error("Cypher generation template not found")
                return None
            
            # Generate query using LLM
            prompt = cypher_template.format(question=question)
            
            if hasattr(llm, 'invoke'):
                response = llm.invoke(prompt)
                cypher_query = response.content if hasattr(response, 'content') else str(response)
            else:
                cypher_query = llm.predict(prompt)
            
            # Clean up the query
            cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
            
            if cypher_query:
                logger.info(f"Generated Cypher: {cypher_query}")
                return cypher_query
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating Cypher: {e}")
            return None
    
    def _validate_cypher_via_mcp(self, cypher_query: str) -> bool:
        """Validate Cypher query syntax using MCP's EXPLAIN capability."""
        try:
            explain_query = f"EXPLAIN {cypher_query}"
            result = self.execute_read_query(explain_query)
            if result is not None:
                logger.info("Query validation passed via MCP")
                return True
            return False
        except Exception as e:
            logger.error(f"Query validation failed: {e}")
            return False
    
    def close(self):
        """Close the MCP server connection."""
        if self.mcp_process:
            try:
                self.mcp_process.stdin.close()
                self.mcp_process.terminate()
                self.mcp_process.wait(timeout=5)
                logger.info("MCP server closed")
            except Exception as e:
                logger.error(f"Error closing MCP server: {e}")
                self.mcp_process.kill()
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
