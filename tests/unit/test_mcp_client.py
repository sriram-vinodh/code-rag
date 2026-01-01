"""Unit tests for MCP Neo4j Client."""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from mcp_client import MCPNeo4jClient


@pytest.mark.unit
class TestMCPNeo4jClient:
    """Test suite for MCPNeo4jClient."""
    
    def test_initialization(self, test_config_file):
        """Test client initialization."""
        client = MCPNeo4jClient(config_path=test_config_file)
        
        assert client.config is not None
        assert client.mcp_process is None
        assert client.request_id == 0
        assert client.connected is False
    
    def test_load_config(self, test_config_file):
        """Test configuration loading."""
        client = MCPNeo4jClient(config_path=test_config_file)
        
        assert "neo4j_server" in client.config
        assert "command" in client.config["neo4j_server"]
    
    def test_load_config_missing_file(self):
        """Test config loading with missing file."""
        client = MCPNeo4jClient(config_path="nonexistent.json")
        
        # Should not raise, just return empty config
        assert client.config == {}
    
    @patch('subprocess.Popen')
    def test_connect_starts_process(self, mock_popen, test_config_file):
        """Test connect starts MCP server process."""
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stderr = MagicMock()
        mock_popen.return_value = mock_process
        
        client = MCPNeo4jClient(config_path=test_config_file)
        
        with patch.object(client, '_send_initialize'):
            client.connect()
        
        assert client.mcp_process is not None
        mock_popen.assert_called_once()
    
    def test_next_id_increments(self, test_config_file):
        """Test request ID increments."""
        client = MCPNeo4jClient(config_path=test_config_file)
        
        id1 = client._next_id()
        id2 = client._next_id()
        id3 = client._next_id()
        
        assert id1 < id2 < id3
        assert id3 == 3
    
    @patch('subprocess.Popen')
    def test_close_terminates_process(self, mock_popen, test_config_file):
        """Test close terminates MCP process."""
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        client = MCPNeo4jClient(config_path=test_config_file)
        client.mcp_process = mock_process
        client.connected = True
        
        client.close()
        
        mock_process.terminate.assert_called_once()
    
    @patch('subprocess.Popen')
    def test_close_terminates_process(self, mock_popen, test_config_file):
        """Test get_schema method."""
        client = MCPNeo4jClient(config_path=test_config_file)
        
        # Mock MCP server response with proper structure
        mock_response = {
            "result": {
                "nodes": ["Class", "Method"],
                "relationships": ["CALLS", "CONTAINS"]
            }
        }
        
        with patch.object(client, '_send_request', return_value=mock_response):
            schema = client.get_schema()
            
            assert schema is not None
            assert "nodes" in schema
            assert "relationships" in schema
    
    def test_execute_read_query(self, test_config_file):
        """Test execute_read_query method."""
        client = MCPNeo4jClient(config_path=test_config_file)
        
        # Mock MCP server response with proper structure
        mock_response = {
            "result": {
                "content": [{"text": json.dumps([{"name": "TestClass", "type": "class"}])}]
            }
        }
        
        with patch.object(client, '_send_request', return_value=mock_response):
            result = client.execute_read_query("MATCH (c:Class) RETURN c")
            
            assert result is not None
            assert len(result) == 1
            assert result[0]["name"] == "TestClass"
