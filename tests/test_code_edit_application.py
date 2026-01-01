"""
Test-Driven Development for Code Edit Application Feature

Tests for:
1. Parsing code edits from LLM responses in any step
2. Session/sandbox management for safe code modifications
3. Version control tracking of changes
4. Integration with Serena for applying edits
"""

import pytest
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag.flow.step_executor import StepExecutor
from rag.flow.types import PlanStep, ExecutionResult, StepComplexity


class TestCodeEditParsing:
    """Test parsing code edits from LLM responses."""
    
    def test_parse_code_edit_from_step_result(self):
        """Test detecting and parsing code_edit blocks in step results."""
        llm_response = """
        Here's my analysis:
        
        <code_edit>
          <type>symbolic</type>
          <target>CacheKey/validateKey</target>
          <file>src/cache/CacheKey.java</file>
          <body>
        public boolean validateKey() {
            return this.key != null && !this.key.isEmpty();
        }
          </body>
          <explanation>Add validation method</explanation>
        </code_edit>
        
        This will improve the class.
        """
        
        # Create executor with mock LLM
        mock_llm = Mock()
        executor = StepExecutor(mock_llm, None, None, None)
        
        # Parse edits
        edits = executor._parse_code_edits(llm_response)
        
        # Assertions
        assert len(edits) == 1
        assert edits[0].file_path == "src/cache/CacheKey.java"
        assert edits[0].target_symbol == "CacheKey/validateKey"
        assert "validateKey()" in edits[0].new_body
    
    def test_parse_multiple_code_edits(self):
        """Test parsing multiple code_edit blocks in one response."""
        llm_response = """
        <code_edit>
          <type>symbolic</type>
          <target>CacheKey/method1</target>
          <file>file1.java</file>
          <body>code1</body>
        </code_edit>
        
        <code_edit>
          <type>insertion</type>
          <target>CacheKey/method2</target>
          <file>file2.java</file>
          <body>code2</body>
        </code_edit>
        """
        
        mock_llm = Mock()
        executor = StepExecutor(mock_llm, None, None, None)
        edits = executor._parse_code_edits(llm_response)
        
        assert len(edits) == 2
        assert edits[0].target_symbol == "CacheKey/method1"
        assert edits[1].target_symbol == "CacheKey/method2"
    
    def test_parse_no_code_edits(self):
        """Test handling responses without code_edit blocks."""
        llm_response = "This is just analysis without code changes."
        
        mock_llm = Mock()
        executor = StepExecutor(mock_llm, None, None, None)
        edits = executor._parse_code_edits(llm_response)
        
        assert len(edits) == 0
    
    def test_parse_malformed_code_edit(self):
        """Test handling malformed code_edit blocks gracefully."""
        llm_response = """
        <code_edit>
          <type>symbolic</type>
          <!-- Missing target and file -->
          <body>some code</body>
        </code_edit>
        """
        
        mock_llm = Mock()
        executor = StepExecutor(mock_llm, None, None, None)
        edits = executor._parse_code_edits(llm_response)
        
        # Should skip malformed edits
        assert len(edits) == 0


class TestSessionManagement:
    """Test session/sandbox management for code modifications."""
    
    @pytest.fixture
    def session_manager(self, tmp_path):
        """Create a SessionManager instance with temp directory."""
        from rag.session.session_manager import SessionManager
        return SessionManager(workspace_root=str(tmp_path))
    
    def test_create_session(self, session_manager, tmp_path):
        """Test creating a new sandbox session."""
        # Create a test file in workspace
        test_file = tmp_path / "test.java"
        test_file.write_text("public class Test {}")
        
        # Create session
        session_id = session_manager.create_session(
            description="Test session"
        )
        
        # Assertions
        assert session_id is not None
        assert len(session_id) == 8  # Short UUID
        assert session_manager.get_session(session_id) is not None
    
    def test_session_isolation(self, session_manager, tmp_path):
        """Test that sessions are isolated (sandboxed)."""
        # Create original file
        original_file = tmp_path / "test.java"
        original_file.write_text("original content")
        
        # Create session
        session_id = session_manager.create_session("Test")
        
        # Modify file in session
        session_manager.write_file(
            session_id,
            "test.java",
            "modified content"
        )
        
        # Original should be unchanged
        assert original_file.read_text() == "original content"
        
        # Session file should be modified
        session_content = session_manager.read_file(session_id, "test.java")
        assert session_content == "modified content"
    
    def test_commit_session(self, session_manager, tmp_path):
        """Test committing session changes to main workspace."""
        # Create original file
        original_file = tmp_path / "test.java"
        original_file.write_text("original")
        
        # Create session and modify
        session_id = session_manager.create_session("Test")
        session_manager.write_file(session_id, "test.java", "modified")
        
        # Commit session
        session_manager.commit_session(session_id)
        
        # Original should now be modified
        assert original_file.read_text() == "modified"
    
    def test_rollback_session(self, session_manager, tmp_path):
        """Test rolling back (discarding) session changes."""
        original_file = tmp_path / "test.java"
        original_file.write_text("original")
        
        session_id = session_manager.create_session("Test")
        session_manager.write_file(session_id, "test.java", "modified")
        
        # Rollback
        session_manager.rollback_session(session_id)
        
        # Original unchanged
        assert original_file.read_text() == "original"
        
        # Session should be deleted
        assert session_manager.get_session(session_id) is None
    
    def test_list_sessions(self, session_manager):
        """Test listing all active sessions."""
        session1 = session_manager.create_session("Session 1")
        session2 = session_manager.create_session("Session 2")
        
        sessions = session_manager.list_sessions()
        
        assert len(sessions) == 2
        assert any(s['id'] == session1 for s in sessions)
        assert any(s['id'] == session2 for s in sessions)


class TestVersionControl:
    """Test version control tracking of changes."""
    
    @pytest.fixture
    def vc_tracker(self, tmp_path):
        """Create VersionControlTracker instance."""
        from rag.session.version_control import VersionControlTracker
        vc_file = tmp_path / ".rag_sessions.json"
        return VersionControlTracker(str(vc_file))
    
    def test_log_session(self, vc_tracker):
        """Test logging a session to version control."""
        session_data = {
            "id": "abc123",
            "description": "Test session",
            "created_at": datetime.now().isoformat(),
            "changes": [
                {"file": "test.java", "type": "modify"}
            ]
        }
        
        vc_tracker.log_session(session_data)
        
        # Verify it's persisted
        sessions = vc_tracker.get_sessions()
        assert len(sessions) == 1
        assert sessions[0]['id'] == "abc123"
    
    def test_get_session_history(self, vc_tracker):
        """Test retrieving session history."""
        # Log multiple sessions
        for i in range(3):
            vc_tracker.log_session({
                "id": f"session{i}",
                "description": f"Session {i}",
                "created_at": datetime.now().isoformat()
            })
        
        # Get history
        history = vc_tracker.get_sessions(limit=2)
        
        assert len(history) == 2
    
    def test_get_session_by_id(self, vc_tracker):
        """Test retrieving specific session by ID."""
        vc_tracker.log_session({
            "id": "target_session",
            "description": "Target"
        })
        
        session = vc_tracker.get_session_by_id("target_session")
        
        assert session is not None
        assert session['id'] == "target_session"
    
    def test_update_session_status(self, vc_tracker):
        """Test updating session status (committed/rolled back)."""
        vc_tracker.log_session({
            "id": "test_session",
            "status": "active"
        })
        
        # Update status
        vc_tracker.update_session_status("test_session", "committed")
        
        # Verify
        session = vc_tracker.get_session_by_id("test_session")
        assert session['status'] == "committed"


class TestCodeEditApplication:
    """Test applying code edits via Serena in steps."""
    
    @pytest.fixture
    def mock_serena_client(self):
        """Create mock Serena client."""
        mock = Mock()
        mock.replace_symbol_body = Mock(return_value=True)
        mock.insert_after_symbol = Mock(return_value=True)
        mock.insert_before_symbol = Mock(return_value=True)
        return mock
    
    @pytest.fixture
    def executor_with_serena(self, mock_serena_client):
        """Create StepExecutor with mock Serena."""
        mock_llm = Mock()
        return StepExecutor(
            llm=mock_llm,
            neo4j_retriever=None,
            mcp_client=None,
            serena_client=mock_serena_client
        )
    
    def test_apply_code_edits_from_step(
        self,
        executor_with_serena,
        mock_serena_client
    ):
        """Test that code edits in step results are automatically applied."""
        step = PlanStep(
            step="Enhance CacheKey class",
            needs_context=True,
            complexity=StepComplexity.MEDIUM,
            required_context_description="Get CacheKey methods",
            knowledge_source="hybrid"
        )
        
        # Mock LLM response with code edit
        llm_response = """
        <code_edit>
          <type>symbolic</type>
          <target>CacheKey/validateKey</target>
          <file>src/CacheKey.java</file>
          <body>public boolean validateKey() { return true; }</body>
        </code_edit>
        """
        
        # Mock the LLM to return this response
        executor_with_serena.llm.predict = Mock(return_value=llm_response)
        
        # Execute step with mocked context fetch
        with patch.object(
            executor_with_serena,
            '_fetch_hybrid_context',
            return_value="mock context"
        ):
            result = executor_with_serena._execute_single_step(
                step,
                accumulated_context=[],
                max_context_retries=1
            )
        
        # Verify Serena was called to apply the edit
        assert mock_serena_client.replace_symbol_body.called
        call_args = mock_serena_client.replace_symbol_body.call_args
        assert "CacheKey/validateKey" in str(call_args)
    
    def test_no_edits_applied_without_code_edit_blocks(
        self,
        executor_with_serena,
        mock_serena_client
    ):
        """Test that no edits are applied when response has no code_edit blocks."""
        step = PlanStep(
            step="Analyze code",
            needs_context=False,
            complexity=StepComplexity.LOW
        )
        
        # Mock LLM response without code edits
        llm_response = "This is just analysis without code changes."
        executor_with_serena.llm.predict = Mock(return_value=llm_response)
        
        # Execute step
        result = executor_with_serena._execute_single_step(
            step,
            accumulated_context=[],
            max_context_retries=1
        )
        
        # Verify Serena was NOT called
        assert not mock_serena_client.replace_symbol_body.called
    
    def test_session_created_before_applying_edits(
        self,
        executor_with_serena,
        mock_serena_client,
        tmp_path
    ):
        """Test that a sandbox session is created before applying edits."""
        from rag.session.session_manager import SessionManager
        
        # Setup session manager
        session_mgr = SessionManager(workspace_root=str(tmp_path))
        executor_with_serena.session_manager = session_mgr
        
        step = PlanStep(
            step="Add validation",
            needs_context=False,
            complexity=StepComplexity.LOW
        )
        
        llm_response = """
        <code_edit>
          <type>symbolic</type>
          <target>Test/method</target>
          <file>test.java</file>
          <body>code</body>
        </code_edit>
        """
        
        executor_with_serena.llm.predict = Mock(return_value=llm_response)
        
        # Execute step
        result = executor_with_serena._execute_single_step(
            step,
            accumulated_context=[],
            max_context_retries=1
        )
        
        # Verify session was created
        sessions = session_mgr.list_sessions()
        assert len(sessions) > 0


class TestEndToEndIntegration:
    """Test complete workflow with real components."""
    
    @pytest.mark.integration
    def test_enhancement_workflow_applies_edits(self, tmp_path):
        """
        Integration test: Enhancement suggestion should apply code edits.
        
        Workflow:
        1. User asks to enhance a class
        2. System analyzes and suggests improvements
        3. Suggestions include code_edit blocks
        4. Edits are automatically applied via Serena
        5. Changes are tracked in version control
        """
        # This will be implemented after the core components are built
        pytest.skip("Integration test - implement after core components")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
