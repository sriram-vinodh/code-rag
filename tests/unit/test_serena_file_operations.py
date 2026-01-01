"""
Unit tests for Serena file interaction operations.

Tests cover:
- File reading operations (list_dir, get_symbols_overview, find_symbol)
- File modification operations (replace_symbol_body, replace_content)
- File insertion operations (insert_before_symbol, insert_after_symbol)
- Error handling and edge cases
- Metrics tracking for file operations
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from serena_client import SerenaClient, SerenaMetrics


class TestSerenaFileReadOperations(unittest.TestCase):
    """Test read-only file operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = SerenaClient(project_path="/test/project")
        self.client.is_connected = True
        self.client._project_activated = True
    
    def test_list_dir_basic(self):
        """Test basic directory listing."""
        mock_response = {
            "directories": ["src", "tests"],
            "files": ["main.py", "config.json"]
        }
        
        with patch.object(self.client, 'call_tool', return_value=mock_response):
            result = self.client.list_dir(".")
            
            self.assertIsNotNone(result)
            self.assertIn("directories", result)
            self.assertIn("files", result)
            self.assertEqual(len(result["directories"]), 2)
            self.assertEqual(len(result["files"]), 2)
    
    def test_list_dir_recursive(self):
        """Test recursive directory listing."""
        with patch.object(self.client, 'call_tool', return_value={"files": ["a.py", "b.py"]}) as mock_call:
            result = self.client.list_dir("src", recursive=True)
            
            mock_call.assert_called_once_with(
                "mcp_oraios_serena_list_dir",
                {"relative_path": "src", "recursive": True}
            )
            self.assertIsNotNone(result)
    
    def test_list_dir_not_connected(self):
        """Test list_dir when not connected."""
        self.client.is_connected = False
        
        result = self.client.list_dir(".")
        
        self.assertIsNone(result)
    
    def test_get_symbols_overview_basic(self):
        """Test getting symbols overview of a file."""
        mock_symbols = {
            "symbols": [
                {"name": "MyClass", "kind": "class", "line": 10},
                {"name": "my_function", "kind": "function", "line": 25}
            ]
        }
        
        with patch.object(self.client, 'call_tool', return_value=mock_symbols):
            result = self.client.get_symbols_overview("module.py")
            
            self.assertIsNotNone(result)
            self.assertIn("symbols", result)
            self.assertEqual(len(result["symbols"]), 2)
    
    def test_get_symbols_overview_with_depth(self):
        """Test symbols overview with depth parameter."""
        with patch.object(self.client, 'call_tool', return_value={}) as mock_call:
            self.client.get_symbols_overview("module.py", depth=2)
            
            mock_call.assert_called_once_with(
                "mcp_oraios_serena_get_symbols_overview",
                {"relative_path": "module.py", "depth": 2}
            )
    
    def test_find_symbol_basic(self):
        """Test finding a symbol without body."""
        mock_result = [{
            "name_path": "MyClass/my_method",
            "relative_path": "module.py",
            "line": 15,
            "kind": "method"
        }]
        
        with patch.object(self.client, 'call_tool', return_value=mock_result):
            result = self.client.find_symbol("MyClass/my_method")
            
            self.assertIsNotNone(result)
            self.assertEqual(result["name_path"], "MyClass/my_method")
    
    def test_find_symbol_with_body(self):
        """Test finding a symbol with its body."""
        mock_result = [{
            "name_path": "MyClass/my_method",
            "body": "def my_method(self):\n    return 42"
        }]
        
        with patch.object(self.client, 'call_tool', return_value=mock_result):
            result = self.client.find_symbol("MyClass/my_method", include_body=True)
            
            self.assertIsNotNone(result)
            self.assertIn("body", result)
    
    def test_find_symbol_with_relative_path(self):
        """Test finding a symbol in a specific file."""
        with patch.object(self.client, 'call_tool', return_value={}) as mock_call:
            self.client.find_symbol("MyClass", relative_path="module.py")
            
            # Verify the correct parameters were passed
            args = mock_call.call_args[0][1]
            self.assertEqual(args["relative_path"], "module.py")


class TestSerenaFileModificationOperations(unittest.TestCase):
    """Test file modification operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = SerenaClient(project_path="/test/project")
        self.client.is_connected = True
        self.client._project_activated = True
    
    def test_replace_symbol_body_success(self):
        """Test successful symbol body replacement."""
        new_body = "def my_method(self):\n    return 100"
        
        with patch.object(self.client, 'call_tool', return_value={"success": True}):
            result = self.client.replace_symbol_body(
                "MyClass/my_method",
                "module.py",
                new_body
            )
            
            self.assertTrue(result)
    
    def test_replace_symbol_body_tracks_metrics(self):
        """Test that replace_symbol_body tracks metrics."""
        with patch.object(self.client, 'call_tool', return_value={"success": True}):
            self.client.replace_symbol_body("Symbol", "file.py", "new code")
            
            self.assertEqual(self.client.metrics.total_operations, 1)
            self.assertIn("replace_symbol_body", self.client.metrics.operations_by_type)
    
    def test_replace_symbol_body_not_connected(self):
        """Test replace_symbol_body when not connected."""
        self.client.is_connected = False
        
        result = self.client.replace_symbol_body("Symbol", "file.py", "code")
        
        self.assertFalse(result)
    
    def test_replace_symbol_body_logs_governance(self):
        """Test that replace_symbol_body logs governance data."""
        with patch.object(self.client, 'call_tool', return_value={"success": True}):
            with patch.object(self.client, '_log_governance') as mock_log:
                self.client.replace_symbol_body("Symbol", "file.py", "code")
                
                # Verify governance logging was called
                mock_log.assert_called_once()
                log_data = mock_log.call_args[0][0]
                self.assertEqual(log_data["operation"], "replace_symbol_body")
                self.assertEqual(log_data["target_symbol"], "Symbol")
                self.assertEqual(log_data["file_path"], "file.py")
    
    def test_replace_content_literal_mode(self):
        """Test replace_content with literal string."""
        with patch.object(self.client, 'call_tool', return_value={"success": True}) as mock_call:
            result = self.client.replace_content(
                "file.py",
                "old_text",
                "new_text",
                is_regex=False
            )
            
            self.assertTrue(result)
            args = mock_call.call_args[0][1]
            self.assertEqual(args["mode"], "literal")
            self.assertEqual(args["needle"], "old_text")
            self.assertEqual(args["repl"], "new_text")
    
    def test_replace_content_regex_mode(self):
        """Test replace_content with regex pattern."""
        with patch.object(self.client, 'call_tool', return_value={"success": True}) as mock_call:
            result = self.client.replace_content(
                "file.py",
                r"def\s+(\w+)\(",
                r"async def \1(",
                is_regex=True
            )
            
            self.assertTrue(result)
            args = mock_call.call_args[0][1]
            self.assertEqual(args["mode"], "regex")
    
    def test_replace_content_tracks_metrics(self):
        """Test that replace_content tracks metrics."""
        with patch.object(self.client, 'call_tool', return_value={"success": True}):
            self.client.replace_content("file.py", "old", "new")
            
            self.assertEqual(self.client.metrics.total_operations, 1)
            self.assertIn("replace_content", self.client.metrics.operations_by_type)
    
    def test_replace_content_handles_failure(self):
        """Test replace_content when operation fails."""
        with patch.object(self.client, 'call_tool', return_value=None):
            result = self.client.replace_content("file.py", "old", "new")
            
            self.assertFalse(result)
            self.assertEqual(self.client.metrics.failed_operations, 1)


class TestSerenaFileInsertionOperations(unittest.TestCase):
    """Test file insertion operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = SerenaClient(project_path="/test/project")
        self.client.is_connected = True
        self.client._project_activated = True
    
    def test_insert_before_symbol(self):
        """Test inserting content before a symbol."""
        content = "# New comment\n"
        
        with patch.object(self.client, 'call_tool', return_value={"success": True}) as mock_call:
            result = self.client.insert_before_symbol(
                "MyClass/my_method",
                "module.py",
                content
            )
            
            self.assertTrue(result)
            args = mock_call.call_args[0][1]
            self.assertEqual(args["name_path"], "MyClass/my_method")
            self.assertEqual(args["relative_path"], "module.py")
            self.assertEqual(args["body"], content)
    
    def test_insert_after_symbol(self):
        """Test inserting content after a symbol."""
        content = "\n# End of class\n"
        
        with patch.object(self.client, 'call_tool', return_value={"success": True}) as mock_call:
            result = self.client.insert_after_symbol(
                "MyClass",
                "module.py",
                content
            )
            
            self.assertTrue(result)
            args = mock_call.call_args[0][1]
            self.assertEqual(args["name_path"], "MyClass")
    
    def test_insert_operations_track_metrics(self):
        """Test that insert operations track metrics."""
        with patch.object(self.client, 'call_tool', return_value={"success": True}):
            self.client.insert_before_symbol("Symbol", "file.py", "code")
            self.client.insert_after_symbol("Symbol2", "file.py", "code2")
            
            self.assertEqual(self.client.metrics.total_operations, 2)
            self.assertEqual(self.client.metrics.operations_by_type["insert_before_symbol"], 1)
            self.assertEqual(self.client.metrics.operations_by_type["insert_after_symbol"], 1)
    
    def test_insert_before_not_connected(self):
        """Test insert_before_symbol when not connected."""
        self.client.is_connected = False
        
        result = self.client.insert_before_symbol("Symbol", "file.py", "code")
        
        self.assertFalse(result)


class TestSerenaRenameOperation(unittest.TestCase):
    """Test rename symbol operation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = SerenaClient(project_path="/test/project")
        self.client.is_connected = True
        self.client._project_activated = True
    
    def test_rename_symbol_success(self):
        """Test successful symbol rename."""
        with patch.object(self.client, 'call_tool', return_value={"success": True}) as mock_call:
            result = self.client.rename_symbol(
                "MyClass/oldMethod",
                "module.py",
                "newMethod"
            )
            
            self.assertTrue(result)
            args = mock_call.call_args[0][1]
            self.assertEqual(args["name_path"], "MyClass/oldMethod")
            self.assertEqual(args["new_name"], "newMethod")
    
    def test_rename_tracks_metrics(self):
        """Test that rename tracks metrics."""
        with patch.object(self.client, 'call_tool', return_value={"success": True}):
            self.client.rename_symbol("Old", "file.py", "New")
            
            self.assertEqual(self.client.metrics.total_operations, 1)
            self.assertIn("rename_symbol", self.client.metrics.operations_by_type)
    
    def test_rename_logs_governance(self):
        """Test that rename logs governance data."""
        with patch.object(self.client, 'call_tool', return_value={"success": True}):
            with patch.object(self.client, '_log_governance') as mock_log:
                self.client.rename_symbol("OldName", "file.py", "NewName")
                
                mock_log.assert_called_once()
                log_data = mock_log.call_args[0][0]
                self.assertEqual(log_data["operation"], "rename_symbol")


class TestSerenaSearchOperation(unittest.TestCase):
    """Test pattern search operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = SerenaClient(project_path="/test/project")
        self.client.is_connected = True
        self.client._project_activated = True
    
    def test_search_for_pattern_literal(self):
        """Test searching for a literal pattern."""
        mock_results = {
            "file1.py": ["10: import os"],
            "file2.py": ["5: import os"]
        }
        
        with patch.object(self.client, 'call_tool', return_value=mock_results) as mock_call:
            result = self.client.search_for_pattern("import os")
            
            self.assertIsNotNone(result)
            self.assertGreater(len(result), 0)
            args = mock_call.call_args[0][1]
            self.assertEqual(args["substring_pattern"], "import os")
    
    def test_search_for_pattern_regex(self):
        """Test searching with regex pattern."""
        with patch.object(self.client, 'call_tool', return_value={}) as mock_call:
            result = self.client.search_for_pattern(r"class\s+\w+:", is_regex=True)
            
            args = mock_call.call_args[0][1]
            # Verify the pattern was passed correctly
            self.assertEqual(args["substring_pattern"], r"class\s+\w+:")
            # Result should be empty list since mock returns empty dict
            self.assertEqual(len(result), 0)
    
    def test_search_with_relative_path(self):
        """Test searching within a specific path."""
        with patch.object(self.client, 'call_tool', return_value=[]) as mock_call:
            self.client.search_for_pattern("TODO", relative_path="src/")
            
            args = mock_call.call_args[0][1]
            self.assertEqual(args["relative_path"], "src/")


class TestSerenaErrorHandling(unittest.TestCase):
    """Test error handling in file operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = SerenaClient(project_path="/test/project")
        self.client.is_connected = True
        self.client._project_activated = True
    
    def test_handle_tool_call_exception(self):
        """Test handling of exceptions during tool calls."""
        with patch.object(self.client, 'call_tool', side_effect=Exception("Connection lost")):
            # Operations should raise the exception (not handle it gracefully by default)
            with self.assertRaises(Exception) as context:
                self.client.find_symbol("Symbol")
            
            self.assertIn("Connection lost", str(context.exception))
            # Failed operations are tracked in the decorator
            self.assertEqual(self.client.metrics.failed_operations, 1)
    
    def test_handle_invalid_response(self):
        """Test handling of invalid tool responses."""
        with patch.object(self.client, 'call_tool', return_value="invalid"):
            result = self.client.list_dir(".")
            
            # Should return the result even if unexpected format
            self.assertIsNotNone(result)
    
    def test_operations_when_disconnected(self):
        """Test that operations fail gracefully when disconnected."""
        self.client.is_connected = False
        
        # All operations should return failure
        self.assertIsNone(self.client.find_symbol("Symbol"))
        self.assertFalse(self.client.replace_symbol_body("S", "f", "code"))
        self.assertFalse(self.client.replace_content("f", "old", "new"))
        self.assertFalse(self.client.insert_before_symbol("S", "f", "code"))
        self.assertFalse(self.client.rename_symbol("Old", "f", "New"))


class TestSerenaMetricsForFileOperations(unittest.TestCase):
    """Test metrics tracking for file operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = SerenaClient(project_path="/test/project")
        self.client.is_connected = True
        self.client._project_activated = True
    
    def test_success_rate_calculation(self):
        """Test success rate calculation with mixed results."""
        with patch.object(self.client, 'call_tool') as mock_call:
            # 3 successes
            mock_call.return_value = {"success": True}
            self.client.replace_symbol_body("S1", "f", "code")
            self.client.replace_content("f", "old", "new")
            
            # 1 failure
            mock_call.return_value = None
            self.client.replace_symbol_body("S2", "f", "code")
            
            # Success rate should be 66.67%
            success_rate = self.client.metrics.get_success_rate()
            self.assertAlmostEqual(success_rate, 66.67, places=1)
    
    def test_operations_by_type(self):
        """Test tracking of operations by type."""
        with patch.object(self.client, 'call_tool', return_value={"success": True}):
            self.client.find_symbol("S1")
            self.client.find_symbol("S2")
            self.client.replace_symbol_body("S3", "f", "code")
            self.client.list_dir(".")
            
            ops = self.client.metrics.operations_by_type
            self.assertEqual(ops["find_symbol"], 2)
            self.assertEqual(ops["replace_symbol_body"], 1)
            self.assertEqual(ops["list_dir"], 1)
    
    def test_average_duration_tracking(self):
        """Test average duration calculation."""
        with patch.object(self.client, 'call_tool', return_value={"success": True}):
            # Perform some operations (they will be measured)
            self.client.find_symbol("S1")
            self.client.replace_content("f", "old", "new")
            
            # Should have average duration
            avg_duration = self.client.metrics.get_avg_duration_ms()
            self.assertGreater(avg_duration, 0)
    
    def test_get_metrics_returns_complete_data(self):
        """Test that get_metrics returns all metric fields."""
        with patch.object(self.client, 'call_tool', return_value={}):
            self.client.find_symbol("Test")
            
            metrics = self.client.get_metrics()
            
            # Verify all expected fields
            self.assertIn("total_operations", metrics)
            self.assertIn("successful_operations", metrics)
            self.assertIn("failed_operations", metrics)
            self.assertIn("success_rate", metrics)
            self.assertIn("avg_duration_ms", metrics)
            self.assertIn("operations_by_type", metrics)
    
    def test_reset_metrics(self):
        """Test metrics reset."""
        with patch.object(self.client, 'call_tool', return_value={}):
            # Perform operations
            self.client.find_symbol("S1")
            self.client.replace_content("f", "old", "new")
            
            # Reset metrics
            self.client.reset_metrics()
            
            # All counters should be zero
            self.assertEqual(self.client.metrics.total_operations, 0)
            self.assertEqual(self.client.metrics.successful_operations, 0)
            self.assertEqual(self.client.metrics.failed_operations, 0)
            self.assertEqual(len(self.client.metrics.operations_by_type), 0)


def run_tests():
    """Run all tests and print results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSerenaFileReadOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestSerenaFileModificationOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestSerenaFileInsertionOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestSerenaRenameOperation))
    suite.addTests(loader.loadTestsFromTestCase(TestSerenaSearchOperation))
    suite.addTests(loader.loadTestsFromTestCase(TestSerenaErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestSerenaMetricsForFileOperations))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("SERENA FILE OPERATIONS TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All Serena file operation tests passed!")
    else:
        print("\n❌ Some tests failed")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
