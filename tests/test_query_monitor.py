"""
Test Query Monitoring System

Validates that instrumentation correctly tracks query execution metrics.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import json
from pathlib import Path
from rag.pipeline.query_monitor import QueryMonitor, QueryTracker, QueryStatus, QueryMetrics


def test_query_tracker_basic():
    """Test basic query tracking"""
    monitor = QueryMonitor(log_dir=tempfile.mkdtemp(), enable_logging=False)
    
    with monitor.track_query("What methods are in UserService?") as tracker:
        tracker.set_classification("STRUCTURAL", 0.95, ["structural pattern"])
        tracker.set_template("cypher_generation_structural", 1670)
        tracker.start_generation()
        tracker.set_generated_cypher("MATCH (c:Class {name: 'UserService'})-[:HAS_METHOD]->(m:Method) RETURN m", iteration=1)
        tracker.start_execution()
        tracker.set_results([{"method": "login"}, {"method": "logout"}], QueryStatus.SUCCESS)
    
    # Verify metrics were recorded
    assert len(monitor.recent_queries) == 1
    metrics = monitor.recent_queries[0]
    
    assert metrics.question == "What methods are in UserService?"
    assert metrics.query_type == "STRUCTURAL"
    assert metrics.confidence == 0.95
    assert metrics.template_key == "cypher_generation_structural"
    assert metrics.status == "success"
    assert metrics.result_count == 2
    assert metrics.iteration_count == 1
    
    print("OK Basic tracking works correctly")


def test_query_tracker_with_error():
    """Test tracking query failures"""
    monitor = QueryMonitor(log_dir=tempfile.mkdtemp(), enable_logging=False)
    
    with monitor.track_query("Invalid query") as tracker:
        tracker.set_classification("SEARCH", 0.6, [])
        tracker.set_template("cypher_generation_search", 1679)
        tracker.start_generation()
        tracker.set_generated_cypher("INVALID CYPHER", iteration=1)
        tracker.set_error(Exception("Syntax error at position 0"), "syntax_error")
    
    assert len(monitor.recent_queries) == 1
    metrics = monitor.recent_queries[0]
    
    assert metrics.status != "success"
    assert metrics.error_message == "Syntax error at position 0"
    assert metrics.error_type == "syntax_error"
    
    print("OK Error tracking works correctly")


def test_failure_rate_calculation():
    """Test failure rate calculation"""
    monitor = QueryMonitor(log_dir=tempfile.mkdtemp(), enable_logging=False)
    
    # Add 10 successful queries
    for i in range(10):
        with monitor.track_query(f"Query {i}") as tracker:
            tracker.set_classification("STRUCTURAL", 0.9, [])
            tracker.set_template("cypher_generation_structural", 1670)
            tracker.set_generated_cypher("MATCH (c:Class) RETURN c", iteration=1)
            tracker.set_results([{"name": "Class1"}], QueryStatus.SUCCESS)
    
    # Add 3 failed queries
    for i in range(3):
        with monitor.track_query(f"Failed query {i}") as tracker:
            tracker.set_classification("SEARCH", 0.7, [])
            tracker.set_template("cypher_generation_search", 1679)
            tracker.set_generated_cypher("BAD QUERY", iteration=1)
            tracker.set_error(Exception("Syntax error"), "syntax_error")
    
    # Check failure rate
    failure_rate = monitor.get_failure_rate(last_n=13)
    expected_rate = 3 / 13  # ~23%
    
    assert abs(failure_rate - expected_rate) < 0.01, f"Expected {expected_rate:.2f}, got {failure_rate:.2f}"
    
    # Should recommend enhancement (>20% failure)
    assert monitor.should_enhance_templates(threshold=0.20, min_samples=10)
    
    print(f"OK Failure rate calculation correct: {failure_rate:.1%}")


def test_failure_patterns():
    """Test failure pattern analysis"""
    monitor = QueryMonitor(log_dir=tempfile.mkdtemp(), enable_logging=False)
    
    # Add various failures
    error_types = ["syntax_error", "syntax_error", "execution_error", "timeout", "syntax_error"]
    
    for i, error_type in enumerate(error_types):
        with monitor.track_query(f"Query {i}") as tracker:
            tracker.set_classification("SEARCH", 0.7, [])
            tracker.set_template("cypher_generation_search", 1679)
            tracker.set_generated_cypher("QUERY", iteration=1)
            tracker.set_error(Exception(f"{error_type} occurred"), error_type)
    
    patterns = monitor.get_failure_patterns(last_n=10)
    
    assert patterns["syntax_error"] == 3
    assert patterns["execution_error"] == 1
    assert patterns["timeout"] == 1
    
    print("OK Failure pattern analysis works correctly")


def test_metrics_by_type():
    """Test filtering metrics by query type"""
    monitor = QueryMonitor(log_dir=tempfile.mkdtemp(), enable_logging=False)
    
    # Add mixed query types
    for query_type in ["STRUCTURAL", "IMPLEMENTATION", "SEARCH", "STRUCTURAL", "TRACE"]:
        with monitor.track_query(f"{query_type} query") as tracker:
            tracker.set_classification(query_type, 0.9, [])
            tracker.set_template(f"cypher_generation_{query_type.lower()}", 1700)
            tracker.set_generated_cypher("MATCH (n) RETURN n", iteration=1)
            tracker.set_results([{"data": "result"}], QueryStatus.SUCCESS)
    
    structural_metrics = monitor.get_metrics_by_type("STRUCTURAL", last_n=10)
    assert len(structural_metrics) == 2
    
    implementation_metrics = monitor.get_metrics_by_type("IMPLEMENTATION", last_n=10)
    assert len(implementation_metrics) == 1
    
    print("OK Filtering by query type works correctly")


def test_report_generation():
    """Test comprehensive report generation"""
    monitor = QueryMonitor(log_dir=tempfile.mkdtemp(), enable_logging=False)
    
    # Add diverse queries
    for i in range(5):
        with monitor.track_query(f"Query {i}") as tracker:
            tracker.set_classification("STRUCTURAL", 0.9, [])
            tracker.set_template("cypher_generation_structural", 1670)
            tracker.start_generation()
            tracker.set_generated_cypher("MATCH (c:Class) RETURN c", iteration=1)
            tracker.start_execution()
            tracker.set_results([{"name": f"Class{i}"}], QueryStatus.SUCCESS)
    
    # Add 2 failures
    for i in range(2):
        with monitor.track_query(f"Failed query {i}") as tracker:
            tracker.set_classification("SEARCH", 0.7, [])
            tracker.set_template("cypher_generation_search", 1679)
            tracker.set_generated_cypher("BAD", iteration=1)
            tracker.set_error(Exception("Error"), "syntax_error")
    
    report = monitor.generate_report(last_n=10)
    
    assert report["total_queries"] == 7
    assert report["success_count"] == 5
    assert report["failure_count"] == 2
    assert abs(report["success_rate"] - 5/7) < 0.01
    assert "STRUCTURAL" in report["query_type_distribution"]
    assert "SEARCH" in report["query_type_distribution"]
    assert "syntax_error" in report["failure_patterns"]
    
    print("OK Report generation works correctly")
    print(f"   Success rate: {report['success_rate']:.1%}")
    print(f"   Query types: {list(report['query_type_distribution'].keys())}")


def test_logging_to_file():
    """Test that metrics are logged to file"""
    temp_dir = tempfile.mkdtemp()
    monitor = QueryMonitor(log_dir=temp_dir, enable_logging=True)
    
    with monitor.track_query("Test query") as tracker:
        tracker.set_classification("STRUCTURAL", 0.95, [])
        tracker.set_template("cypher_generation_structural", 1670)
        tracker.set_generated_cypher("MATCH (c:Class) RETURN c", iteration=1)
        tracker.set_results([{"name": "Class1"}], QueryStatus.SUCCESS)
    
    # Check log file was created
    log_files = list(Path(temp_dir).glob("queries_*.jsonl"))
    assert len(log_files) == 1
    
    # Verify content
    with open(log_files[0], 'r') as f:
        line = f.readline()
        data = json.loads(line)
        assert data["question"] == "Test query"
        assert data["query_type"] == "STRUCTURAL"
        assert data["status"] == "success"
    
    print("OK File logging works correctly")


def test_empty_results_detection():
    """Test automatic detection of empty results"""
    monitor = QueryMonitor(log_dir=tempfile.mkdtemp(), enable_logging=False)
    
    with monitor.track_query("Query returning nothing") as tracker:
        tracker.set_classification("SEARCH", 0.8, [])
        tracker.set_template("cypher_generation_search", 1679)
        tracker.set_generated_cypher("MATCH (c:Class {name: 'NonExistent'}) RETURN c", iteration=1)
        tracker.set_results([], QueryStatus.SUCCESS)  # Empty results
    
    metrics = monitor.recent_queries[0]
    assert metrics.status == "empty_results"
    assert metrics.result_count == 0
    
    print("OK Empty results detection works correctly")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("QUERY MONITORING SYSTEM TESTS")
    print("="*70 + "\n")
    
    test_query_tracker_basic()
    test_query_tracker_with_error()
    test_failure_rate_calculation()
    test_failure_patterns()
    test_metrics_by_type()
    test_report_generation()
    test_logging_to_file()
    test_empty_results_detection()
    
    print("\n" + "="*70)
    print("OK ALL TESTS PASSED")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
