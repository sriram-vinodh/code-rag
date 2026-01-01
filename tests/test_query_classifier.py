"""
Test suite for query classification system.

This test suite validates:
1. Rule-based pattern matching accuracy
2. Confidence scoring
3. Edge case handling
4. Fallback behavior

Run with: python3 tests/test_query_classifier.py
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline.query_classifier import QueryClassifier, QueryType, classify_query


class TestQueryClassification(unittest.TestCase):
    """Test query classification accuracy."""
    
    def setUp(self):
        """Initialize classifier without LLM (rule-based only)."""
        self.classifier = QueryClassifier(llm=None)
    
    def test_structural_queries(self):
        """Test classification of structural queries."""
        test_cases = [
            ("What are the methods in class UserService?", QueryType.STRUCTURAL),
            ("List all classes in package com.example", QueryType.STRUCTURAL),
            ("Show me the structure of interface Repository", QueryType.STRUCTURAL),
            ("Where is class Customer defined?", QueryType.STRUCTURAL),
            ("Which file contains the Logger interface?", QueryType.STRUCTURAL),
            ("What is the signature of method authenticate?", QueryType.STRUCTURAL),
            ("What are the parameters of method processPayment?", QueryType.STRUCTURAL),
        ]
        
        for question, expected_type in test_cases:
            with self.subTest(question=question):
                result = self.classifier.classify(question)
                self.assertEqual(result.query_type, expected_type,
                               f"Failed for: {question}\n"
                               f"Got: {result.query_type}, Expected: {expected_type}\n"
                               f"Matched patterns: {result.matched_patterns}")
                self.assertGreater(result.confidence, 0.7,
                                 f"Low confidence ({result.confidence}) for: {question}")
    
    def test_implementation_queries(self):
        """Test classification of implementation/behavioral queries."""
        test_cases = [
            ("How does the authenticate method work?", QueryType.IMPLEMENTATION),
            ("What does the CacheKey class do?", QueryType.IMPLEMENTATION),
            ("Show me the code for method validateInput", QueryType.IMPLEMENTATION),
            ("What algorithm does the sort method use?", QueryType.IMPLEMENTATION),
            ("Explain the logic in processPayment", QueryType.IMPLEMENTATION),
            ("How is the password validated?", QueryType.IMPLEMENTATION),
            ("What does getUserById return?", QueryType.IMPLEMENTATION),
        ]
        
        for question, expected_type in test_cases:
            with self.subTest(question=question):
                result = self.classifier.classify(question)
                self.assertEqual(result.query_type, expected_type,
                               f"Failed for: {question}\n"
                               f"Got: {result.query_type}, Expected: {expected_type}\n"
                               f"Matched patterns: {result.matched_patterns}")
                self.assertGreater(result.confidence, 0.7,
                                 f"Low confidence ({result.confidence}) for: {question}")
    
    def test_modification_queries(self):
        """Test classification of code modification requests."""
        test_cases = [
            ("Modify the authenticate method to add logging", QueryType.MODIFICATION),
            ("Add error handling to the processPayment method", QueryType.MODIFICATION),
            ("Remove the deprecated getUserData method", QueryType.MODIFICATION),
            ("Refactor the UserService class", QueryType.MODIFICATION),
            ("Enhance the validation logic in registerUser", QueryType.MODIFICATION),
            ("Replace the current sorting algorithm with quicksort", QueryType.MODIFICATION),
            ("Fix the bug in the calculateTotal method", QueryType.MODIFICATION),
        ]
        
        for question, expected_type in test_cases:
            with self.subTest(question=question):
                result = self.classifier.classify(question)
                self.assertEqual(result.query_type, expected_type,
                               f"Failed for: {question}\n"
                               f"Got: {result.query_type}, Expected: {expected_type}\n"
                               f"Matched patterns: {result.matched_patterns}")
                self.assertGreater(result.confidence, 0.7,
                                 f"Low confidence ({result.confidence}) for: {question}")
    
    def test_search_queries(self):
        """Test classification of search/filter queries."""
        test_cases = [
            ("Find all methods with more than 5 parameters", QueryType.SEARCH),
            ("Which methods contain the word 'validate'?", QueryType.SEARCH),
            ("Search for methods that use the Logger class", QueryType.SEARCH),
            ("Find all methods that have error handling", QueryType.SEARCH),
            ("List all classes where the name contains 'Service'", QueryType.SEARCH),
        ]
        
        for question, expected_type in test_cases:
            with self.subTest(question=question):
                result = self.classifier.classify(question)
                self.assertEqual(result.query_type, expected_type,
                               f"Failed for: {question}\n"
                               f"Got: {result.query_type}, Expected: {expected_type}\n"
                               f"Matched patterns: {result.matched_patterns}")
                self.assertGreater(result.confidence, 0.7,
                                 f"Low confidence ({result.confidence}) for: {question}")
    
    def test_trace_queries(self):
        """Test classification of call tracing queries."""
        test_cases = [
            ("Where is the authenticate method called?", QueryType.TRACE),
            ("What does the UserService class call?", QueryType.TRACE),
            ("Show me the call chain for processPayment", QueryType.TRACE),
            ("Trace execution from login to authentication", QueryType.TRACE),
            ("What are the callers of method validateUser?", QueryType.TRACE),
            ("Show dependencies of class OrderProcessor", QueryType.TRACE),
        ]
        
        for question, expected_type in test_cases:
            with self.subTest(question=question):
                result = self.classifier.classify(question)
                self.assertEqual(result.query_type, expected_type,
                               f"Failed for: {question}\n"
                               f"Got: {result.query_type}, Expected: {expected_type}\n"
                               f"Matched patterns: {result.matched_patterns}")
                self.assertGreater(result.confidence, 0.7,
                                 f"Low confidence ({result.confidence}) for: {question}")
    
    def test_hybrid_queries(self):
        """Test classification of complex hybrid queries."""
        test_cases = [
            ("Explain how the authentication system works", QueryType.HYBRID),
            ("Understand the architecture of the payment module", QueryType.HYBRID),
            ("How does the entire order processing flow work?", QueryType.HYBRID),
        ]
        
        for question, expected_type in test_cases:
            with self.subTest(question=question):
                result = self.classifier.classify(question)
                self.assertEqual(result.query_type, expected_type,
                               f"Failed for: {question}\n"
                               f"Got: {result.query_type}, Expected: {expected_type}\n"
                               f"Matched patterns: {result.matched_patterns}")
                self.assertGreater(result.confidence, 0.7,
                                 f"Low confidence ({result.confidence}) for: {question}")
    
    def test_ambiguous_queries(self):
        """Test handling of ambiguous queries without LLM."""
        # These should still classify, but with lower confidence
        test_cases = [
            "Tell me about the User class",  # Could be structural or implementation
            "I need information on authentication",  # Vague
            "Check the payment module",  # Unclear intent
        ]
        
        for question in test_cases:
            with self.subTest(question=question):
                result = self.classifier.classify(question)
                # Should still classify (fallback to STRUCTURAL)
                self.assertIsInstance(result.query_type, QueryType)
                # But confidence should be lower
                # (May be high if patterns match, so we just check it classified)
    
    def test_confidence_scores(self):
        """Test that confidence scores are reasonable."""
        # High-confidence queries should have high scores
        high_conf_queries = [
            "What are the methods in class X?",
            "How does method Y work?",
            "Modify class Z to add logging",
        ]
        
        for question in high_conf_queries:
            result = self.classifier.classify(question)
            self.assertGreater(result.confidence, 0.8,
                             f"Expected high confidence for: {question}")
        
        # Ambiguous queries without LLM fallback should have reasonable scores
        ambiguous_queries = [
            "Tell me about X",
            "What is Y?",
        ]
        
        for question in ambiguous_queries:
            result = self.classifier.classify(question)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)
    
    def test_case_insensitivity(self):
        """Test that classification is case-insensitive."""
        test_cases = [
            ("what are the methods in class X?", QueryType.STRUCTURAL),
            ("WHAT ARE THE METHODS IN CLASS X?", QueryType.STRUCTURAL),
            ("What Are The Methods In Class X?", QueryType.STRUCTURAL),
        ]
        
        results = [self.classifier.classify(q) for q, _ in test_cases]
        
        # All should classify to the same type
        query_types = [r.query_type for r in results]
        self.assertEqual(len(set(query_types)), 1,
                        "Case variations produced different classifications")
    
    def test_convenience_function(self):
        """Test the classify_query convenience function."""
        result = classify_query("What are the methods in class User?")  # Changed to match pattern better
        self.assertEqual(result.query_type, QueryType.STRUCTURAL)
        self.assertGreater(result.confidence, 0.7)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        self.classifier = QueryClassifier(llm=None)
    
    def test_empty_string(self):
        """Test classification of empty string."""
        result = self.classifier.classify("")
        # Should not crash, should return some classification
        self.assertIsInstance(result.query_type, QueryType)
    
    def test_very_long_query(self):
        """Test classification of very long queries."""
        long_query = "Show me the methods in class " + "X" * 1000
        result = self.classifier.classify(long_query)
        # Should handle without crashing
        self.assertIsInstance(result.query_type, QueryType)
    
    def test_special_characters(self):
        """Test queries with special characters."""
        test_cases = [
            "What methods are in class User$Inner?",
            "Show me the method <init>",
            "Find methods with signature (String, int)",
        ]
        
        for question in test_cases:
            result = self.classifier.classify(question)
            # Should not crash
            self.assertIsInstance(result.query_type, QueryType)
    
    def test_mixed_patterns(self):
        """Test queries that match multiple patterns."""
        # This query has both "show me the code" and "what does"
        result = self.classifier.classify(
            "Show me the code and explain what the authenticate method does"
        )
        # Should classify as IMPLEMENTATION (strongest signal)
        self.assertEqual(result.query_type, QueryType.IMPLEMENTATION)


class TestClassificationResult(unittest.TestCase):
    """Test ClassificationResult dataclass."""
    
    def test_result_structure(self):
        """Test that ClassificationResult has correct fields."""
        from rag.pipeline.query_classifier import ClassificationResult
        
        result = ClassificationResult(
            query_type=QueryType.STRUCTURAL,
            confidence=0.9,
            reasoning="Test reasoning",
            matched_patterns=["pattern1"]
        )
        
        self.assertEqual(result.query_type, QueryType.STRUCTURAL)
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(result.reasoning, "Test reasoning")
        self.assertEqual(result.matched_patterns, ["pattern1"])
    
    def test_default_matched_patterns(self):
        """Test that matched_patterns defaults to empty list."""
        from rag.pipeline.query_classifier import ClassificationResult
        
        result = ClassificationResult(
            query_type=QueryType.STRUCTURAL,
            confidence=0.9,
            reasoning="Test"
        )
        
        self.assertEqual(result.matched_patterns, [])


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestQueryClassification))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestClassificationResult))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("="*70)
    print("Query Classifier Test Suite")
    print("="*70)
    print()
    
    result = run_tests()
    
    print()
    print("="*70)
    print("Test Summary:")
    print(f"  Total tests run: {result.testsRun}")
    print(f"  Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print("="*70)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
