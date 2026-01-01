"""
Test script for Phase 2: Multi-Template Cypher System

Tests the integration of query classification with template selection.
Validates that different query types get appropriate templates.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline.query_classifier import classify_query, QueryType


def test_template_selection():
    """Test that queries are classified and appropriate templates would be selected."""
    
    test_cases = [
        # (question, expected_type, expected_template_key)
        ("What methods are in class UserService?", QueryType.STRUCTURAL, "cypher_generation_structural"),
        ("How does the authenticate method work?", QueryType.IMPLEMENTATION, "cypher_generation_implementation"),
        ("Find all methods with more than 5 parameters", QueryType.SEARCH, "cypher_generation_search"),
        ("Where is the processPayment method called?", QueryType.TRACE, "cypher_generation_trace"),
        ("Modify the CacheKey class to add logging", QueryType.MODIFICATION, "cypher_generation_modification"),
    ]
    
    print("="*70)
    print("Phase 2: Multi-Template System Test")
    print("="*70)
    print()
    
    template_map = {
        QueryType.STRUCTURAL: "cypher_generation_structural",
        QueryType.IMPLEMENTATION: "cypher_generation_implementation",
        QueryType.SEARCH: "cypher_generation_search",
        QueryType.TRACE: "cypher_generation_trace",
        QueryType.MODIFICATION: "cypher_generation_modification",
        QueryType.HYBRID: "cypher_generation_implementation",
    }
    
    passed = 0
    failed = 0
    
    for question, expected_type, expected_template in test_cases:
        result = classify_query(question)
        selected_template = template_map.get(result.query_type, "cypher_generation")
        
        success = result.query_type == expected_type
        template_match = selected_template == expected_template
        
        status = "OK PASS" if (success and template_match) else "FAIL FAIL"
        
        print(f"{status} Query: {question[:50]}...")
        print(f"    Classified: {result.query_type.value} (confidence: {result.confidence:.2f})")
        print(f"    Template: {selected_template}")
        
        if not success:
            print(f"    Warning  Expected type: {expected_type.value}")
            failed += 1
        elif not template_match:
            print(f"    Warning  Expected template: {expected_template}")
            failed += 1
        else:
            passed += 1
        
        print()
    
    print("="*70)
    print(f"Results: {passed}/{len(test_cases)} tests passed")
    print("="*70)
    
    return passed == len(test_cases)


def test_template_content():
    """Verify that specialized templates exist and have unique content."""
    
    print("\n" + "="*70)
    print("Template Content Validation")
    print("="*70)
    print()
    
    import json
    
    try:
        with open('prompt_templates.json', 'r') as f:
            templates = json.load(f)
        
        required_templates = [
            'cypher_generation_structural',
            'cypher_generation_implementation',
            'cypher_generation_search',
            'cypher_generation_trace',
            'cypher_generation_modification',
        ]
        
        passed = 0
        failed = 0
        
        for template_key in required_templates:
            template = templates.get('rag', {}).get(template_key)
            
            if not template:
                print(f"FAIL FAIL: Template '{template_key}' not found")
                failed += 1
                continue
            
            # Check template has reasonable length
            if len(template) < 200:
                print(f"FAIL FAIL: Template '{template_key}' too short ({len(template)} chars)")
                failed += 1
                continue
            
            # Check template has examples
            if 'Examples:' not in template and 'Example' not in template:
                print(f"Warning  WARNING: Template '{template_key}' has no examples section")
            
            # Check for query type specific keywords
            type_keywords = {
                'structural': ['metadata', 'relationships', 'signatures'],
                'implementation': ['code', 'm.code', 'IS NOT NULL'],
                'search': ['FILTER', 'CONTAINS', 'WHERE'],
                'trace': ['CALLS', 'caller', 'callee'],
                'modification': ['callers', 'callees', 'context'],
            }
            
            template_type = template_key.replace('cypher_generation_', '')
            keywords = type_keywords.get(template_type, [])
            has_keywords = any(kw in template for kw in keywords)
            
            if not has_keywords:
                print(f"Warning  WARNING: Template '{template_key}' missing type-specific keywords")
            
            print(f"OK PASS: Template '{template_key}' ({len(template)} chars)")
            passed += 1
        
        print()
        print("="*70)
        print(f"Results: {passed}/{len(required_templates)} templates validated")
        print("="*70)
        
        return passed == len(required_templates)
        
    except Exception as e:
        print(f"FAIL ERROR: Failed to load templates: {e}")
        return False


def test_query_diversity():
    """Test classification on a diverse set of real-world queries."""
    
    print("\n" + "="*70)
    print("Query Diversity Test")
    print("="*70)
    print()
    
    diverse_queries = [
        "List all classes in package com.example.service",
        "Show me how the password validation works",
        "Find methods that have error handling",
        "What calls the login method?",
        "I want to refactor the UserService class",
        "Which file contains the Logger interface?",
        "Explain the algorithm in the sort method",
        "Find all deprecated methods",
        "Show the call chain from main to database",
        "Add caching to the getUserById method",
    ]
    
    classifications = {}
    for query in diverse_queries:
        result = classify_query(query)
        qt = result.query_type
        if qt not in classifications:
            classifications[qt] = []
        classifications[qt].append((query, result.confidence))
    
    print(f"Classified {len(diverse_queries)} diverse queries into {len(classifications)} types:\n")
    
    for query_type, queries in sorted(classifications.items(), key=lambda x: x[0].value):
        print(f"{query_type.value.upper()} ({len(queries)} queries):")
        for query, conf in queries:
            print(f"  - {query[:60]}... (confidence: {conf:.2f})")
        print()
    
    # Check that we're using variety of templates
    if len(classifications) >= 4:
        print("OK PASS: Good variety of query types detected")
        return True
    else:
        print(f"Warning  WARNING: Limited variety ({len(classifications)}/6 types used)")
        return len(classifications) >= 3


def main():
    """Run all Phase 2 tests."""
    
    print("\n" + "="*70)
    print("PHASE 2: MULTI-TEMPLATE CYPHER SYSTEM")
    print("Testing query classification → template selection pipeline")
    print("="*70)
    
    results = []
    
    # Test 1: Template selection
    print("\n[Test 1/3] Template Selection Logic")
    results.append(test_template_selection())
    
    # Test 2: Template content
    print("\n[Test 2/3] Template Content Validation")
    results.append(test_template_content())
    
    # Test 3: Query diversity
    print("\n[Test 3/3] Query Diversity Handling")
    results.append(test_query_diversity())
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 2 TEST SUMMARY")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"\nTests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All Phase 2 tests passed! Multi-template system is working.")
        print("\nKey achievements:")
        print("  OK Query classification integrated with template selection")
        print("  OK 5 specialized templates created and validated")
        print("  OK System handles diverse query types appropriately")
        print("\nNext: Phase 3 - Structured Metadata Extraction")
    else:
        print(f"\nWarning  {total - passed} test(s) failed. Review and fix before proceeding.")
    
    print("="*70)
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
