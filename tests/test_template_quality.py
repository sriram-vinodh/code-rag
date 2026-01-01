"""
Test the quality and completeness of Cypher generation templates.

This tests whether the templates have:
1. Sufficient examples for the LLM to learn from
2. Clear instructions
3. Schema information
4. Query guidelines

Critical: LLM performance heavily depends on good examples!
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def analyze_template_quality():
    """Analyze each template for completeness and quality."""
    
    print("="*70)
    print("TEMPLATE QUALITY ANALYSIS")
    print("="*70)
    print()
    
    with open('prompt_templates.json', 'r') as f:
        templates = json.load(f)
    
    template_keys = [
        'cypher_generation_structural',
        'cypher_generation_implementation',
        'cypher_generation_search',
        'cypher_generation_trace',
        'cypher_generation_modification',
    ]
    
    issues = []
    
    for key in template_keys:
        template = templates['rag'].get(key, '')
        
        print(f"\n{'='*70}")
        print(f"Template: {key}")
        print(f"{'='*70}")
        
        # Check 1: Length
        print(f"Length: {len(template)} characters")
        if len(template) < 500:
            issues.append(f"{key}: Too short ({len(template)} chars) - likely insufficient")
        
        # Check 2: Has schema information
        has_schema = 'schema' in template.lower() or 'node' in template.lower()
        print(f"Has Schema Info: {'OK' if has_schema else 'FAIL MISSING'}")
        if not has_schema:
            issues.append(f"{key}: Missing schema information")
        
        # Check 3: Has examples
        example_markers = ['Example', 'Q:', 'A:', 'MATCH']
        has_examples = any(marker in template for marker in example_markers)
        print(f"Has Examples: {'OK' if has_examples else 'FAIL MISSING'}")
        if not has_examples:
            issues.append(f"{key}: Missing examples")
        
        # Check 4: Count example queries
        example_count = template.count('Q:') + template.count('Example:')
        print(f"Example Count: {example_count}")
        if example_count < 2:
            issues.append(f"{key}: Insufficient examples ({example_count}) - recommend 3-5")
        
        # Check 5: Has MATCH statements (actual Cypher)
        match_count = template.count('MATCH')
        print(f"Cypher Examples: {match_count} MATCH statements")
        if match_count < 2:
            issues.append(f"{key}: Missing Cypher query examples")
        
        # Check 6: Type-specific keywords
        type_keywords = {
            'structural': ['relationships', 'signature', 'metadata'],
            'implementation': ['m.code', 'IS NOT NULL', 'code'],
            'search': ['WHERE', 'CONTAINS', 'filter'],
            'trace': ['CALLS', 'caller', 'callee'],
            'modification': ['context', 'callers', 'callees'],
        }
        
        template_type = key.replace('cypher_generation_', '')
        keywords = type_keywords.get(template_type, [])
        found_keywords = [kw for kw in keywords if kw in template]
        print(f"Type-Specific Keywords: {len(found_keywords)}/{len(keywords)} found")
        if len(found_keywords) < len(keywords) // 2:
            issues.append(f"{key}: Missing type-specific guidance")
        
        # Show a preview
        print(f"\nPreview (first 300 chars):")
        print(template[:300] + "...")
    
    print("\n" + "="*70)
    print("ISSUES SUMMARY")
    print("="*70)
    
    if issues:
        print(f"\nFAIL Found {len(issues)} issues:\n")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\nOK All templates pass quality checks!")
        return True


def test_template_with_mock_llm():
    """Test what the LLM would see for different query types."""
    
    print("\n" + "="*70)
    print("MOCK LLM TEMPLATE TEST")
    print("Testing what the LLM actually receives for each query type")
    print("="*70)
    
    with open('prompt_templates.json', 'r') as f:
        templates = json.load(f)
    
    test_queries = [
        ("What methods are in class UserService?", "cypher_generation_structural"),
        ("How does the authenticate method work?", "cypher_generation_implementation"),
        ("Find all methods with error handling", "cypher_generation_search"),
        ("Where is processPayment called?", "cypher_generation_trace"),
        ("Modify CacheKey class", "cypher_generation_modification"),
    ]
    
    for question, template_key in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {question}")
        print(f"Template: {template_key}")
        print(f"{'='*70}")
        
        template = templates['rag'].get(template_key, '')
        if not template:
            print("FAIL Template not found!")
            continue
        
        # Format the template as the LLM would see it
        # Use replace() instead of format() since templates have {} in Cypher examples
        prompt = template.replace("{question}", question)
        
        print(f"\nPrompt Length: {len(prompt)} characters")
        print(f"Prompt Token Estimate: ~{len(prompt) // 4} tokens")
        
        # Count examples in the formatted prompt
        example_count = prompt.count('Q:') + prompt.count('Example')
        cypher_count = prompt.count('MATCH')
        
        print(f"Examples in prompt: {example_count}")
        print(f"Cypher examples: {cypher_count}")
        
        # Show what LLM sees (first 500 chars)
        print(f"\nLLM sees (preview):")
        print("-" * 70)
        print(prompt[:500])
        print("...")
        print(prompt[-200:])
        print("-" * 70)
        
        # Quality assessment
        if len(prompt) < 500:
            print("Warning  WARNING: Prompt is very short - LLM may struggle")
        elif len(prompt) > 8000:
            print("Warning  WARNING: Prompt is very long - may hit token limits")
        
        if cypher_count < 2:
            print("Warning  WARNING: Few Cypher examples - LLM may generate invalid syntax")
        elif cypher_count >= 3:
            print("OK Good: Sufficient Cypher examples for LLM to learn from")


def compare_with_generic_template():
    """Compare specialized templates with the generic fallback."""
    
    print("\n" + "="*70)
    print("COMPARISON: Specialized vs Generic Template")
    print("="*70)
    
    with open('prompt_templates.json', 'r') as f:
        templates = json.load(f)
    
    generic = templates['rag'].get('cypher_generation', '')
    specialized_keys = [
        'cypher_generation_structural',
        'cypher_generation_implementation',
        'cypher_generation_search',
    ]
    
    print(f"\nGeneric template length: {len(generic)} chars")
    print(f"Generic MATCH examples: {generic.count('MATCH')}")
    
    print("\nSpecialized templates:")
    for key in specialized_keys:
        template = templates['rag'].get(key, '')
        match_count = template.count('MATCH')
        print(f"  {key}: {len(template)} chars, {match_count} MATCH examples")
    
    # Key insight
    print("\n" + "-"*70)
    print("INSIGHT: Model Dependency")
    print("-"*70)
    print("""
The effectiveness of these templates heavily depends on the LLM's training:

1. GPT-4/Claude (Strong):
   - Can generate Cypher with minimal examples (2-3 examples)
   - Understands graph concepts from training
   - Can infer patterns from schema
   
2. GPT-3.5/Smaller models (Moderate):
   - Needs 4-5 clear examples per query type
   - May struggle with complex graph traversals
   - Benefits from explicit guidelines
   
3. codellama/local models (Weak):
   - Requires 5-7 examples minimum
   - Needs very explicit instructions
   - May produce syntactically correct but logically wrong queries
   
RECOMMENDATION: Each template should have 5-7 diverse examples.
    """)


def main():
    """Run all template quality tests."""
    
    print("\n" + "="*70)
    print("PHASE 2 TEMPLATE QUALITY VALIDATION")
    print("Assessing if templates are sufficient for LLM Cypher generation")
    print("="*70)
    
    # Test 1: Analyze template completeness
    print("\n[Test 1/3] Template Completeness Analysis")
    quality_ok = analyze_template_quality()
    
    # Test 2: Mock LLM test
    print("\n[Test 2/3] Mock LLM Template Processing")
    test_template_with_mock_llm()
    
    # Test 3: Compare with generic
    print("\n[Test 3/3] Specialized vs Generic Comparison")
    compare_with_generic_template()
    
    # Final recommendation
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if not quality_ok:
        print("""
FAIL Templates need improvement before production use.

Action Items:
1. Add 2-3 more examples to each template
2. Ensure each example covers different patterns
3. Add common mistakes section
4. Include edge cases in examples

Risk: Current templates may work with GPT-4 but fail with smaller models.
        """)
    else:
        print("""
OK Templates meet minimum quality standards.

However, for production robustness:
1. Consider adding more examples (current: 3-4, recommended: 5-7)
2. Test with your target LLM (codellama)
3. Monitor generated query quality
4. Add feedback loop to improve templates

Current templates should work with:
- GPT-4/Claude: Excellent
- GPT-3.5: Good
- codellama/local: May need tuning
        """)
    
    print("="*70)
    
    return quality_ok


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
