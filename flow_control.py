"""
Automatic Template Enhancement Flow Control

Monitors query performance and automatically suggests or applies template improvements
when failure rates exceed thresholds.

Usage:
    python flow_control.py check           # Check if enhancement needed
    python flow_control.py suggest         # Suggest improvements
    python flow_control.py apply --dry-run # Preview changes
    python flow_control.py apply           # Apply improvements
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from rag.pipeline.query_monitor import get_monitor


class TemplateEnhancer:
    """
    Analyzes query failures and suggests template improvements.
    """
    
    def __init__(self, log_dir: str = "logs/queries"):
        self.log_dir = log_dir
        self.monitor = get_monitor(log_dir=log_dir, enable_logging=False)
        self._load_recent_metrics()
    
    def _load_recent_metrics(self, n: int = 50):
        """Load recent metrics from log files"""
        log_path = Path(self.log_dir)
        if not log_path.exists():
            self.monitor.recent_queries = []
            return
        
        log_files = sorted(log_path.glob("queries_*.jsonl"), reverse=True)
        if not log_files:
            self.monitor.recent_queries = []
            return
        
        from rag.pipeline.query_monitor import QueryMetrics
        
        all_metrics = []
        for log_file in log_files[:5]:  # Last 5 days
            with open(log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        all_metrics.append(QueryMetrics(**data))
            
            if len(all_metrics) >= n:
                break
        
        self.monitor.recent_queries = all_metrics[-n:]
    
    def check_enhancement_needed(self, threshold: float = 0.20, min_samples: int = 10) -> Tuple[bool, Dict]:
        """
        Check if template enhancement is needed.
        
        Returns:
            Tuple of (needs_enhancement, analysis_dict)
        """
        if len(self.monitor.recent_queries) < min_samples:
            return False, {
                "status": "insufficient_data",
                "message": f"Only {len(self.monitor.recent_queries)} queries, need {min_samples}",
            }
        
        failure_rate = self.monitor.get_failure_rate(last_n=20)
        needs_enhancement = failure_rate > threshold
        
        # Analyze by query type
        type_failures = {}
        for metrics in self.monitor.recent_queries[-20:]:
            qtype = metrics.query_type
            if qtype not in type_failures:
                type_failures[qtype] = {"total": 0, "failures": 0}
            
            type_failures[qtype]["total"] += 1
            if metrics.status != "success":
                type_failures[qtype]["failures"] += 1
        
        # Calculate failure rates per type
        type_rates = {}
        for qtype, stats in type_failures.items():
            type_rates[qtype] = {
                "failure_rate": stats["failures"] / stats["total"],
                "total": stats["total"],
                "failures": stats["failures"]
            }
        
        return needs_enhancement, {
            "status": "needs_enhancement" if needs_enhancement else "ok",
            "overall_failure_rate": failure_rate,
            "threshold": threshold,
            "total_queries": len(self.monitor.recent_queries[-20:]),
            "type_failure_rates": type_rates,
            "failure_patterns": self.monitor.get_failure_patterns(last_n=20),
        }
    
    def suggest_improvements(self) -> Dict:
        """
        Analyze failures and suggest specific template improvements.
        
        Returns:
            Dictionary with improvement suggestions
        """
        needs_enhancement, analysis = self.check_enhancement_needed()
        
        if not needs_enhancement:
            return {
                "needs_improvement": False,
                "message": "Templates performing well",
                "analysis": analysis,
            }
        
        suggestions = {
            "needs_improvement": True,
            "overall_failure_rate": analysis["overall_failure_rate"],
            "improvements": []
        }
        
        # Analyze each query type
        for qtype, stats in analysis["type_failure_rates"].items():
            if stats["failure_rate"] > 0.3:  # >30% failure
                urgency = "HIGH"
                action = "Add 3-4 diverse examples immediately"
            elif stats["failure_rate"] > 0.2:  # 20-30% failure
                urgency = "MEDIUM"
                action = "Add 2-3 targeted examples"
            elif stats["failure_rate"] > 0.1:  # 10-20% failure
                urgency = "LOW"
                action = "Add 1-2 examples for edge cases"
            else:
                continue  # No action needed
            
            # Get sample failures for this type
            failures = [m for m in self.monitor.recent_queries 
                       if m.query_type == qtype and m.status != "success"]
            
            suggestions["improvements"].append({
                "query_type": qtype,
                "urgency": urgency,
                "failure_rate": stats["failure_rate"],
                "failures": stats["failures"],
                "total": stats["total"],
                "action": action,
                "sample_failures": [
                    {
                        "question": f.question,
                        "error_type": f.error_type,
                        "generated_cypher": f.generated_cypher[:100] + "..." if f.generated_cypher and len(f.generated_cypher) > 100 else f.generated_cypher
                    }
                    for f in failures[:3]
                ]
            })
        
        # Analyze error patterns
        patterns = analysis["failure_patterns"]
        if "syntax_error" in patterns and patterns["syntax_error"] > 5:
            suggestions["improvements"].append({
                "issue": "High syntax error rate",
                "count": patterns["syntax_error"],
                "action": "Add examples showing correct Cypher syntax patterns",
                "urgency": "HIGH"
            })
        
        if "empty_results" in patterns and patterns["empty_results"] > 5:
            suggestions["improvements"].append({
                "issue": "High empty results rate",
                "count": patterns["empty_results"],
                "action": "Review query logic - may be correct queries but wrong data assumptions",
                "urgency": "MEDIUM"
            })
        
        return suggestions
    
    def generate_new_examples(self, query_type: str, failures: List) -> List[str]:
        """
        Generate new example Q&A pairs based on failures.
        
        This is a template - actual examples should be manually crafted.
        """
        examples = []
        
        # Template for new examples
        for failure in failures[:2]:
            example = f"""
# Generated from failure analysis
Q: {failure['question']}
# Previous attempt failed with: {failure['error_type']}
# Corrected example needed here
"""
            examples.append(example)
        
        return examples
    
    def apply_improvements(self, dry_run: bool = True) -> Dict:
        """
        Apply suggested improvements to templates.
        
        Args:
            dry_run: If True, only show what would be changed
            
        Returns:
            Dictionary with changes made or would be made
        """
        suggestions = self.suggest_improvements()
        
        if not suggestions["needs_improvement"]:
            return {"status": "no_action_needed", "message": "Templates performing well"}
        
        changes = {
            "status": "dry_run" if dry_run else "applied",
            "changes": []
        }
        
        for improvement in suggestions["improvements"]:
            if "query_type" not in improvement:
                continue
            
            qtype = improvement["query_type"]
            change = {
                "query_type": qtype,
                "urgency": improvement["urgency"],
                "failure_rate": f"{improvement['failure_rate']:.1%}",
                "action": improvement["action"],
                "generated_examples": []
            }
            
            # Generate example templates
            if "sample_failures" in improvement:
                new_examples = self.generate_new_examples(qtype, improvement["sample_failures"])
                change["generated_examples"] = new_examples
            
            changes["changes"].append(change)
        
        if not dry_run:
            # In a real implementation, this would modify prompt_templates.json
            # For now, we just return the suggestions
            changes["note"] = "Automatic template modification not implemented. Review suggestions above."
        
        return changes


def cmd_check(args):
    """Check if enhancement is needed"""
    enhancer = TemplateEnhancer()
    needs_enhancement, analysis = enhancer.check_enhancement_needed()
    
    print("\n" + "="*70)
    print("TEMPLATE ENHANCEMENT CHECK")
    print("="*70)
    
    if analysis["status"] == "insufficient_data":
        print(f"\nWarning  {analysis['message']}")
        print("   Run more queries before checking enhancement needs.")
        return
    
    print(f"\nOverall Failure Rate: {analysis['overall_failure_rate']:.1%}")
    print(f"Threshold: {analysis['threshold']:.1%}")
    print(f"Total Queries Analyzed: {analysis['total_queries']}")
    
    if needs_enhancement:
        print("\nAlert ENHANCEMENT RECOMMENDED")
    else:
        print("\nOK Templates performing well")
    
    print("\nPer-Query-Type Analysis:")
    for qtype, stats in sorted(analysis["type_failure_rates"].items()):
        rate = stats["failure_rate"]
        status = "FAIL" if rate > 0.3 else "Warning" if rate > 0.2 else "OK"
        print(f"  {status} {qtype:15s}: {stats['failures']}/{stats['total']} ({rate:.1%})")
    
    if analysis["failure_patterns"]:
        print("\nFailure Patterns:")
        for pattern, count in sorted(analysis["failure_patterns"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {pattern}: {count}")
    
    print("="*70 + "\n")


def cmd_suggest(args):
    """Suggest improvements"""
    enhancer = TemplateEnhancer()
    suggestions = enhancer.suggest_improvements()
    
    print("\n" + "="*70)
    print("TEMPLATE IMPROVEMENT SUGGESTIONS")
    print("="*70)
    
    if not suggestions["needs_improvement"]:
        print(f"\nOK {suggestions['message']}")
        print("="*70 + "\n")
        return
    
    print(f"\nOverall Failure Rate: {suggestions['overall_failure_rate']:.1%}")
    print(f"\nRecommended Improvements ({len(suggestions['improvements'])}):")
    
    for i, improvement in enumerate(suggestions["improvements"], 1):
        print(f"\n{i}. ", end="")
        
        if "query_type" in improvement:
            urgency_icon = "Alert" if improvement["urgency"] == "HIGH" else "Warning" if improvement["urgency"] == "MEDIUM" else "Info"
            print(f"{urgency_icon} {improvement['query_type']} (Urgency: {improvement['urgency']})")
            print(f"   Failure Rate: {improvement['failure_rate']:.1%} ({improvement['failures']}/{improvement['total']})")
            print(f"   Action: {improvement['action']}")
            
            if improvement.get("sample_failures"):
                print(f"\n   Sample Failures:")
                for failure in improvement["sample_failures"]:
                    print(f"   - Q: {failure['question']}")
                    print(f"     Error: {failure['error_type']}")
                    if failure['generated_cypher']:
                        print(f"     Generated: {failure['generated_cypher']}")
        else:
            print(f"Warning  {improvement['issue']} ({improvement['count']} occurrences)")
            print(f"   Action: {improvement['action']}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Review failures above")
    print("2. Manually craft 2-3 new examples for each failing query type")
    print("3. Add examples to appropriate template in prompt_templates.json")
    print("4. Test with: python monitor_cli.py report")
    print("\n" + "="*70 + "\n")


def cmd_apply(args):
    """Apply improvements"""
    enhancer = TemplateEnhancer()
    changes = enhancer.apply_improvements(dry_run=args.dry_run)
    
    print("\n" + "="*70)
    if args.dry_run:
        print("TEMPLATE IMPROVEMENTS (DRY RUN)")
    else:
        print("APPLYING TEMPLATE IMPROVEMENTS")
    print("="*70)
    
    if changes["status"] == "no_action_needed":
        print(f"\nOK {changes['message']}")
        print("="*70 + "\n")
        return
    
    print(f"\nChanges: {len(changes['changes'])}")
    
    for change in changes["changes"]:
        print(f"\n{change['query_type']} ({change['urgency']} urgency)")
        print(f"  Failure Rate: {change['failure_rate']}")
        print(f"  Action: {change['action']}")
        
        if change.get("generated_examples"):
            print(f"\n  Generated Example Templates:")
            for example in change["generated_examples"]:
                print("  " + example.replace("\n", "\n  "))
    
    if "note" in changes:
        print(f"\nWarning  NOTE: {changes['note']}")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Template Enhancement Flow Control")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check if enhancement is needed')
    
    # Suggest command
    suggest_parser = subparsers.add_parser('suggest', help='Suggest improvements')
    
    # Apply command
    apply_parser = subparsers.add_parser('apply', help='Apply improvements')
    apply_parser.add_argument('--dry-run', action='store_true', help='Preview changes without applying')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    commands = {
        'check': cmd_check,
        'suggest': cmd_suggest,
        'apply': cmd_apply,
    }
    
    commands[args.command](args)


if __name__ == '__main__':
    main()
