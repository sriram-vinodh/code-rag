#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to discover available tools in the Neo4j MCP server.
"""

import logging
from mcp_client import MCPNeo4jClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    print("\n" + "="*60)
    print("Discovering Neo4j MCP Server Tools")
    print("="*60 + "\n")
    
    try:
        # Initialize and connect to MCP server
        with MCPNeo4jClient() as client:
            print("✓ Connected to MCP server\n")
            
            # List available tools
            print("Fetching available tools...\n")
            tools = client.list_tools()
            
            if tools:
                print("\n" + "="*60)
                print("Found {} Available Tools:".format(len(tools)))
                print("="*60 + "\n")
                
                for i, tool in enumerate(tools, 1):
                    name = tool.get('name', 'Unknown')
                    description = tool.get('description', 'No description')
                    input_schema = tool.get('inputSchema', {})
                    
                    print("{}. {}".format(i, name))
                    print("   Description: {}".format(description))
                    
                    if input_schema:
                        properties = input_schema.get('properties', {})
                        required = input_schema.get('required', [])
                        
                        if properties:
                            print("   Parameters:")
                            for param_name, param_info in properties.items():
                                param_type = param_info.get('type', 'unknown')
                                param_desc = param_info.get('description', '')
                                is_required = " (required)" if param_name in required else " (optional)"
                                print("     - {}: {}{}".format(param_name, param_type, is_required))
                                if param_desc:
                                    print("       {}".format(param_desc))
                    print()
                
                print("="*60)
                
                # Check for natural language query capability
                print("\n" + "="*60)
                print("Checking for Natural Language Query Capability...")
                print("="*60 + "\n")
                
                nl_tools = [t for t in tools if 'natural' in t.get('name', '').lower() or 
                           'language' in t.get('name', '').lower() or
                           'nl' in t.get('name', '').lower() or
                           'text' in t.get('name', '').lower()]
                
                if nl_tools:
                    print("Found natural language query tools:")
                    for tool in nl_tools:
                        print("  - {}".format(tool.get('name')))
                else:
                    print("No natural language query tools found")
                    print("  Available tools use Cypher directly:")
                    for tool in tools:
                        if 'cypher' in tool.get('name', '').lower():
                            print("  - {}".format(tool.get('name')))
                
            else:
                print("✗ No tools found or failed to retrieve tools")
                
    except Exception as e:
        print("\nError: {}".format(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
