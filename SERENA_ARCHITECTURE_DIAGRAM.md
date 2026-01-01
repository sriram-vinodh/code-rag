# Serena Integration Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            USER REQUEST                                  │
│  "Add rate limiting to the authentication service and update all calls" │
└─────────────────────────────────────────┬───────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         PROMPT PROCESSOR                                 │
│  Analyzes request → Determines complexity → Creates execution plan       │
└─────────────────────────────────────────┬───────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      TASK EXECUTION PLANNER (LLM)                        │
│                                                                           │
│  Generates: {                                                            │
│    "total_steps": 3,                                                     │
│    "steps": [                                                            │
│      {                                                                   │
│        "step": "Analyze AuthService implementation",                     │
│        "action_type": "analyze",            ← NEW                        │
│        "knowledge_source": "graph_database",                             │
│        "requires_code_edit": false          ← NEW                        │
│      },                                                                  │
│      {                                                                   │
│        "step": "Add rate limiting to authenticate method",               │
│        "action_type": "modify",             ← NEW                        │
│        "knowledge_source": "serena_ide",    ← NEW                        │
│        "requires_code_edit": true,          ← NEW                        │
│        "target_symbols": ["AuthService/authenticate"],  ← NEW            │
│        "edit_strategy": "symbolic"          ← NEW                        │
│      },                                                                  │
│      {                                                                   │
│        "step": "Update all callers",                                     │
│        "action_type": "modify",             ← NEW                        │
│        "knowledge_source": "hybrid",        ← NEW (Graph + Serena)       │
│        "requires_code_edit": true,          ← NEW                        │
│        "edit_strategy": "file_based"        ← NEW                        │
│      }                                                                   │
│    ]                                                                     │
│  }                                                                       │
└─────────────────────────────────────────┬───────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         STEP EXECUTOR                                    │
│                                                                           │
│  For each step:                                                          │
│    1. Check action_type and requires_code_edit                           │
│    2. Route to appropriate knowledge source                              │
│    3. Execute and track governance metadata                              │
└────┬──────────────────────────────────┬──────────────────────────────┬──┘
     │                                  │                              │
     │ knowledge_source:                │ knowledge_source:            │ knowledge_source:
     │ "graph_database"                 │ "serena_ide"                 │ "hybrid"
     │                                  │                              │
     ▼                                  ▼                              ▼
┌─────────────────┐          ┌──────────────────────┐      ┌─────────────────────┐
│  GRAPH QUERY    │          │   SERENA IDE AGENT   │      │  HYBRID EXECUTOR    │
│   EXECUTOR      │          │                      │      │                     │
│                 │          │  24 Semantic Tools:  │      │  1. Graph Analysis  │
│  • Neo4j MCP   │          │                      │      │  2. Serena Editing  │
│  • Cypher Gen  │          │  Read:               │      │                     │
│  • Validation  │          │  - find_symbol       │      │  Combines:          │
│  • Execution   │          │  - get_symbols_      │      │  • Relationships    │
│                 │          │    overview          │      │  • Actual code      │
│  Returns:       │          │  - find_referencing_ │      │                     │
│  • Classes      │          │    symbols           │      │  Then routes to     │
│  • Methods      │          │  - search_for_       │      │  Serena for edits   │
│  • Relationships│          │    pattern           │      │                     │
│  • Call graphs  │          │                      │      └─────────────────────┘
│  • Code snippets│          │  Edit:               │
│                 │          │  - replace_symbol_   │
└─────────────────┘          │    body              │
                             │  - insert_before_    │
                             │    symbol            │
                             │  - insert_after_     │
                             │    symbol            │
                             │  - replace_content   │
                             │    (regex)           │
                             │  - rename_symbol     │
                             │                      │
                             │  Memory:             │
                             │  - read_memory       │
                             │  - write_memory      │
                             └──────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      GOVERNANCE & AUDIT LAYER                            │
│                                                                           │
│  Tracks for each code modification:                                      │
│  • Timestamp                                                             │
│  • Action type (MODIFY, REFACTOR, etc.)                                  │
│  • Edit strategy used                                                    │
│  • Target symbols                                                        │
│  • Files modified                                                        │
│  • Tool invoked                                                          │
│  • Success/failure                                                       │
│  • Rollback capability                                                   │
│                                                                           │
│  Example:                                                                │
│  {                                                                       │
│    "timestamp": "2025-12-15T10:30:00Z",                                  │
│    "step_id": "step_2_of_3",                                             │
│    "action_type": "modify",                                              │
│    "edit_strategy": "symbolic",                                          │
│    "target_symbols": ["AuthService/authenticate"],                       │
│    "files_modified": ["src/auth/service.py"],                            │
│    "tool_used": "serena:replace_symbol_body",                            │
│    "success": true                                                       │
│  }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘


DECISION FLOW: When to use which knowledge source?
═══════════════════════════════════════════════════

┌──────────────────┐
│  Step Analysis   │
└────────┬─────────┘
         │
         ▼
    ┌─────────────────┐
    │ action_type =?  │
    └────┬────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────┐
│ READ / │ │ MODIFY / │
│ANALYZE │ │ REFACTOR │
└───┬────┘ └────┬─────┘
    │           │
    │           ▼
    │      ┌──────────────────┐
    │      │requires_code_edit│
    │      │     = true       │
    │      └────┬─────────────┘
    │           │
    │      ┌────┴────┐
    │      │         │
    ▼      ▼         ▼
┌──────────────┐ ┌─────────────┐
│ graph_       │ │ serena_ide  │
│ database     │ │ OR          │
│              │ │ hybrid      │
│ Best for:    │ │             │
│ • Structure  │ │ Best for:   │
│ • Relations  │ │ • Edits     │
│ • Call graphs│ │ • Refactor  │
│ • Analysis   │ │ • Renames   │
└──────────────┘ └─────────────┘


DATA FLOW EXAMPLE: "Add rate limiting to authenticate method"
═══════════════════════════════════════════════════════════════

Step 1: ANALYZE (graph_database)
┌─────────────────────────────────────┐
│ Query: "Show AuthService methods"   │
│ Tool: MCP Neo4j Cypher              │
│ Result:                             │
│   - authenticate(user, pass)        │
│   - validateToken(token)            │
│   - refreshToken(oldToken)          │
│   + Call graph showing usage        │
└─────────────────────────────────────┘
         │
         ▼
Step 2: MODIFY (serena_ide)
┌─────────────────────────────────────┐
│ Action: Replace authenticate method │
│ Tool: serena:replace_symbol_body    │
│ Input:                              │
│   target: "AuthService/authenticate"│
│   new_body: [full implementation]   │
│ Result:                             │
│   ✓ File modified                   │
│   ✓ Governance logged               │
└─────────────────────────────────────┘
         │
         ▼
Step 3: MODIFY (hybrid)
┌─────────────────────────────────────┐
│ Sub-step 3a: Graph query            │
│   Find all calls to authenticate()  │
│   Result: 12 call sites found       │
│                                     │
│ Sub-step 3b: Serena batch edit      │
│   Tool: serena:replace_content      │
│   Pattern: "auth.authenticate("     │
│   Replacement: "auth.authenticate_  │
│                with_rate_limit("    │
│   Result: 12 files modified         │
│   ✓ Governance logged for each      │
└─────────────────────────────────────┘


COMPARISON: Before vs After Integration
════════════════════════════════════════

BEFORE (Current System):
┌─────────────────────────────────────┐
│ User: "Add logging to saveUser"     │
│                                     │
│ System Response:                    │
│ "Here's what you should change:     │
│                                     │
│  1. Import logging module           │
│  2. Add logger.info() after save    │
│  3. Add logger.error() in catch     │
│                                     │
│  Example code:                      │
│  def saveUser(self, user):          │
│    try:                             │
│      self.db.save(user)             │
│      logger.info(...)  ← Add this   │
│    except:                          │
│      logger.error(...) ← Add this"  │
│                                     │
│ USER MUST MANUALLY EDIT CODE ❌     │
└─────────────────────────────────────┘

AFTER (With Serena):
┌─────────────────────────────────────┐
│ User: "Add logging to saveUser"     │
│                                     │
│ System Response:                    │
│ "✓ Modified UserService/saveUser    │
│ ✓ Added logger.info() after save    │
│ ✓ Added logger.error() in exception │
│ ✓ Files modified: user_service.py   │
│                                     │
│ [Show diff of changes]              │
│                                     │
│ CODE AUTOMATICALLY MODIFIED ✅      │
└─────────────────────────────────────┘


KEY BENEFITS
════════════

1. End-to-end automation
   Graph analysis → Serena execution → Done!

2. Precision
   Symbol-level operations prevent errors

3. Safety
   Governance tracking + rollback capability

4. Efficiency
   Serena's symbolic reading reduces token usage

5. Scalability
   Batch operations across multiple files
```
