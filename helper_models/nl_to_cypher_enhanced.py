import torch
import torch.nn as nn
import logging
import pickle
import os
import re
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, 'rnn_embedding_model.pth')
VOCAB_PATH = os.path.join(MODEL_DIR, 'rnn_vocab.pkl')

class CypherGenerator:
    """Enhanced Cypher query generation with validation and fallbacks"""
    
    COMMON_PATTERNS = {
        # Code Structure Patterns
        "class_info": (
            "MATCH (c:Class {name: '%s'}) "
            "OPTIONAL MATCH (c)-[r]->(related) "
            "RETURN c, collect(r), collect(related) LIMIT 5"
        ),
        "method_calls": (
            "MATCH (m:Method {name: '%s'}) "
            "OPTIONAL MATCH (m)-[r:CALLS]->(called:Method) "
            "RETURN m, collect(r), collect(called) LIMIT 5"
        ),
        "class_methods": (
            "MATCH (c:Class {name: '%s'})-[:HAS_METHOD]->(m:Method) "
            "RETURN c, collect(m) LIMIT 10"
        ),
        # Test Related Patterns
        "test_methods": (
            "MATCH (c:Class)-[:HAS_METHOD]->(m:Method) "
            "WHERE m.code CONTAINS '@Test' AND c.name CONTAINS '%s' "
            "RETURN c.name, collect(m.name)"
        ),
        # Database Schema Patterns
        "table_info": (
            "MATCH (t:Table {name: '%s'}) "
            "OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column) "
            "RETURN t, collect(c)"
        ),
        # CI/CD Pipeline Patterns
        "pipeline_stage": (
            "MATCH (f:File)-[:CONTAINS]->(s:Stage {name: '%s'}) "
            "RETURN f.path, s.code"
        ),
        # Feature Flag Patterns
        "feature_access": (
            "MATCH (u:User)-[:HAS_ACCESS_TO]->(f:FeatureFlag {name: '%s'}) "
            "RETURN u.name"
        )
    }
    
    def __init__(self):
        self.model, self.vocab = self._load_model_and_vocab()
        
    def _load_model_and_vocab(self) -> Tuple[Optional[nn.Module], Optional[Dict[str, Any]]]:
        """Load the trained model and vocabulary with proper error handling"""
        try:
            if not (os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH)):
                logger.warning("Model or vocabulary files not found")
                return None, None
                
            with open(VOCAB_PATH, 'rb') as f:
                vocab = pickle.load(f)
                
            model = torch.load(MODEL_PATH)
            model.eval()
            
            logger.info("✅ Successfully loaded model and vocabulary")
            return model, vocab
            
        except Exception as e:
            logger.error(f"Failed to load model/vocabulary: {e}")
            return None, None
            
    def validate_cypher_query(self, query: str) -> bool:
        """Validate generated Cypher query"""
        if not query:
            return False
            
        # Check for basic Cypher syntax
        required_keywords = ['MATCH', 'RETURN']
        if not all(keyword in query.upper() for keyword in required_keywords):
            return False
            
        # Check for code structure patterns
        code_patterns = ['Class', 'Method', 'CALLS', 'HAS_METHOD']
        if not any(pattern in query for pattern in code_patterns):
            return False
            
        # Check for basic query structure
        try:
            parts = query.upper().split()
            if 'MATCH' not in parts or 'RETURN' not in parts:
                return False
                
            # Validate node/relationship patterns
            if '(' not in query or ')' not in query:
                return False
                
            return True
            
        except Exception:
            return False
            
    def extract_keywords(self, text: str) -> Dict[str, str]:
        """Extract relevant keywords from the input text"""
        keywords = {}
        
        # Look for class names
        class_match = re.search(r'class\s+([A-Za-z_][A-Za-z0-9_]*)', text, re.I)
        if class_match:
            keywords['class'] = class_match.group(1)
            
        # Look for method names
        method_match = re.search(r'method\s+([A-Za-z_][A-Za-z0-9_]*)', text, re.I)
        if method_match:
            keywords['method'] = method_match.group(1)
            
        return keywords
        
    def generate_fallback_query(self, text: str) -> str:
        """Generate a fallback query based on keywords"""
        keywords = self.extract_keywords(text)
        
        if 'class' in keywords:
            return self.COMMON_PATTERNS['class_info'] % keywords['class']
            
        if 'method' in keywords:
            return self.COMMON_PATTERNS['method_calls'] % keywords['method']
            
        # Ultimate fallback
        return "MATCH (c:Class) RETURN c LIMIT 3"
        
    def generate_query(self, text: str) -> Optional[str]:
        """Generate a Cypher query with fallbacks"""
        logger.info(f"Generating Cypher query for: {text}")
        
        # Try model-based generation first
        if self.model and self.vocab:
            try:
                # Model-based generation would go here
                # For now, we'll use the fallback
                pass
            except Exception as e:
                logger.warning(f"Model-based generation failed: {e}")
        
        # Use pattern-based generation
        query = self.generate_fallback_query(text)
        
        # Validate the query
        if self.validate_cypher_query(query):
            logger.info(f"Generated valid query: {query}")
            return query
            
        logger.warning("Failed to generate valid query")
        return None

def generate_cypher(text: str) -> Optional[str]:
    """Main entry point for Cypher query generation"""
    generator = CypherGenerator()
    return generator.generate_query(text)
