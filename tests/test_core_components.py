import unittest
import os
import tempfile
import shutil
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

# Import components to test
from rag.chunking.text_chunker import RecursiveTextChunker
from rag.parser.document_parser import DefaultDocumentParser
from rag.loader.filesystem_loader import FileSystemLoader
from rag.flow.rag_pipeline import RAGPipeline
from rag.retriever.cypher_query_helper import CypherQueryHelper

class TestCoreComponents(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for file loader tests
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the temporary directory after tests
        shutil.rmtree(self.test_dir)

    def test_recursive_text_chunker(self):
        """Test RecursiveTextChunker functionality."""
        chunker = RecursiveTextChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a long text that needs to be split into smaller chunks. " * 5
        doc = Document(page_content=text, metadata={"source": "test_doc"})
        
        chunks = chunker.chunk_documents([doc])
        
        self.assertTrue(len(chunks) > 1)
        self.assertEqual(chunks[0].metadata["source"], "test_doc")
        self.assertIn("chunk_index", chunks[0].metadata)
        self.assertTrue(len(chunks[0].page_content) <= 50)

    def test_default_document_parser(self):
        """Test DefaultDocumentParser functionality."""
        parser = DefaultDocumentParser()
        
        # Test plain text parsing
        file_data = {"file_path": "test.txt", "content": "Hello World"}
        doc = parser._process_single_file(file_data)
        self.assertEqual(doc.page_content, "Hello World")
        self.assertEqual(doc.metadata["source"], "test.txt")
        
        # Test Java parsing
        file_data_java = {"file_path": "Test.java", "content": "public class Test {}"}
        doc_java = parser._process_single_file(file_data_java)
        self.assertEqual(doc_java.page_content, "public class Test {}")
        self.assertEqual(doc_java.metadata["language"], "java")

    def test_filesystem_loader(self):
        """Test FileSystemLoader functionality."""
        # Create dummy files
        with open(os.path.join(self.test_dir, "test1.txt"), "w") as f:
            f.write("Content 1")
        with open(os.path.join(self.test_dir, "test2.py"), "w") as f:
            f.write("print('Content 2')")
        with open(os.path.join(self.test_dir, "ignore.md"), "w") as f:
            f.write("Ignore me")

        loader = FileSystemLoader(self.test_dir, allowed_extensions=[".txt", ".py"])
        loaded_files = list(loader.load())
        
        self.assertEqual(len(loaded_files), 2)
        filenames = [os.path.basename(f["file_path"]) for f in loaded_files]
        self.assertIn("test1.txt", filenames)
        self.assertIn("test2.py", filenames)
        self.assertNotIn("ignore.md", filenames)

    @patch("rag.flow.rag_pipeline.PromptProcessor")
    @patch("rag.flow.rag_pipeline.StepExecutor")
    def test_rag_pipeline(self, MockStepExecutor, MockPromptProcessor):
        """Test RAGPipeline orchestration."""
        mock_llm = MagicMock()
        pipeline = RAGPipeline(llm=mock_llm)
        
        # Mock dependencies
        mock_prompt_processor_instance = MockPromptProcessor.return_value
        mock_step_executor_instance = MockStepExecutor.return_value
        
        mock_prompt_processor_instance.get_execution_plan.return_value = ["step1", "step2"]
        mock_step_executor_instance.execute_plan.return_value = "Final Answer"
        
        # Test process_query
        result = pipeline.process_query("Test question")
        
        self.assertEqual(result, "Final Answer")
        mock_prompt_processor_instance.get_execution_plan.assert_called_once_with("Test question")
        mock_step_executor_instance.execute_plan.assert_called_once_with(["step1", "step2"])

    def test_cypher_query_helper(self):
        """Test CypherQueryHelper functionality."""
        # Test extract_identifiers
        identifiers = CypherQueryHelper.extract_identifiers("What methods does User class have?")
        self.assertIn("User", identifiers)
        self.assertIn("class", identifiers)
        
        # Test build_flexible_query for class
        query_class = CypherQueryHelper.build_flexible_query("Show me class User")
        self.assertIn("MATCH (c:Class {name: 'User'})", query_class)
        
        # Test build_flexible_query for method
        query_method = CypherQueryHelper.build_flexible_query("Show me method login")
        self.assertIn("MATCH (m:Method {name: 'login'})", query_method)

    def test_java_code_chunker(self):
        """Test JavaCodeChunker functionality."""
        try:
            from rag.chunking.java_code_chunker import JavaCodeChunker
        except ImportError:
            self.skipTest("javalang not installed")
            
        chunker = JavaCodeChunker()
        java_code = """
        public class TestClass {
            private int x;
            
            public void methodOne() {
                System.out.println("One");
            }
            
            public int methodTwo(int a) {
                return a * 2;
            }
        }
        """
        doc = Document(page_content=java_code, metadata={"source": "TestClass.java"})
        chunks = chunker.chunk_documents([doc])
        
        self.assertEqual(len(chunks), 2)
        method_names = [c.metadata["java_method"] for c in chunks]
        self.assertIn("methodOne", method_names)
        self.assertIn("methodTwo", method_names)
        self.assertEqual(chunks[0].metadata["java_class"], "TestClass")

    def test_graph_metadata_extraction(self):
        """Test GraphMetadata extraction from Neo4j results."""
        from rag.pipeline.graph_metadata import extract_graph_metadata, MethodMetadata

        # Mock Neo4j result rows
        rows = [
            {
                "method": "login",
                "className": "User",
                "filePath": "src/User.java",
                "code": "public void login() { ... }",
                "calledBy": [{"method": "main", "className": "App"}]
            },
            {
                "class": "Order",
                "methods": [{"name": "process", "signature": "void process()"}]
            }
        ]

        metadata = extract_graph_metadata(rows)

        # Check User.login
        user_login_key = "User::login"
        self.assertIn(user_login_key, metadata.methods)
        method = metadata.methods[user_login_key]
        self.assertEqual(method.name, "login")
        self.assertEqual(method.class_name, "User")
        self.assertEqual(len(method.called_by), 1)
        self.assertEqual(method.called_by[0].method_name, "main")

        # Check Order.process
        order_process_key = "Order::process::void process()"
        self.assertIn(order_process_key, metadata.methods)
        method_process = metadata.methods[order_process_key]
        self.assertEqual(method_process.name, "process")
        self.assertEqual(method_process.class_name, "Order")
        
        # Check Class metadata
        self.assertIn("Order", metadata.classes)
        self.assertEqual(len(metadata.classes["Order"].methods), 1)

    def test_query_classifier(self):
        """Test QueryClassifier functionality."""
        from rag.pipeline.query_classifier import QueryClassifier, QueryType

        classifier = QueryClassifier()

        # Test STRUCTURAL
        result = classifier.classify("List all methods in class User")
        self.assertEqual(result.query_type, QueryType.STRUCTURAL)
        self.assertTrue(result.confidence >= 0.8)

        # Test IMPLEMENTATION
        result = classifier.classify("How does the login method work?")
        self.assertEqual(result.query_type, QueryType.IMPLEMENTATION)
        self.assertTrue(result.confidence >= 0.8)

        # Test MODIFICATION
        result = classifier.classify("Refactor the User class")
        self.assertEqual(result.query_type, QueryType.MODIFICATION)
        self.assertTrue(result.confidence >= 0.8)

        # Test SEARCH
        result = classifier.classify("Find all methods with more than 5 parameters")
        self.assertEqual(result.query_type, QueryType.SEARCH)
        self.assertTrue(result.confidence >= 0.8)

        # Test TRACE
        result = classifier.classify("Find callers of login method")
        self.assertEqual(result.query_type, QueryType.TRACE)
        self.assertTrue(result.confidence >= 0.8)

    def test_query_monitor(self):
        """Test QueryMonitor functionality."""
        from rag.pipeline.query_monitor import QueryMonitor, QueryStatus

        # Use a temporary directory for logs
        log_dir = os.path.join(self.test_dir, "logs")
        monitor = QueryMonitor(log_dir=log_dir, enable_logging=True)

        # Track a successful query
        with monitor.track_query("Test question") as tracker:
            tracker.set_classification("structural", 0.9)
            tracker.set_template("template_1", 100)
            tracker.start_generation()
            tracker.set_generated_cypher("MATCH (n) RETURN n")
            tracker.start_execution()
            tracker.set_results([{"n": "node"}], QueryStatus.SUCCESS)

        # Check metrics
        self.assertEqual(len(monitor.recent_queries), 1)
        metrics = monitor.recent_queries[0]
        self.assertEqual(metrics.question, "Test question")
        self.assertEqual(metrics.status, QueryStatus.SUCCESS.value)
        self.assertEqual(metrics.result_count, 1)

        # Check log file creation
        self.assertTrue(os.path.exists(log_dir))
        log_files = os.listdir(log_dir)
        self.assertTrue(len(log_files) > 0)

    def test_step_executor(self):
        """Test StepExecutor functionality."""
        from rag.flow.step_executor import StepExecutor
        from rag.flow.task_execution_plan import TaskExecutionPlan, PlanStep
        from rag.flow.types import ExecutionResult

        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        
        executor = StepExecutor(llm=mock_llm, neo4j_retriever=mock_retriever)
        
        # Mock LLM response for step execution
        mock_llm.predict.return_value = "Step executed successfully"
        
        # Create a simple plan
        step = PlanStep(
            step="Test step",
            complexity="simple",
            needs_context=False
        )
        plan = TaskExecutionPlan(steps=[step], total_steps=1)
        
        # Execute plan
        result = executor.execute_plan(plan)
        
        self.assertIn("Step executed successfully", result)
        mock_llm.predict.assert_called()



if __name__ == "__main__":
    unittest.main()
