from application import Application
import logging
def main():
    # Configure logging for CLI mode
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    """
    Main entry point to run the RAG application.
    """
    app = Application()
    app.run_cli()
if __name__ == "__main__":
    main()