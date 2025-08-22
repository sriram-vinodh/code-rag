import argparse
import os
import platform
import shutil
import logging
import subprocess
import sys
def run_command(command, shell=False, check=True):
    """Runs a command and handles errors."""
    try:
        subprocess.run(command, shell=shell, check=check)
        return True # noqa
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.error(f"Error running command: {command}\n{e}", exc_info=True)
        return False # noqa
def _run_application_mode(args: argparse.Namespace, remaining_argv: list):
    """Runs the application in the specified mode (web or cli)."""
    if args.mode == "web":
        logging.info("\n--- Starting RAG Web Server ---")
        logging.info("Access the web interface at http://localhost:8000")
        run_command([sys.executable, "-m", "uvicorn", "web.web_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "info"])
    else:
        logging.info("\n--- Starting RAG App (CLI Mode) ---")
        run_command([sys.executable, "main.py"] + remaining_argv)

def main():
    """Main entry point for the runner script."""
    parser = argparse.ArgumentParser(description="Platform-agnostic runner for the RAG application.")
    parser.add_argument("mode", nargs='?', default="cli", choices=["cli", "web"], help="Mode to run: 'cli' or 'web'.")
    args, remaining_argv = parser.parse_known_args()

    # The new run.sh script handles all environment setup.
    # This script now only focuses on launching the application in the correct mode.
    _run_application_mode(args, remaining_argv)

if __name__ == "__main__":
    main()