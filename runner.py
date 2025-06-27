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

def install_requirements():
    """Installs Python packages from requirements.txt."""
    logging.info("--- Installing/Verifying Python Dependencies ---")
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def check_and_install_joern():
    """Checks for joern and guides the user through installation if needed."""
    if not shutil.which("joern"):
        logging.warning("\n--- Optional Dependency Missing: 'joern' ---")
        logging.warning("The 'joern' command was not found in your PATH.")
        logging.warning("The 'cpg' chunking strategy will be unavailable unless joern is installed.")
        logging.warning("See https://joern.io for installation instructions.")
        return False
    return True

def check_joern_dependencies():
    """Checks for joern's dependencies, like coreutils on macOS."""
    if not shutil.which("joern"):
        return

    if platform.system() == "Darwin":  # macOS
        if not shutil.which("greadlink"):
            logging.info("\n--- 'joern' Dependency Missing on macOS ---")
            logging.info("The 'joern' tool requires 'greadlink' from the 'coreutils' package.")
            if shutil.which("brew"):
                response = input("Homebrew is detected. Would you like to install 'coreutils' now? (y/n) ").lower()
                if response == 'y':
                    logging.info("Installing coreutils via Homebrew...")
                    run_command(["brew", "install", "coreutils"])
                    logging.info("'coreutils' installed. Please re-run this script in a new terminal.")
                    sys.exit(0)
            else:
                logging.warning("Please install 'coreutils' to proceed. Recommended command: 'brew install coreutils'")
            logging.info("Skipping 'coreutils' installation. The CPG chunker may fail.")

def _perform_initial_setup():
    """Performs initial setup steps like installing requirements and checking joern."""
    logging.info("--- Setting up RAG App Environment ---")
    install_requirements()
    if check_and_install_joern():
        check_joern_dependencies()

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

    _perform_initial_setup()
    _run_application_mode(args, remaining_argv)

if __name__ == "__main__":
    main()