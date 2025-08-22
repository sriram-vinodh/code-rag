#!/bin/bash
# This script sets up the environment and runs the RAG application.

# Exit immediately if a command exits with a non-zero status.
set -e

VENV_DIR="venv"

# --- Cleanup Function ---
# Removes stale directories for a clean start.
clean_stale_files() {
    echo "--- Cleaning up stale files and directories ---"
    rm -rf "$VENV_DIR"
    rm -rf ./chroma_db
    rm -rf ./chroma_cpg_db
    rm -rf ./vendor
    echo "Cleanup complete."
}

# --- Main Logic ---

# Check for a 'clean' argument
if [ "$1" == "clean" ]; then
    clean_stale_files
    # If only 'clean' was passed, exit. Otherwise, continue with setup.
    if [ $# -eq 1 ]; then
        exit 0
    fi
    # Shift the arguments so the script can proceed with other commands like 'web' or 'cli'
    shift
fi


# 1. Set up Virtual Environment
if [ ! -d "$VENV_DIR" ]; then
    echo "--- Creating Python virtual environment in '$VENV_DIR' ---"
    python3 -m venv "$VENV_DIR"
fi

# 2. Activate Virtual Environment and Install Dependencies
echo "--- Activating virtual environment and installing dependencies ---"
source "$VENV_DIR/bin/activate"
pip install -r requirements.txt

# 3. Run the application using the new package entry point
echo "--- Starting the application ---"
python -m rag "$@"