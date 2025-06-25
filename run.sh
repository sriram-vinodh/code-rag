#!/bin/bash

# This script sets up the environment and runs the RAG application.
# It should be run from the root of the project directory (e.g., ~/code/code-rag).
# It forwards all command-line arguments to the main.py script.

set -e # Exit immediately if a command exits with a non-zero status.

echo "--- Setting up RAG App Environment ---"

# Find the directory where the script is located to ensure paths are correct
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Installing dependencies from requirements.txt..."
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "--- Starting RAG App ---"

# Run the main application, passing along any arguments
python "$SCRIPT_DIR/main.py" "$@"