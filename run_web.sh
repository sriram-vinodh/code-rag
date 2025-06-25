#!/bin/bash

# This script sets up the environment and runs the web server for the RAG application.
# It should be run from the root of the project directory (e.g., ~/code/code-rag).

set -e # Exit immediately if a command exits with a non-zero status.

echo "--- Setting up RAG App Environment ---"

# Find the directory where the script is located to ensure paths are correct
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Installing dependencies from requirements.txt..."
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "--- Starting RAG Web Server ---"
echo "Access the web interface at http://localhost:8000"

# Run the web server application as a module
python -m web.web_server