#!/bin/bash

# Define the local environment directory
VENV_DIR="venv"

echo "🚀 Starting Setup..."

# 1. Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Error: 'python' command not found. Please ensure Python is installed and added to PATH."
    exit 1
fi

# 2. Create Virtual Environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment in ./$VENV_DIR..."
    python -m venv $VENV_DIR
else
    echo "✅ Virtual environment found."
fi

# 3. Activate Virtual Environment
# Windows Git Bash / Cygwin uses Scripts/activate
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
elif [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "❌ Error: Could not find activation script."
    exit 1
fi

# 4. Install Dependencies
echo "⬇️ Installing dependencies..."
pip install --upgrade pip
pip install streamlit pandas plotly

# 5. Run the App
echo "📈 Starting Streamlit App..."
# Use 'python -m streamlit' to avoid path issues on Windows
python -m streamlit run 2-visualize-data.py
