#!/bin/bash

# Enterprise Phishing Detection ML Pipeline Script
# This script runs the complete machine learning pipeline end-to-end

set -e  # Exit on error

echo "================================================"
echo "Phishing Detection - Machine Learning Pipeline Execution"
echo "================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

PYTHON_CMD_INIT="python3"
echo "Python version: $($PYTHON_CMD_INIT --version)"
echo ""

# Check for and activate virtual environment
VENV_DIR=""
if [ -d ".venv" ]; then
    VENV_DIR=".venv"
elif [ -d "venv" ]; then
    VENV_DIR="venv"
fi

if [ -n "$VENV_DIR" ]; then
    echo "Found virtual environment: $VENV_DIR"
    source "$VENV_DIR/bin/activate"
    echo "Virtual environment activated."
    echo "Python: $(which python)"
    echo ""
else
    echo "⚠️  Warning: No virtual environment found (.venv or venv)."
    echo "   Consider creating one with: python3 -m venv .venv"
    echo ""
fi

# Determine which Python and pip to use
if [ -n "$VENV_DIR" ]; then
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
fi

# Check and install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Checking critical dependencies..."
    
    # Check if pip is available
    if ! command -v $PIP_CMD &> /dev/null; then
        echo "⚠️  Warning: pip not found. Cannot install dependencies."
    else
        # Check for critical dependencies
        MISSING_CRITICAL=false
        
        # Check numpy
        if ! $PYTHON_CMD -c "import numpy" 2>/dev/null; then
            echo "   • numpy: missing"
            MISSING_CRITICAL=true
        fi
        
        # Check pandas
        if ! $PYTHON_CMD -c "import pandas" 2>/dev/null; then
            echo "   • pandas: missing"
            MISSING_CRITICAL=true
        fi
        
        # Check yaml (pyyaml)
        if ! $PYTHON_CMD -c "import yaml" 2>/dev/null; then
            echo "   • pyyaml: missing"
            MISSING_CRITICAL=true
        fi
        
        if [ "$MISSING_CRITICAL" = true ]; then
            echo "Installing dependencies from requirements.txt..."
            $PIP_CMD install -r requirements.txt
            echo "✅ Dependencies installed."
        else
            echo "✅ Critical dependencies are installed."
        fi
        echo ""
    fi
else
    echo "⚠️  Warning: requirements.txt not found. Cannot verify dependencies."
    echo ""
fi

# Check if optional analysis notebook exists
if [ ! -f "eda.ipynb" ]; then
    echo "⚠️  Warning: eda.ipynb not found. Exploratory analysis notebook is recommended for local analysis."
fi

# Check if src/ directory exists and has required modules
if [ ! -d "src" ]; then
    echo "❌ Error: src/ directory not found. ML pipeline modules are required."
    exit 1
fi

REQUIRED_MODULES=("config.py" "data_loader.py" "preprocessor.py" "model_trainer.py" "model_evaluator.py" "pipeline.py")
MISSING_MODULES=()

for module in "${REQUIRED_MODULES[@]}"; do
    if [ ! -f "src/$module" ]; then
        MISSING_MODULES+=("$module")
    fi
done

if [ ${#MISSING_MODULES[@]} -gt 0 ]; then
    echo "❌ Error: Missing required Python modules:"
    for module in "${MISSING_MODULES[@]}"; do
        echo "   • src/$module"
    done
    exit 1
fi

# Check if config file exists (optional - will use defaults if not found)
CONFIG_FILE="config.yaml"
CONFIG_ARG=""
if [ -f "$CONFIG_FILE" ]; then
    # Check if PyYAML is available (required for YAML config)
    if $PYTHON_CMD -c "import yaml" 2>/dev/null; then
        CONFIG_ARG="--config $CONFIG_FILE"
        echo "Using configuration file: $CONFIG_FILE"
    else
        echo "Warning: $CONFIG_FILE found but PyYAML not available."
        echo "         Using default configuration instead."
    fi
else
    echo "Info: $CONFIG_FILE not found. Using default configuration."
fi
echo ""

# Execute the MLP pipeline
echo "Executing Machine Learning Pipeline..."
echo ""

# Create results directory for output files
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

# Generate output file names
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_LOG="${RESULTS_DIR}/pipeline_execution_${TIMESTAMP}.log"
OUTPUT_MD="${RESULTS_DIR}/pipeline_execution_output.md"

# Execute pipeline and capture output (display on terminal AND save to log)
$PYTHON_CMD src/pipeline.py $CONFIG_ARG "$@" 2>&1 | tee "$OUTPUT_LOG"

EXIT_CODE=${PIPESTATUS[0]}

# Create clean markdown version for assessors (strip ANSI codes, add header)
{
    echo "# Pipeline Execution Output"
    echo ""
    echo "**Execution Date:** $(date '+%Y-%m-%d %H:%M:%S')"
    echo "**Timestamp:** \`${TIMESTAMP}\`"
    echo ""
    echo "---"
    echo ""
    echo "\`\`\`"
    # Strip ANSI escape codes for clean markdown
    sed 's/\x1b\[[0-9;]*m//g' "$OUTPUT_LOG"
    echo "\`\`\`"
} > "$OUTPUT_MD"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Pipeline execution completed successfully."
    echo "📝 Output saved to markdown: $OUTPUT_MD"
else
    echo ""
    echo "❌ Pipeline execution failed with exit code $EXIT_CODE."
    echo "📝 Log saved to: $OUTPUT_LOG"
    exit $EXIT_CODE
fi

