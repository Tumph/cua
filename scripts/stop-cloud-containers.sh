#!/bin/bash

# Simple wrapper script to stop CUA cloud containers
# This script will run the Python container stopper

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}==> $1${NC}"
}

print_success() {
    echo -e "${GREEN}==> $1${NC}"
}

print_error() {
    echo -e "${RED}==> $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}==> $1${NC}"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

print_info "🛑 CUA Cloud Container Stopper"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not found"
    exit 1
fi

# Check if the Python script exists
PYTHON_SCRIPT="$SCRIPT_DIR/stop_cloud_containers.py"
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    print_error "Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Check if CUA modules are available
print_info "Checking CUA dependencies..."
cd "$PROJECT_ROOT"

if ! python3 -c "from computer import Computer" &> /dev/null; then
    print_warning "CUA computer module not found. Installing dependencies..."
    
    # Try to install the computer module
    if [[ -d "libs/python/computer" ]]; then
        pip3 install -e libs/python/computer
    else
        print_error "CUA computer module not found. Please run this from the CUA project root."
        exit 1
    fi
fi

print_success "Dependencies OK"
echo

# Run the Python script
print_info "Running cloud container stopper..."
python3 "$PYTHON_SCRIPT" "$@"

exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    print_success "✅ Cloud container stopping completed successfully"
else
    print_error "❌ Cloud container stopping failed with exit code $exit_code"
fi

exit $exit_code
