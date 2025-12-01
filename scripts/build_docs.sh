#!/bin/bash
# Build Sphinx documentation

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Building Sphinx documentation...${NC}"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Must be run from project root directory${NC}"
    exit 1
fi

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}Error: Poetry is not installed${NC}"
    echo "Install Poetry: https://python-poetry.org/docs/#installation"
    exit 1
fi

# Check if docs dependencies are installed
echo -e "${BLUE}Checking documentation dependencies...${NC}"
if ! poetry run python -c "import sphinx" 2>/dev/null; then
    echo -e "${BLUE}Installing documentation dependencies...${NC}"
    poetry install --with docs
fi

# Clean previous build
echo -e "${BLUE}Cleaning previous build...${NC}"
cd docs
if [ -d "build" ]; then
    rm -rf build
fi

# Build HTML documentation
echo -e "${BLUE}Building HTML documentation...${NC}"
poetry run sphinx-build -W -b html source build/html

# Check if build was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Documentation built successfully!${NC}"
    echo -e "${GREEN}📄 Open docs/build/html/index.html to view${NC}"

    # Print file location
    DOCS_PATH=$(pwd)/build/html/index.html
    echo -e "${BLUE}File location: ${DOCS_PATH}${NC}"

    # Optionally open in browser (platform-specific)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v open &> /dev/null; then
            echo -e "${BLUE}Opening in browser...${NC}"
            open build/html/index.html
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v xdg-open &> /dev/null; then
            echo -e "${BLUE}Opening in browser...${NC}"
            xdg-open build/html/index.html
        fi
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows (Git Bash, Cygwin, or native)
        if command -v start &> /dev/null; then
            echo -e "${BLUE}Opening in browser...${NC}"
            start build/html/index.html
        fi
    fi
else
    echo -e "${RED}❌ Documentation build failed${NC}"
    exit 1
fi

cd ..
