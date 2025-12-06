#!/bin/bash
# Script to set up Python environment

echo "=========================================="
echo "🔧 Setting Up Python Environment"
echo "=========================================="
echo ""

# Check for Python 3.11 or 3.12
PYTHON_CMD=""
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    echo "✅ Found Python 3.12"
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo "✅ Found Python 3.11"
else
    echo "❌ Python 3.11 or 3.12 not found"
    echo ""
    echo "Please install Python 3.11 or 3.12:"
    echo "  Method 1: Using Homebrew"
    echo "    brew install python@3.12"
    echo ""
    echo "  Method 2: Download from official website"
    echo "    https://www.python.org/downloads/"
    echo ""
    exit 1
fi

echo ""
echo "📦 Creating virtual environment..."
$PYTHON_CMD -m venv venv

if [ $? -ne 0 ]; then
    echo "❌ Virtual environment creation failed"
    exit 1
fi

echo "✅ Virtual environment created: venv/"
echo ""
echo "=========================================="
echo "📝 Next Steps:"
echo "=========================================="
echo ""
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Upgrade pip:"
echo "   pip install --upgrade pip"
echo ""
echo "3. Install dependencies:"
echo "   pip install -r requirements.txt"
echo ""
echo "4. Start training:"
echo "   ./quick_train.sh"
echo ""
