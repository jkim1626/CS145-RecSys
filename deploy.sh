#!/bin/bash

# CS145 Leaderboard Deployment Script
# Run this script on your lab server (fts@scai2.cs.ucla.edu)

echo "🚀 Starting CS145 Leaderboard deployment..."

# Set up directories
echo "📁 Setting up directories..."
mkdir -p uploads
mkdir -p logs

# Check Python version
echo "🐍 Checking Python version..."
python3 --version

# Install/upgrade pip
echo "📦 Installing dependencies..."
python3 -m pip install --user --upgrade pip

# Install requirements
python3 -m pip install --user Flask==2.3.3 Werkzeug==2.3.7 numpy pandas scikit-learn pyspark

# Set up environment variables
echo "🔧 Setting up environment..."
export FLASK_APP=app.py
export FLASK_ENV=production
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Make the script executable
chmod +x deploy.sh

echo "✅ Dependencies installed successfully!"

# Check if Java is available for PySpark
echo "☕ Checking Java availability..."
if command -v java &> /dev/null; then
    echo "✅ Java is available: $(java -version 2>&1 | head -n 1)"
else
    echo "⚠️  Java not found. PySpark may not work properly."
    echo "   You may need to install Java: sudo apt-get install openjdk-11-jre-headless"
fi

# Check if port is available
echo "🔍 Checking if port 5431 is available..."
if ! command -v netstat &> /dev/null; then
    echo "⚠️  netstat not available, skipping port check"
else
    if netstat -tuln | grep -q ":5431 "; then
        echo "⚠️  Port 5431 is already in use. You may need to choose a different port."
    else
        echo "✅ Port 5431 is available"
    fi
fi

echo ""
echo "🎯 Deployment complete! To start the application:"
echo ""
echo "   # Option 1: Run in foreground (for testing)"
echo "   python3 app.py"
echo ""
echo "   # Option 2: Run in background (recommended)"
echo "   nohup python3 app.py > logs/app.log 2>&1 &"
echo ""
echo "   # Option 3: Use the start script"
echo "   ./start.sh"
echo ""
echo "📍 Access the application at: http://scai2.cs.ucla.edu:5431"
echo "📊 Admin dashboard at: http://scai2.cs.ucla.edu:5431/admin"
echo ""
echo "📝 To stop the application:"
echo "   pkill -f 'python3 app.py'"
echo ""
echo "📋 To check if it's running:"
echo "   ps aux | grep 'python3 app.py'"
echo ""
echo "📄 To view logs:"
echo "   tail -f logs/app.log"