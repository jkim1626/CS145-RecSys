#!/bin/bash

# Start script for CS145 Leaderboard
# This script starts the Flask application in the background

echo "🚀 Starting CS145 Leaderboard..."

# Check if already running
if pgrep -f "python3 app.py" > /dev/null; then
    echo "⚠️  Application is already running!"
    echo "   PID: $(pgrep -f 'python3 app.py')"
    echo "   To stop it: pkill -f 'python3 app.py'"
    exit 1
fi

# Set up environment
export FLASK_APP=app.py
export FLASK_ENV=production
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the application in background
nohup python3 app.py > logs/app.log 2>&1 &

# Get the PID
APP_PID=$!

# Wait a moment and check if it started successfully
sleep 2

if ps -p $APP_PID > /dev/null; then
    echo "✅ Application started successfully!"
    echo "   PID: $APP_PID"
    echo "   URL: http://scai2.cs.ucla.edu:5431"
    echo "   Admin: http://scai2.cs.ucla.edu:5431/admin"
    echo ""
    echo "📄 To view logs: tail -f logs/app.log"
    echo "📝 To stop: pkill -f 'python3 app.py'"
else
    echo "❌ Failed to start application!"
    echo "📄 Check logs: cat logs/app.log"
    exit 1
fi