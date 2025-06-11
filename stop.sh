#!/bin/bash

# Stop script for CS145 Leaderboard

echo "🛑 Stopping CS145 Leaderboard..."

# Check if running
if ! pgrep -f "python3 app.py" > /dev/null; then
    echo "⚠️  Application is not running!"
    exit 1
fi

# Get PID before stopping
APP_PID=$(pgrep -f "python3 app.py")
echo "   Stopping PID: $APP_PID"

# Stop the application
pkill -f "python3 app.py"

# Wait a moment and verify it stopped
sleep 2

if ! pgrep -f "python3 app.py" > /dev/null; then
    echo "✅ Application stopped successfully!"
else
    echo "⚠️  Application may still be running. Try: kill -9 $APP_PID"
fi