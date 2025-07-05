#!/bin/bash
echo "Starting AI Forecast Intelligence Platform..."
echo "Waiting for app to be ready..."
sleep 5
echo "Starting gunicorn..."
exec gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout 120 --preload app:app 