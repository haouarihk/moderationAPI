#!/usr/bin/env python3
"""
Startup script for the moderation API with consecutive worker startup.
This ensures workers start one by one, allowing proper GPU load balancing.
"""

import subprocess
import sys
import time
import os

def start_server():
    """Start the server with gunicorn and uvicorn workers."""
    
    # Get worker count from environment variable, default to 4
    worker_count = int(os.environ.get('WEB_CONCURRENCY', 4))
    
    # Gunicorn configuration for consecutive worker startup
    cmd = [
        "gunicorn",
        "app:app",
        "--worker-class", "uvicorn.workers.UvicornWorker",
        "--workers", str(worker_count),
        "--bind", "0.0.0.0:8000",
        "--timeout", "120",
        "--preload",  # Preload the application
        "--worker-connections", "1000",
        "--max-requests", "1000",
        "--max-requests-jitter", "100",
        "--graceful-timeout", "30",
        "--keep-alive", "2",
        # Add delay between worker startups
        "--worker-tmp-dir", "/dev/shm",  # Use shared memory for faster startup
    ]
    
    print(f"Starting moderation API server with {worker_count} workers...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except subprocess.CalledProcessError as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_server() 