# Gunicorn configuration for consecutive worker startup
import multiprocessing
import time
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes - use WEB_CONCURRENCY env var, default to 4
workers = int(os.environ.get('WEB_CONCURRENCY', 4))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
preload_app = True

# Timeouts
timeout = 120
graceful_timeout = 30
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Worker startup timing
def on_starting(server):
    """Called just after the server is started."""
    print(f"Server starting with {workers} workers...")

def post_worker_init(worker):
    """Called just after a worker has been initialized."""
    print(f"Worker {worker.pid} initialized")
    # Add a small delay to ensure workers start consecutively
    time.sleep(2)

def worker_int(worker):
    """Called just after a worker has been initialized."""
    print(f"Worker {worker.pid} received INT or QUIT signal")

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    print(f"About to fork worker {worker.age}")

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    print(f"Worker {worker.pid} forked")

def worker_abort(worker):
    """Called when a worker received SIGABRT signal."""
    print(f"Worker {worker.pid} received SIGABRT signal") 