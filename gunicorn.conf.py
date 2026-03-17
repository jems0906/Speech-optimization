# Gunicorn configuration for production deployment.
# Run with: gunicorn src.serving.app:app -c gunicorn.conf.py
#
# Each worker handles one concurrent request; for GPU inference keep workers=1
# so all requests share the same loaded model and GPU context.

bind = "0.0.0.0:8000"
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = "info"
