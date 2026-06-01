"""MAC-CROSS — cross-company search and embeddings service."""
from __future__ import annotations

import logging
import os

from flask import Flask, jsonify

from cross_routes import bp as cross_bp
from session_store import register_session_teardown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = Flask(__name__)
register_session_teardown(app)
app.register_blueprint(cross_bp)


@app.get("/")
def index():
    return jsonify(
        {
            "service": "mac-cross",
            "version": os.environ.get("MAC_CROSS_VERSION", "1"),
            "endpoints": [
                "GET /health",
                "GET /ready",
                "GET /api/embeddings/status",
                "POST /api/embeddings/rebuild",
                "POST /api/cross-company/match",
            ],
        }
    )
