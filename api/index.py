"""
Vercel Serverless Function entry point for Smart Job Portal.
Imports the Flask app and exposes it as a WSGI handler.
"""
import sys
from pathlib import Path

# Add project root directory to Python module search path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app, application  # noqa: F401
