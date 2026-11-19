"""
WSGI entry point for production servers (Waitress, uWSGI, mod_wsgi).
Import target:  wsgi:application
"""
import sys
import os

# Ensure backend/ is on the path when this module is imported directly
sys.path.insert(0, os.path.dirname(__file__))

from app import app as application  # noqa: F401 – exposed for WSGI servers
