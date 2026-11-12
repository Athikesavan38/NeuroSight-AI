"""
Run script for Brain Hemorrhage Detection Project.
Launches the Flask web application from any directory.
"""
import os
import sys

# Ensure FRONT END directory is in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONT_END_DIR = os.path.join(BASE_DIR, 'FRONT END')
sys.path.insert(0, FRONT_END_DIR)

from app import app

if __name__ == '__main__':
    print("=" * 60)
    print("  Brain Hemorrhage Detection Web Application")
    print("  Server running at: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)

