# tests/conftest.py
import os
import sys

# Add project root to sys.path so `import hrm` works when running `pytest` from anywhere
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
