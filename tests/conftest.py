import os
import sys

# Package uses a src/ layout and is not pip-installed; put it on the path.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
