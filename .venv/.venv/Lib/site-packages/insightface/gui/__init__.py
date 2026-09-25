"""InsightFace Evaluation Studio.

The GUI package is intentionally import-light. Importing ``insightface.gui``
does not require PySide6; GUI dependencies are loaded by the entry point.
"""

__version__ = "1.0.1"

APP_NAME = "InsightFace Evaluation Studio"
APP_DISPLAY_NAME = "InsightFace Evaluation Studio v1.0.1"

__all__ = ["__version__", "APP_NAME", "APP_DISPLAY_NAME"]
