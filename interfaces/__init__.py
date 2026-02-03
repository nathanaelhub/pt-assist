"""
PT Assist Interfaces

This package contains the user interfaces for the PT Assistant:
- app.py: Streamlit web application
- cli.py: Command-line interface
"""

from .cli import main as cli_main

__all__ = ['cli_main']
