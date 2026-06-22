"""
Shared test fixtures for the full test-suite.

Sets up an offscreen QApplication (session-scoped) so that any test
file that imports PyQt6 widgets can run without a physical display.
The environment variable must be set *before* any Qt module is imported,
so we place it here in conftest.py which is loaded first.
"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="session")
def qapp():
    """Session-scoped QApplication for all GUI tests."""
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    # Don't quit – other tests in the same session may still need it
