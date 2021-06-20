"""
Main module.
App object is instantiated which in turn loads the UI.
"""

import asyncio
import logging.config
import sys

import yaml
from PyQt5.QtWidgets import QApplication
from qasync import QEventLoop

from logic.app import App


def config_logging():
    """Configure the logging package for the whole application."""

    with open("logging_config.yaml", "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)


if __name__ == "__main__":

    config_logging()

    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    App(loop)
    app.exec_()

    with loop:
        sys.exit(loop.run_forever())
