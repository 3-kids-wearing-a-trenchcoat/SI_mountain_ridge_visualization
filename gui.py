from __future__ import annotations
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore, QtGui
from pyqtgraph.Qt import mkQApp, QtCore

class MainWindow (QtWidgets.QMainWindow):
    """pyqtgraph-based GUI"""
    # TODO: update

    # TODO: pause

    # TODO: (if I really need to) iterations per second

    # TODO: init

def run() -> None:
    mkQApp("SI simple demo")
    main_window = MainWindow()
    pg.exec()