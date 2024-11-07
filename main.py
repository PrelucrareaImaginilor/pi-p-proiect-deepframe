import cv2 as cv
import numpy as np
import PyQt5 as qt
from interfata import *
import sys

# def rescaleFrame(frame, scale=0.75):
#     width = int(frame.shape[1]*scale)
#     height = int(frame.shape[0]*scale)
#     dimensions = (width, height)
#
#     return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())