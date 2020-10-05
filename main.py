from PyQt5.QtWidgets import *
from src.ui import paint

if __name__ == '__main__':

    app = QApplication([])
    window = paint.MainWindow()
    app.exec_()