from PyQt5.QtWidgets import QApplication
from GUI.MainWindow import MainWindow
from Model import Detector
import sys

if __name__=='__main__':
	app = QApplication(sys.argv)
	model = Detector()
	main_window = MainWindow(model)
	app.exec()