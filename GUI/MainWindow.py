from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QFileDialog, QMessageBox
from imutils.video import VideoStream
import cv2
import matplotlib.pyplot as plt

class MainWindow(QWidget):

	def __init__(self, detector):
		super().__init__()

		#Import variables
		self.detector = detector

		#Box layout
		box_layout = QGridLayout()

		#Buttons
		image_use_button = QPushButton('Use image')
		image_use_button.setStyleSheet('background-color: #FFFFFF;font-size: 24px;')
		image_use_button.clicked.connect(self.detect_image)

		video_use_button = QPushButton('Use video')
		video_use_button.setStyleSheet('background-color: #FFFFFF;font-size: 24px;')
		video_use_button.clicked.connect(self.detect_video)

		box_layout.addWidget(image_use_button,0,0)
		box_layout.addWidget(video_use_button,0,1)
		self.setLayout(box_layout)

		#Window settings
		self.setStyleSheet('background-color: #cc99ff;')
		self.setWindowTitle('Face detector')
		self.setFixedSize(260,60)
		self.show()

	def detect_image(self):
		file = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.png *.jpg)')
		if file[0]!='':
			image = self.detector.show_face(file[0])
			try:
				plt.figure(figsize=(12,8))
				plt.imshow(image)
				plt.show()
			except:
				msg = QMessageBox()
				msg.setIcon(QMessageBox.Critical)
				msg.setText('Faces not found!')
				msg.setWindowTitle('Error')
				msg.exec_()

	def detect_video(self):
		vs = VideoStream(src=0).start()
		while True:
			frame = vs.read()
			key = cv2.waitKey(1)
			image = self.detector.show_face(frame)
			if image is None:
				msg = QMessageBox()
				msg.setIcon(QMessageBox.Critical)
				msg.setText('Faces not found!')
				msg.setWindowTitle('Error')
				msg.exec_()
				break
			else:
				cv2.putText(image,
					'Print "Q" to exit',
					(5,30),
					cv2.FONT_HERSHEY_SIMPLEX,
					1.0,
					(255,255,255),
					2,
					cv2.LINE_AA)
				cv2.imshow('Your face',image)
				if key==ord('q'):
					break
		cv2.destroyAllWindows()