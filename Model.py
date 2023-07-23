import cv2
import os

class Detector:

	def __init__(self):
		#Import models and set mean values
		face1 = 'model/opencv_face_detector.pbtxt'
		face2 = 'model/opencv_face_detector_uint8.pb'
		age1 = 'model/age_deploy.prototxt'
		age2 = 'model/age_net.caffemodel'
		gen1 = 'model/gender_deploy.prototxt'
		gen2 = 'model/gender_net.caffemodel'

		self.model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)

		#Using models
		self.face = cv2.dnn.readNet(face2,face1)
		self.age = cv2.dnn.readNet(age2,age1)
		self.gen = cv2.dnn.readNet(gen2,gen1)

		#Set categories
		self.age_categories = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
		self.gen_categories = ['Male','Female']


	def show_face(self,input_data):
		#Get image
		try:
			image = cv2.cvtColor(cv2.imread(input_data),cv2.COLOR_BGR2RGB)
		except:
			image = input_data
		h = image.shape[0]
		w = image.shape[1]

		#Indentification the face blob
		blob = cv2.dnn.blobFromImage(image,1.0,(300,300),[104,117,123])
		self.face.setInput(blob)
		detections = self.face.forward()

		#Creating the bounding box
		faceBoxes = []
		for i in range(detections.shape[2]):
			confindence = detections[0,0,i,2]
			if confindence > 0.7:
				x1 = int(detections[0,0,i,3]*w)
				y1 = int(detections[0,0,i,4]*h)
				x2 = int(detections[0,0,i,5]*w)
				y2 = int(detections[0,0,i,6]*h)

				faceBoxes.append([x1,y1,x2,y2])
				cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),int(round(h/150)),8)

		#Checking face detection
		if not faceBoxes:
			return

		#Showing faces in photo
		for faceBox in faceBoxes:
			
			#Extract face
			face_res = image[max(0,faceBox[1]-15):min(faceBox[3]+15,image.shape[0]-1),
						 max(0,faceBox[0]-15):min(faceBox[2]+15,image.shape[1]-1)]

			#Extract blob
			blob_res = cv2.dnn.blobFromImage(face_res,1.0,(227,227),self.model_mean_values,swapRB=False)

			#Predict gender
			self.gen.setInput(blob_res)
			gen_pred = self.gen.forward()
			gen_res = self.gen_categories[gen_pred[0].argmax()]

			#Predict age
			self.age.setInput(blob_res)
			age_pred = self.age.forward()
			age_res = self.age_categories[age_pred[0].argmax()]

			#Putting text of age and gender at the top of box
			cv2.putText(image,
						f'{gen_res}, {age_res}',
						(faceBox[0]-30,faceBox[1]-20),
						cv2.FONT_HERSHEY_SIMPLEX,
						1.3,
						(255,255,255),
						2,
						cv2.LINE_AA)
			return image
