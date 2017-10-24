# USAGE
# python align_faces.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# # import the necessary packages
# from imutils import face_utils
from imutils import face_utils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import os

# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="path to facial landmark predictor")
# # ap.add_argument("-i", "--image", required=True,
# # 	help="path to input image")
# # args = vars(ap.parse_args())
# args = vars(ap.parse_args())

shape_pred="/home/das/opencv/samples/Das/Das_pre_trained/shape_predictor_68_face_landmarks.dat"

source_folder="/home/das/opencv/samples/Das/Das_pre_trained/pics"
dest_folder="/home/das/opencv/samples/Das/Das_pre_trained/aligned_faces/"

for filename in os.listdir(source_folder):
	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(os.path.join(source_folder,filename))
	image = imutils.resize(image, width=1000)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor and the face aligner
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(shape_pred)
	fa = FaceAligner(predictor, desiredFaceWidth=152)

	# show the original input image and detect faces in the grayscale
	# image
	# cv2.imshow("Input", image)
	rects = detector(gray, 2)
	numImg=1
	# loop over the face detections
	for rect in rects:
		# extract the ROI-region of interest of the *original* face, then align the face
		# using facial landmarks
		(x, y, w, h) = rect_to_bb(rect)
		faceOrig = imutils.resize(image[y:y + h, x:x + w], width=152)
		faceAligned = fa.align(image, gray, rect)
		
		#save aligned & original images- for comparison
		cv2.imwrite(dest_folder+"aligned"+str(numImg)+filename, faceAligned)
		
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

		numImg=numImg+1

	# # save the output image with the face detections + facial landmarks
	cv2.imwrite("/home/das/opencv/samples/Das/Das_pre_trained/results_das/res_" + filename,image) 