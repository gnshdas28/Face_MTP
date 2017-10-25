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

cnn_pred="/home/das/opencv/samples/Das/Das_pre_trained/mmod_human_face_detector.dat"


cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_pred)
win = dlib.image_window()


date = "19-10-17"

source_folder="/home/das/opencv/samples/Das/Das_pre_trained/"+date
dest_folder="/home/das/opencv/samples/Das/Das_pre_trained/aligned_faces"+ "_" +date


try:
    os.stat(dest_folder)
except:
    os.mkdir(dest_folder)  

for filename in os.listdir(source_folder):
	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(os.path.join(source_folder,filename))
	# image = imutils.resize(image, width=1000)
	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# # initialize dlib's face detector (HOG-based) and then create
	# # the facial landmark predictor and the face aligner
	# detector = dlib.get_frontal_face_detector()
	# predictor = dlib.shape_predictor(shape_pred)
	# fa = FaceAligner(predictor, desiredFaceWidth=152)

	# # show the original input image and detect faces in the grayscale
	# # image
	# # cv2.imshow("Input", image)
	# rects = detector(gray, 2)
	# numImg=1
	# # loop over the face detections
	# for rect in rects:
	# 	# extract the ROI-region of interest of the *original* face, then align the face
	# 	# using facial landmarks
	# 	(x, y, w, h) = rect_to_bb(rect)
	# 	faceOrig = imutils.resize(image[y:y + h, x:x + w], width=152)
	# 	faceAligned = fa.align(image, gray, rect)
		
	# 	#save aligned & original images- for comparison
	# 	cv2.imwrite(dest_folder+"/"+str(numImg)+filename, faceAligned)
		
	# 	shape = predictor(gray, rect)
	# 	shape = face_utils.shape_to_np(shape)
	# 	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	# 	# loop over the (x, y)-coordinates for the facial landmarks
	# 	# and draw them on the image
	# 	for (x, y) in shape:
	# 		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

	# 	numImg=numImg+1


	# The 1 in the second argument indicates that we should upsample the image
	# 1 time.  This will make everything bigger and allow us to detect more
	# faces.
	dets = cnn_face_detector(image, 1)
	'''
	This detector returns a mmod_rectangles object. This object contains a list of mmod_rectangle objects.
	These objects can be accessed by simply iterating over the mmod_rectangles object
	The mmod_rectangle object has two member variables, a dlib.rectangle object, and a confidence score.

	It is also possible to pass a list of images to the detector.
	- like this: dets = cnn_face_detector([image list], upsample_num, batch_size = 128)

	In this case it will return a mmod_rectangless object.
	This object behaves just like a list of lists and can be iterated over.
	'''
	print("Number of faces detected: {}".format(len(dets)))
	for i, d in enumerate(dets):
		print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

	rects = dlib.rectangles()
	rects.extend([d.rect for d in dets])

	win.clear_overlay()
	win.set_image(image)
	win.add_overlay(rects)
	dlib.hit_enter_to_continue()


	# # save the output image with the face detections + facial landmarks
	cv2.imwrite("/home/das/opencv/samples/Das/Das_pre_trained/results_das/res_" + filename,image) 