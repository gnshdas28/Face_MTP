import face_recognition
import os
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pylab
import scipy.io as sio
import xlsxwriter


# Import training images
source_folder="/home/das/opencv/samples/Das/Das_pre_trained/training_images"

known_faces=[]
map_label_name={}
numImgs=0
for filename in os.listdir(source_folder):
	# Load the jpg files into numpy arrays
	face_det = face_recognition.load_image_file(os.path.join(source_folder,filename))
	# Get the face encodings for each face in each image file
	face_det_encoding = face_recognition.face_encodings(face_det)[0]
	map_label_name[numImgs] = filename
	if numImgs is 0:
		known_faces = [face_det_encoding]
		numImgs=numImgs+1
	else:
		known_faces=np.concatenate((known_faces, [face_det_encoding]), axis=0)
		numImgs=numImgs+1

date = "19-10-17"

dest_folder="/home/das/opencv/samples/Das/Das_pre_trained/aligned_faces"+ "_" +date

# file_1="/home/das/opencv/samples/Das/Das_pre_trained/aligned_faces_3d/al_08.png"
# unknown_face = face_recognition.load_image_file(file_1)
# # Get the face encodings for each face in each image file
# unknown_face_encoding = face_recognition.face_encodings(unknown_face)[0]
# # results is an array of True/False telling if the unknown face matched anyone in the known_faces array
# results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

# print(results)


# Create an new Excel file and add a worksheet.
workbook = xlsxwriter.Workbook('results_recog1.xlsx')
worksheet = workbook.add_worksheet()

workbook2 = xlsxwriter.Workbook('attendance.xlsx')
worksheet2 = workbook2.add_worksheet()

results={}
num=2

# Widen the first column to make the text clearer.
worksheet.set_column('A:A', 20)
worksheet.set_column('B:B', 20)
worksheet.set_column('C:C', 20)
worksheet.set_column('F:F', 20)

worksheet.write('A1', 'FileName')
worksheet.write('B1', 'similarity measure')
worksheet.write('C1', 'Similar Training Image')
worksheet.write('F1', 'Test Image')

worksheet2.set_column('A:A', 20)
worksheet2.set_column('B:B', 10)
# worksheet2.set_column('C:C', 10)

worksheet2.write('A1', 'Roll No.s')
worksheet2.write('B1', date)
# worksheet2.write('C1', 'PIC')

roll_nos=[]
for idx in map_label_name:
	roll_nos.append(map_label_name[idx][:-4])
roll_nos.sort()

nm=2
map_roll_no_nm = {}
for elem in roll_nos:

	worksheet2.write('A{:d}'.format(nm), elem)
	worksheet2.write('B{:d}'.format(nm), 'A')


	# img1=mpimg.imread(os.path.join(source_folder,elem+'.JPG'))
	# shp=img1.shape
	# image_width = shp[0]
	# image_height = shp[1]
	# cell_width = 80.0
	# cell_height = 60.0
	# x_scale = cell_width/image_width * 1/10
	# y_scale = cell_height/image_height * 1/10
	# worksheet2.insert_image('C{:d}'.format(nm), os.path.join(source_folder,elem+'.JPG'),{'x_scale': x_scale, 'y_scale': y_scale})
	
	map_roll_no_nm[elem] = nm
	nm=nm+1
			
for filename in os.listdir(dest_folder):
	unknown_face = face_recognition.load_image_file(os.path.join(dest_folder,filename))
	# Get the face encodings for each face in each image file
	check=face_recognition.face_encodings(unknown_face)
	if check is not None:
		unknown_face_encoding = face_recognition.face_encodings(unknown_face)[0]
		# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
		results[filename] = face_recognition.compare_faces(known_faces, unknown_face_encoding)
		face_dist =face_recognition.face_distance(known_faces, unknown_face_encoding)
		max_face_dist = 10000

		for idx, val in enumerate(results[filename]):
			if val==True:
				# print(filename, map_label_name[idx])
				img1=mpimg.imread(os.path.join(source_folder,map_label_name[idx]))
				img2=mpimg.imread(os.path.join(dest_folder,filename))
				
				shp=img1.shape
				image_width = shp[0]
				image_height = shp[1]
				cell_width = 80.0
				cell_height = 60.0
				x_scale = cell_width/image_width
				y_scale = cell_height/image_height

				if face_dist[idx] < max_face_dist:
					max_face_dist = face_dist[idx]
					max_face_dist_label = map_label_name[idx][:-4]

				# Insert an image.
				worksheet.write('A{:d}'.format(num),  map_label_name[idx][:-4])
				worksheet.write('B{:d}'.format(num), face_dist[idx])
				worksheet.insert_image('C{:d}'.format(num), os.path.join(source_folder,map_label_name[idx]),{'x_scale': x_scale, 'y_scale': y_scale})

				shp=img2.shape
				image_width = shp[0]
				image_height = shp[1]
				cell_width = 100.0
				cell_height = 100.0
				x_scale = cell_width/image_width
				y_scale = cell_height/image_height
				worksheet.insert_image('F{:d}'.format(num), os.path.join(dest_folder,filename),{'x_scale': x_scale, 'y_scale': y_scale})
				num=num+15

		try:
			worksheet2.write('B{:d}'.format(map_roll_no_nm[max_face_dist_label]), 'P')
		except:
			pass

workbook2.close()
workbook.close()


