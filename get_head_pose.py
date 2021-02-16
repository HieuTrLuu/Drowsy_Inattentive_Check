#!/usr/bin/env python
from imutils import face_utils
import cv2
import numpy as np
import dlib

def get_6_main_keypoints(key_points):
	# nose 31
	# chin 9
	# left eye left corner 37
	# right eye right corner 46
	# Left Mouth corner 49
	# Right mouth corner 55
	return key_points[[30,8,36,45,48,54]]
	


# Read Image
im = cv2.imread("headPose.jpg");
size = im.shape
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data\shape_predictor_68_face_landmarks.dat')
rects = detector(gray, 0)

for rect in rects:
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	image_pints = get_6_main_keypoints(shape)
	image_pints = np.array(image_pints, dtype="double") 
	# image_pints = [(x,y) for x,y in image_pints]
	# print(image_pints)
	# extract the left and right eye coordinates, then use the
	# coordinates to compute the eye aspect ratio for both eyes
   




# 2D image points. If you change the image, you need to change vector
# image_pints = np.array([
#                             (359, 391),     # Nose tip
#                             (399, 561),     # Chin
#                             (337, 297),     # Left eye left corner
#                             (513, 301),     # Right eye right corne
#                             (345, 465),     # Left Mouth corner
#                             (453, 469)      # Right mouth corner
#                         ], dtype="double")

print(f"image_pints.shape {image_pints.shape}")
# key_points = face_utils.shape_to_np(preds)
# image_pints = get_6_main_keypoints(key_points)

# def get_head_pose(image_pints, gray):
# 	# 3D model points.
# 	model_points = np.array([
# 								(0.0, 0.0, 0.0),             # Nose tip
# 								(0.0, -330.0, -65.0),        # Chin
# 								(-225.0, 170.0, -135.0),     # Left eye left corner
# 								(225.0, 170.0, -135.0),      # Right eye right corne
# 								(-150.0, -150.0, -125.0),    # Left Mouth corner
# 								(150.0, -150.0, -125.0)      # Right mouth corner
							
# 							])


# 	# Camera internals

# 	focal_length = size[1]
# 	center = (size[1]/2, size[0]/2)
# 	camera_matrix = np.array(
# 							[[focal_length, 0, center[0]],
# 							[0, focal_length, center[1]],
# 							[0, 0, 1]], dtype = "double"
# 							)

# 	print("Camera Matrix :\n {0}".format(camera_matrix))

# 	dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
# 	(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_pints, camera_matrix, dist_coeffs)

# 	print("Rotation Vector:\n {0}".format(rotation_vector))
# 	print("Translation Vector:\n {0}".format(translation_vector))


# 	# Project a 3D point (0, 0, 1000.0) onto the image plane.
# 	# We use this to draw a line sticking out of the nose


# 	(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

# 	for p in image_pints:
# 		cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)


# 	p1 = ( int(image_pints[0][0]), int(image_pints[0][1]))
# 	p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

# 	cv2.line(im, p1, p2, (255,0,0), 2)

# 	# Display image
	
# 	return im

    
def get_head_pose(image_pints, image):
	# 3D model points.
	size = image.shape
	model_points = np.array([
								(0.0, 0.0, 0.0),             # Nose tip
								(0.0, -330.0, -65.0),        # Chin
								(-225.0, 170.0, -135.0),     # Left eye left corner
								(225.0, 170.0, -135.0),      # Right eye right corne
								(-150.0, -150.0, -125.0),    # Left Mouth corner
								(150.0, -150.0, -125.0)      # Right mouth corner
							
							])


	# Camera internals

	focal_length = size[1]
	center = (size[1]/2, size[0]/2)
	camera_matrix = np.array(
							[[focal_length, 0, center[0]],
							[0, focal_length, center[1]],
							[0, 0, 1]], dtype = "double"
							)

	print("Camera Matrix :\n {0}".format(camera_matrix))
	# print(f"image_pints {(model_points, image_pints, camera_matrix, dist_coeffs)}")
	dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
	(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_pints, camera_matrix, dist_coeffs)


	# print("Rotation Vector:\n {0}".format(rotation_vector))
	# print("Translation Vector:\n {0}".format(translation_vector))


	# Project a 3D point (0, 0, 1000.0) onto the image plane.
	# We use this to draw a line sticking out of the nose


	(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

	for p in image_pints:
		cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)


	p1 = ( int(image_pints[0][0]), int(image_pints[0][1]))
	p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

	cv2.line(image, p1, p2, (255,0,0), 2)

	# Display image
	# cv2.imshow("Output", im)
	# cv2.waitKey(0)
	return image

im = get_head_pose(image_pints, im)
print(im.dtype)
cv2.imshow("Output", im)
cv2.waitKey(0)