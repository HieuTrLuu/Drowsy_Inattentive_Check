# see more at: https://www.pyimagesearch.com/category/object-tracking/
# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
	def __init__(self, maxDisappeared=1000):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared

	def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)
			# return early as there are no centroids or tracking info
			# to update
			return self.objects

# correlation tracker
# if conf > args["confidence"] and label == args["label"]:
# 	# compute the (x, y)-coordinates of the bounding box
# 	# for the object
# 	box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
# 	(startX, startY, endX, endY) = box.astype("int")
# 	# construct a dlib rectangle object from the bounding
# 	# box coordinates and then start the dlib correlation
# 	# tracker
# 	tracker = dlib.correlation_tracker()
# 	rect = dlib.rectangle(startX, startY, endX, endY)
# 	tracker.start_track(rgb, rect)
# 	# draw the bounding box and text for the object
# 	cv2.rectangle(frame, (startX, startY), (endX, endY),
# 		(0, 255, 0), 2)
# 	cv2.putText(frame, label, (startX, startY - 15),
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
# else:
# 	# update the tracker and grab the position of the tracked
# 	# object
# 	tracker.update(rgb)
# 	pos = tracker.get_position()
# 	# unpack the position object
# 	startX = int(pos.left())
# 	startY = int(pos.top())
# 	endX = int(pos.right())
# 	endY = int(pos.bottom())
# 	# draw the bounding box from the correlation object tracker
# 	cv2.rectangle(frame, (startX, startY), (endX, endY),
# 		(0, 255, 0), 2)
# 	cv2.putText(frame, label, (startX, startY - 15),
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)		