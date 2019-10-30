# import the necessary 
# use NumPy for numerical processing
import numpy as np
# for OpenCV bindings
import cv2
# to check OpenCV version
import imutils

# This class will encapsulate all the necessary logic to extract our 3D HSV color histogram from our images
class ColorDescriptor:
	def __init__(self, bins):
		# store the number of bins for the 3D histogram
		self.bins = bins

# describe  method requires an image which is the image we want to describe
	def describe(self, image):
		# convert from the RGB color space to the HSV color space
        # followed by initializing our list of features to quantify and represent our image .
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []

		# grab the dimensions and compute the center of the image
		(h, w) = image.shape[:2]
		(cX, cY) = (int(w * 0.5), int(h * 0.5))

		# divide the image into four rectangles/segments 
		# (top-left,top-right, bottom-right, bottom-left)
		segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
			(0, cX, cY, h)]

		# construct an elliptical mask representing the center of the image
		(axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        #  initialize a blank image with the same dimensions of the image we want to describe
		ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
        # draw the actual ellipse
		cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
        # describe cv2.ellipse() parameters
        # ellipMask : The image we want to draw the ellipse on
        # (cX, cY) : representing the center (x, y) coordinates of the image.
        # (axesX, axesY) : representing the length of the axes of the ellipse. In this case, the ellipse will stretch to be 75% of the width and height of the image  that we are describing.
        # 0 : The rotation of the ellipse. In this case, no rotation is required so we supply a value of 0 degrees.
        # 0 : The starting angle of the ellipse.
        # 360 : The ending angle of the ellipse. Looking at the previous parameter, this indicates that we’ll be drawing an ellipse from 0 to 360 degrees (a full “circle”).
        # 255 : The color of the ellipse. The value of 255 indicates “white”, meaning that our ellipse will be drawn white on a black background.
        # -1 : The border size of the ellipse. Supplying a positive integer r will draw a border of size r pixels. Supplying a negative value for r will make the ellipse “filled in”

		# loop over the segments
		for (startX, endX, startY, endY) in segments:
			# construct a mask for each corner of the image 
			# subtracting the elliptical center from it
			cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
			cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
			cornerMask = cv2.subtract(cornerMask, ellipMask)

			# extract a color histogram from the image then update the feature vector
			hist = self.histogram(image, cornerMask)
			features.extend(hist)

		# extract a color histogram from the elliptical region and update the feature vector
		hist = self.histogram(image, ellipMask)
		features.extend(hist)

		# return the feature vector
		return features


    # histogram  method requires two arguments : 
    # the first is the image  that we want to describe 
    # the second is the mask  that represents the region of the image we want to describe.
	def histogram(self, image, mask):
		# extract a 3D color histogram from the masked region of the image using the supplied number of bins per channel
		hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
			[0, 180, 0, 256, 0, 256])

		# normalize the histogram if we are using OpenCV 2.4
		if imutils.is_cv2():
			hist = cv2.normalize(hist).flatten()
!!!!!!!
		# otherwise handle for OpenCV 3+
		else:
			hist = cv2.normalize(hist, hist).flatten()

		# return the histogram
		return hist