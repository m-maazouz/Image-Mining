# import the necessary packages
# to extarct features from our dataset
from colordescriptor import ColorDescriptor
#to grabbing the file paths to our images
import glob
# for OpenCV bindings
import cv2
 
# initialize our ColorDescriptor using 8 Hue bins, 12 Saturation bins, and 3 Value bins.
cd = ColorDescriptor((8, 12, 3))
# open (if exist) or create the output index file for writing
output = open("index.csv", "w")
 
# use glob to grab the image paths and loop over them
for imagePath in glob.glob("dataset" + "/*.jpg"):
	# extract the image ID (i.e. the unique filename) from the image path and load the image itself
	imageID = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath) 
	# describe the image
	features = cd.describe(image) 
	# write the features to file
	features = [str(f) for f in features]
	output.write("%s,%s\n" % (imageID, ",".join(features)))
 
# close the index file
output.close()