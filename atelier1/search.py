# import the necessary packages
from colordescriptor import ColorDescriptor
from tkinter import filedialog
from searcher import Searcher
import cv2
import matplotlib.pyplot as plt
import tkinter as tk

root = tk.Tk()
root.withdraw()
query_path = filedialog.askopenfilename(title='chose your query image file')
index_path = filedialog.askopenfilename(title='chose your index file')

# initialize the image descriptor
cd = ColorDescriptor((8, 12, 3))

# load the query image and describe it
query = cv2.imread(query_path)
features = cd.describe(query)

# perform the search
searcher = Searcher(index_path)
results = searcher.search(features)

plt.subplot(2, 3, 1)
plt.title('query_image')
plt.imshow(query)
plt.rcParams['figure.figsize'] = [15, 7]

i = 2
# loop over the results
for (score, resultID) in results:
	# load the result image and display it
	result = cv2.imread(resultID)
	plt.subplot(2, 3, i)
    
	plt.title('result '+ str(i-1))
	plt.imshow(result)
	i=i+1
plt.show()