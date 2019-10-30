# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 09:22:40 2019

@author: LENOVO
"""

import cv2, glob, imutils, sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn import svm
from sklearn.model_selection import train_test_split
import random
import joblib
    
def getColorMoments(img):
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    feature = [np.mean(R), np.std(R), np.mean(G), np.std(G), np.mean(B), np.std(B)]
    return feature/np.mean(feature) 
    
def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])

	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)

	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist) 

	# return the flattened histogram as the feature vector
	return hist.flatten()

def getFeatures(img):
    cm = getColorMoments(img)   
    ch = extract_color_histogram(img)
#    print(ch)
#    ch = np.array(ch)
#    ch=np.reshape(ch, (-1,6))
#    print(len(ch))
#    print(type(ch))
##    ch=np.reshape(ch, 6)
#    ch.append(cm)
#    return [cm, ch]
#    return np.reshape([cm, ch], 6, order='F')
    return ch
 

def Apprentissage(modelName, data_path):
    # Extraire les paths des images de type jpg qui se trouve dans le dossier : data_path
    all_files = glob.glob(str(data_path)+os.path.sep+"*"+os.path.sep)
    # ovrir un fichier de type csv on mode ecriture pour stocker les entites extrait pour ces images 
    print("Apprentissage : ", end="", flush=True)
       
    features = []
    labels = []
    for dir in all_files:
        if os.path.isdir(dir):
            label = dir.split(os.path.sep)[-2]
            print("\nClasse : " +str(label))
            files = glob.glob(str(dir)+os.path.sep+"*.*")
            for imagePath in files:
                img = cv2.imread(imagePath)
                feature = getFeatures(img)
                features.append(feature)
                labels.append(label)
                print(".", end="", flush=True)
    
    features = np.array(features)
    labels = np.array(labels)

    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
            features, labels, test_size=0.2, random_state=random.seed())
    model = svm.SVC(gamma=0.01, C=100)
#    cl= svm.LinearSVC()
    model.fit(trainFeat,trainLabels)
    
    # save the model to disk
    joblib.dump(model, modelName)
    
    print("[DONE]")
    print("\n\nTaux de précision :", model.score(testFeat,testLabels))
    

        
def Classification(filename , ImageRequetePath):
    loaded_model = joblib.load(filename)
    
    ImageRequete = cv2.imread(ImageRequetePath)
    feature = getFeatures(ImageRequete)
    
    plt.xticks([]),plt.yticks([])
    plt.title('ImageRequete')
    plt.imshow(mpimg.imread(ImageRequetePath))
    plt.show()
    print("\nObject  Type : ==> ", loaded_model.predict([feature])[0])
    
    
# La fonction main
if __name__ == '__main__':
#    data_path = 'Classes'
    data_path = "DB2C"
    model = 'finalized_model.sav'

    Apprentissage(model, data_path)
    Classification(model, "DataToPredict/535000.jpg")

    sys.exit