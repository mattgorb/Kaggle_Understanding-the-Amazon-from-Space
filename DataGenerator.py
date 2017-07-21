import csv
import itertools
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from random import randint
import random
import numpy as np
import cv2
import sys
import scipy.io
import numpy as np
import os.path
from keras.utils import np_utils
from string import strip
import math
from keras.preprocessing.image import ImageDataGenerator
import os.path
import os
 
class DataGenerator:
    outputData=[]
    Cars85th=0
    categories={}

    def TrainDataGenerate(self,batch_size,image_aug=False):
        rounds=6
        datagen = ImageDataGenerator(rotation_range=90,
            horizontal_flip=True,
	    width_shift_range=0.1,
    	    height_shift_range=0.1,
    	    shear_range=0.1,
    	    zoom_range=0.1,
            vertical_flip=True,
	    fill_mode='nearest')
        batch_features=np.zeros((batch_size,256,256,3))
        batch_labels=np.zeros((batch_size,17))

        while True:
            batch_features=np.zeros(batch_features.shape)
            batch_labels=np.zeros(batch_labels.shape)
            for i in range(batch_size):
		
                index= randint(0, self.Cars85th)
		row=self.outputData[index]
                filename=row[0]
                img = cv2.imread("train-jpg/"+filename+".jpg", cv2.IMREAD_COLOR)
                resized_image=img
		resized_image= resized_image.astype('float32')
                resized_image = resized_image[np.newaxis, :, :, :]
		                
                label=row[1].split()
                out=[]

                for item in label:
                    categoryInt=self.categories[item]
                    out.append(categoryInt)
               
		labelArray=[0]*17
		for item in out:
		    labelArray[item]=1
		labelArray=np.array(labelArray)
		labelArray=labelArray[np.newaxis, :]
		if(image_aug):
			x, y=next(datagen.flow(resized_image,np.array(labelArray),shuffle=False,batch_size=rounds))
			x/=255
		else:
			resized_image/=255
			x=resized_image
			y=np.array(labelArray)
		batch_features[i] = x
		batch_labels[i] = y
#	    print batch_features.shape
            yield batch_features, batch_labels

   
                
    def ValDataGenerate(self,batch_size):
        rounds=4
        datagen = ImageDataGenerator()
        batch_features=np.zeros((batch_size,256,256,3))
        batch_labels=np.zeros((batch_size,17))

        while True:
            batch_features=np.zeros(batch_features.shape)
            batch_labels=np.zeros(batch_labels.shape)
            for i in range(batch_size):
		
                index= randint(self.Cars85th, len(self.outputData)-1)
                row=self.outputData[index]
                filename=row[0]
                img = cv2.imread("train-jpg/"+filename+".jpg", cv2.IMREAD_COLOR)
                #resized_image = cv2.resize(img, (224, 224))
                resized_image=img
		resized_image= resized_image.astype('float32')
                resized_image = resized_image[np.newaxis, :, :, :]
                
                label=row[1].split()
                out=[]

                for item in label:
                    categoryInt=self.categories[item]
                    out.append(categoryInt)
               
		labelArray=[0]*17
		#labelArray=np.zeros((1,17))
		for item in out:
		    labelArray[item]=1
		labelArray=np.array(labelArray)
		labelArray=labelArray[np.newaxis, :]

		x, y=next(datagen.flow(resized_image,np.array(labelArray),batch_size=rounds))
		x/=255
		batch_features[i] = x
		batch_labels[i] = y
       
            yield batch_features, batch_labels
   
    def loadValData(self):
	    length=int((len(self.outputData)-1)-self.Cars85th)
            batch_features=np.zeros((length,256,256,3))
            batch_labels=np.zeros((length,17))
	    i=0
	    batch_features=np.zeros(batch_features.shape)
            batch_labels=np.zeros(batch_labels.shape)
            for index in range(self.Cars85th, len(self.outputData)-1):
                row=self.outputData[index]
		filename=row[0]
                img = cv2.imread("train-jpg/"+filename+".jpg", cv2.IMREAD_COLOR)
                resized_image=img
		resized_image= resized_image.astype('float32')
                resized_image = resized_image[np.newaxis, :, :, :]

                label=row[1].split()
                out=[]

                for item in label:
                    categoryInt=self.categories[item]
                    out.append(categoryInt)

                labelArray=[0]*17
                #labelArray=np.zeros((1,17))
                for item in out:
                    labelArray[item]=1
                labelArray=np.array(labelArray)
                labelArray=labelArray[np.newaxis, :]
                resized_image/=255
                batch_features[i] = resized_image
                batch_labels[i] = np.array(labelArray)
		i+=1
	    return batch_features, batch_labels	



    def loadData(self):
        global outputData, Cars85th
        with open('train_v2.csv', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in itertools.islice(reader, 1,40479):
                self.outputData.append(row)
#		print row
        random.shuffle(self.outputData)
	random.shuffle(self.outputData)
	random.shuffle(self.outputData)
	random.shuffle(self.outputData)
	random.shuffle(self.outputData)
#	print self.outputData
        self.Cars85th=int(len(self.outputData)*.9)
       
 
    def createCategories(self):
        global categories
        i=0
        with open('train_v2.csv', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in itertools.islice(reader, 1,40479):
                label=row[1].split()
                for item in label:
                    if self.categories.get(item)==None:
                        self.categories[item]=i
                        i=i+1
                if i==15:
                    self.categories['blow_down']=16
		    

                    #categoryInt=categories[item]
                    #out.append(categoryInt)
               
                #outputData.append(out)
              
        #self.multiLabelArray=MultiLabelBinarizer().fit_transform(outputData)
        #print multiLabelArray

    def TrainDataGenerate_All(self,batch_size,image_aug=False):
        rounds=4
        datagen = ImageDataGenerator(rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        batch_features=np.zeros((batch_size,224,224,3))
        batch_labels=np.zeros((batch_size,17))

        while True:
            batch_features=np.zeros(batch_features.shape)
            batch_labels=np.zeros(batch_labels.shape)
            for i in range(batch_size):
		
                index= randint(0, len(self.outputData))
#                print index
		row=self.outputData[index]
                filename=row[0]
                img = cv2.imread("train-jpg/"+filename+".jpg", cv2.IMREAD_COLOR)
		resized_image = cv2.resize(img, (224, 224))
                resized_image= resized_image.astype('float32')
                resized_image = resized_image[np.newaxis, :, :, :]
		                
                label=row[1].split()
                out=[]

                for item in label:
                    categoryInt=self.categories[item]
                    out.append(categoryInt)
               
		labelArray=[0]*17
		for item in out:
		    labelArray[item]=1
		labelArray=np.array(labelArray)
		labelArray=labelArray[np.newaxis, :]
		if(image_aug):
			x, y=next(datagen.flow(resized_image,np.array(labelArray),batch_size=rounds))
			x/=255
		else:
			resized_image/=255
			x=resized_image
			y=np.array(labelArray)
		batch_features[i] = x
		batch_labels[i] = y
#	    print batch_features.shape
            yield batch_features, batch_labels



   
