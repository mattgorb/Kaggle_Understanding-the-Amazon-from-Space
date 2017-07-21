from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense,Activation, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Dropout,Conv2D, BatchNormalization, MaxPooling2D, Flatten
from keras import backend as K
from DataGenerator import DataGenerator
from random import randint
import random
import sys
import cv2
import os
import numpy as np
from keras.models import model_from_yaml
from sklearn.metrics import fbeta_score
import pandas as pd
from keras.callbacks import ModelCheckpoint

DataGenerator=DataGenerator()
DataGenerator.createCategories()
DataGenerator.loadData()

batch_size=200

df_train = pd.read_csv('train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}

inv_label_map = {i: l for l, i in DataGenerator.categories.items()}
print inv_label_map

def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
  def mf(x):
    p2 = np.zeros_like(p)
    for i in range(17):
      p2[:, i] = (p[:, i] > x[i]).astype(np.int)
    score = fbeta_score(y, p2, beta=2, average='samples')
    return score

  x = [0.2]*17
  for i in range(17):
    best_i2 = 0
    best_score = 0
    for i2 in range(resolution):
      i2 /= float(resolution)
      x[i] = i2
      score = mf(x)
      if score > best_score:
        best_i2 = i2
        best_score = score
    x[i] = best_i2
    if verbose:
      print(i, best_i2, best_score)

  return x

for p in range(10):
	u=p+5
	filepath="weights"+str(u)+".h5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]

	random.shuffle(DataGenerator.outputData)
	generator=DataGenerator.TrainDataGenerate(batch_size, True)
	valGenerator=DataGenerator.ValDataGenerate(batch_size)
	xTest,yTest=DataGenerator.loadValData()

	base_model = InceptionV3(weights='imagenet', include_top=False)
	x = base_model.output
	x = GlobalAveragePooling2D(name='mg1')(x)
	x = BatchNormalization(name='mg3')(x)
	x = Dropout(0.5, name='mg4')(x)
	predictions = Dense(17, activation='sigmoid',name='mg6')(x)
	model = Model(inputs=base_model.input, outputs=predictions)

	for layer in model.layers[:310]:
	    layer.trainable = False
	for layer in model.layers[310:]:
   	    layer.trainable = True

	model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

	model.fit_generator(generator, samples_per_epoch=36000/batch_size,   
		        verbose=1, nb_epoch=2,validation_data=valGenerator,
			 callbacks=callbacks_list, validation_steps=4000/batch_size,max_q_size=1,workers=1)

	for layer in model.layers[:249]:
   	    layer.trainable = False	
	for layer in model.layers[249:]:
   	    layer.trainable = True

	model.compile(lr=0.005,optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

	model.fit_generator(generator, samples_per_epoch=36000/batch_size,   
                        verbose=1, nb_epoch=8,validation_data=valGenerator,
                        validation_steps=4000/batch_size,max_q_size=4,workers=1, callbacks=callbacks_list)
	model.compile(lr=0.0025,optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

	model.fit_generator(generator, samples_per_epoch=36000/batch_size,   
                        verbose=1, nb_epoch=4,validation_data=valGenerator,
                        validation_steps=4000/batch_size,max_q_size=4,workers=1, callbacks=callbacks_list)

	model.compile(lr=0.0005,optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

	model.fit_generator(generator, samples_per_epoch=36000/batch_size,   
                        verbose=1, nb_epoch=2,validation_data=valGenerator,
                        validation_steps=4000/batch_size,max_q_size=4,workers=1, callbacks=callbacks_list)

	predict=model.predict(xTest, batch_size=batch_size)
	optimised_classes=optimise_f2_thresholds(yTest, predict)

	model_yaml = model.to_yaml()
	with open("model.yaml", "w") as yaml_file:
	    yaml_file.write(model_yaml)








a=0
test=[]
for filename in os.listdir('test/test/'):

	newLine=filename[:-4]+','
        for b in range(len(x_final_preds[a])):  
        	if(x_final_preds[a][b]==1):
                	newLine=newLine+' '+inv_label_map[b]
	test.append(newLine)
	a+=1

with open('results99.csv', 'w') as f:
	for line in test:
		f.write(line+'\n')



datagen = ImageDataGenerator(
   rotation_range=90,
   horizontal_flip=True,
   vertical_flip=True,
   rescale=1./255)

nb_steps=61191/200+1

for y in range(1):
	testGenerator = datagen.flow_from_directory(
        	'test',
        	target_size=(256, 256),
        	batch_size=200,
       		shuffle=False)

	x_final_preds=[]
	x_final_preds=model.predict_generator(testGenerator, steps=nb_steps, workers=1)


	for i in x_final_preds:
		for j in range(len(i)):
		        if(i[j]>optimised_classes[j]):
		                i[j]=1
		        else:
		                i[j]=0
	a=0
	test=[]


        print x_final_preds[3]	

	for filename in os.listdir('test/test/'):

		newLine=filename[:-4]+','
		for b in range(len(x_final_preds[a])):  
		        if(x_final_preds[a][b]==1):
		                newLine=newLine+' '+inv_label_map[b]
		test.append(newLine)
		a+=1

	with open('results'+str(y)+'.csv', 'w') as f:
	    for line in test:
		f.write(line+'\n')





