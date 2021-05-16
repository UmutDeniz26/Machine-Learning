# -*- coding: utf-8 -*-
"""
Created on Sun May 15 17:41:16 2021

@author: Umut
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(trainImg, trainLbl), (testImg, testLbl) = mnist.load_data()

class_names = ['0','1', '2', '3', '4', '5',
               '6', '7', '8', '9']
i=0
plt.figure(figsize=(14,10))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(trainImg[i],cmap='gray')
    plt.xlabel(trainLbl[i])
plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(trainImg, trainLbl, epochs=10)
test_loss, test_acc = model.evaluate(testImg,  testLbl, verbose=2)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(testImg)

print('\nTest accuracy:', test_acc)
i=0
j=0
plt.figure(figsize=(14,10))
try: 
   for i in range(1000):
        if(testLbl[i]!=np.argmax(predictions[i])):
            plt.subplot(5,5,j+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(testImg[i])
            plt.xlabel("pred: "+str(np.argmax(predictions[i]))+" true: "+str(testLbl[i]))
            j=j+1
   plt.show()
except:
    None




