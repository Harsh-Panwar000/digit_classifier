#Import libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#Load the data and reduce the size
data = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()
train_images = train_images/255.0
test_images = test_images/255.0

#Create the architecture and compile the model
model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)), keras.layers.Dense(100 , activation="relu"), keras.layers.Dense(10, activation = "softmax")])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#Fit the model
model.fit(train_images,train_labels, epochs= 10)

#evaluate the model
test_loss, test_acc = model.evaluate(test_images,test_labels)

#Save the model
model.save("digitclassifier.h5")

#Allow user to test any input from test_images and show the input, the prediction, and the image  
predictions = model.predict(test_images)
print("The input was: " + str(test_labels[i]))
print("The prediction is: " + str(np.argmax(predictions[i])))
plt.imshow(test_images[i], cmap= plt.cm.binary)
