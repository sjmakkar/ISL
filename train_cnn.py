import pickle
from traceback import print_tb

import cv2
import numpy
import numpy as np
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Convolution2D
from keras._tf_keras.keras.layers import MaxPool2D
from keras._tf_keras.keras.layers import Flatten
from keras._tf_keras.keras.layers import Dense,Dropout
import os

from keras.src.models import model
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sz = 128

#load data from pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract features (data) and labels from the data dictionary
data_list = data_dict['data']
labels = data_dict['labels']

# Resize images in the data_list to the target size (128x128)
resized_data = []
for img in data_list:
    img_resized = cv2.resize(np.array(img), (sz, sz))  # Resize image to 128x128
    resized_data.append(img_resized)

# Convert the resized data list into a NumPy array
data = np.array(resized_data)

# Preprocess the data (scale the numerical features)
scaler = StandardScaler()
# Reshape data to be 2D (flatten) for scaling
num_samples = data.shape[0]
data_flattened = data.reshape(num_samples, -1)  # Flatten to (num_samples, 128 * 128)
data_scaled = scaler.fit_transform(data_flattened)

# Reshape the scaled data back into 4D array for CNN input (samples, height, width, channels)
data_reshaped = data_scaled.reshape(num_samples, sz, sz, 1)  # (num_samples, 128, 128, 1)

# Preprocess the labels (convert to categorical for multi-class classification)
labels_categorical = to_categorical(labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data_reshaped, labels_categorical, test_size=0.2, shuffle=True, stratify=labels)
#Step 01: Building the CNN
# Initializing the CNN
classifier = Sequential()
# First convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))
# Second convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPool2D(pool_size=(2, 2)))
#classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
#classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Adding a fully connected layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=96, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=26, activation='softmax')) # softmax for more than 2

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2


# Train the model
history = classifier.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model's accuracy
test_loss, test_accuracy = classifier.evaluate(x_test, y_test)
print('Test accuracy: {:.2f}%'.format(test_accuracy * 100))

# Save the trained model
with open('cnnmodel.p', 'wb') as f:
    pickle.dump({'Classifier': classifier}, f)

