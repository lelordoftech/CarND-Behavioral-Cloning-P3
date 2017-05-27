import csv

data_path = './data_sample/'

samples = []
with open(data_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if line[3] == 'steering':
            continue # skip first line
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # read in images from center, left and right cameras
                img_center_filename = (line[0].split('/')[-1]).split('\\')[-1] # support Linux and Windows
                img_center = cv2.imread(data_path + 'IMG/' + img_center_filename)

                img_left_filename = (line[1].split('/')[-1]).split('\\')[-1]
                img_left = cv2.imread(data_path + 'IMG/' + img_left_filename)

                img_right_filename = (line[2].split('/')[-1]).split('\\')[-1]
                img_right = cv2.imread(data_path + 'IMG/' + img_right_filename)

                # read steering angle
                steering_center = float(line[3])

                correction = 0.2
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # add image and angles to data set
                images.append(img_center)
                images.append(img_left)
                images.append(img_right)

                angles.append(steering_center)
                angles.append(steering_left)
                angles.append(steering_right)

            # Data Augmentation
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1)) #image_flipped = np.fliplr(image)
                augmented_angles.append(angle*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)

            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
import os.path

model_file_path = 'model.h5'
model = None
if os.path.exists(model_file_path) == True: # Load old model
    model = load_model(model_file_path)
else: # Create new model
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    ### Summary representation of the model.
    model.summary()

history_object = model.fit_generator(train_generator, \
        samples_per_epoch=len(train_samples), \
        validation_data=validation_generator, \
        nb_val_samples=len(validation_samples), \
        nb_epoch=5, \
        verbose=1)

model.save(model_file_path)

### print the keys contained in the history object
print(history_object.history.keys())

import matplotlib.pyplot as plt

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
# save data loss
fig = plt.gcf()
fig.savefig('model_mean_squared_error_loss.png')
#plt.show()

exit()
