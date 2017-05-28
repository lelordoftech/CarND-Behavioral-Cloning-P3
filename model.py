################################################################################
# Behavioral Cloning Project
__author__ = "Vuong Le"
__email__ = "lelordoftech@gmail.com"
__date__ = "28-May-2017"
################################################################################


import csv
import os.path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
import matplotlib.pyplot as plt


data_paths = ['./data_sample/', './data1/', './data2/', './data3/']
model_file_path = 'model.h5'


def getAllSamples(data_paths):
    print('Get All Samples')

    samples = []
    for data_path in data_paths:
        print('    From data path ' + data_path)
        csv_file_path = data_path + 'driving_log.csv'
        if os.path.exists(csv_file_path) == True:
            with open(csv_file_path) as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    if line[3] == 'steering':
                        continue # skip first line
                    samples.append([data_path, line])
        else:
            print('[ERROR] Does not exist file' + csv_file_path)

    print('Total samples: ' + str(len(samples)))

    return samples

def getImage(batch_data_path, batch_data_full_path):
    # support Linux and Windows
    img_filename = (batch_data_full_path.split('/')[-1]).split('\\')[-1]
    img = cv2.imread(batch_data_path + 'IMG/' + img_filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def generator(samples, batch_size=32):
    num_samples = len(samples)

    while 1:
        shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            images = []
            angles = []
            batch_samples = samples[offset:offset+batch_size]

            for batch_sample in batch_samples:
                batch_data_path = batch_sample[0]
                batch_line = batch_sample[1]

                # read in images from center, left and right cameras
                img_center = getImage(batch_data_path, batch_line[0])
                img_left = getImage(batch_data_path, batch_line[1])
                img_right = getImage(batch_data_path, batch_line[2])

                # read steering angle
                steering_center = float(batch_line[3])
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
                # append original data
                augmented_images.append(image)
                augmented_angles.append(angle)
                # append augmented data
                augmented_images.append(cv2.flip(image,1))
                #np.fliplr(image)
                augmented_angles.append(angle*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)

            yield shuffle(X_train, y_train)

def createNewModel():
    print('Create New Model')

    model = Sequential()
    # Normalization
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    # Cropping
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    # NVIDIA Architecture
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
    model.add(Dropout(0.3))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    model.add(Dropout(0.3))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    model.add(Dropout(0.3))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

def main():
    model = None

    ### Read all driving log from all folder
    samples = getAllSamples(data_paths)

    ### Create training samples and validation samples
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    ### Create training set and validation set
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    ### Define model
    if os.path.exists(model_file_path) == True: # Load old model
        print('Load model ' + model_file_path)
        model = load_model(model_file_path)
    else: # Create new model
        model = createNewModel()
    # Summary representation of the model.
    model.summary()

    ### Train the model using the generator function
    history_object = model.fit_generator(train_generator, \
        samples_per_epoch=len(train_samples), \
        validation_data=validation_generator, \
        nb_val_samples=len(validation_samples), \
        nb_epoch=5, \
        verbose=1)

    ### Save model
    model.save(model_file_path)

    ### print the keys contained in the history object
    #print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')

    ### save data loss
    fig = plt.gcf()
    fig.savefig('model_mean_squared_error_loss.png')
    #plt.show()

    ### Exit
    exit()

if __name__ == '__main__':
    main()
