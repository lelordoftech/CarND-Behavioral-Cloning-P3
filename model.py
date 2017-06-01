########################## Behavioral Cloning Project ##########################
__author__ = "Vuong Le"
__email__ = "lelordoftech@gmail.com"
__date__ = "01-Jun-2017"
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
#from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
from keras import __version__ as keras_version
from keras.layers.advanced_activations import ELU
from keras.callbacks import TensorBoard

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import collections
import datetime


keras_ver = str(keras_version).encode('utf8')
data_paths = ['./data_sample/', \
                './data_clockwise/', \
                './data_anti_clockwise/', \
                './data_recovery/', \
                './data_curve_clockwise/', \
                './data_curve_anti_clockwise/']

model_file_path = 'model.h5'
IMG_COLS = 200
IMG_ROWS = 66
IMG_CH = 3

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

def visualation_data(samples):
    print('Start Visualation data')
    start = datetime.datetime.now()

    sample_values_lines = [x[1] for x in samples] # get all lines
    sample_values = [float(x[3]) for x in sample_values_lines] # get all angle in lines))

    visual = collections.Counter(sample_values)
    ord_dict = collections.OrderedDict(sorted(visual.items()))
    labels = ord_dict.keys()
    values = ord_dict.values()

    indexes_bar = np.arange(len(labels))
    step_labels = 0.1
    step_indexes = len(labels)*step_labels/(max(labels) - min(labels))
    new_labels = np.arange(min(labels), max(labels), step_labels)
    indexes = np.arange(0, len(labels), step_indexes)
    width = 1

    fig = plt.figure()
    plt.bar(indexes_bar, values, width)
    plt.xticks(indexes + width * 0.5, labels, rotation=45)
    plt.ylabel('Occurrence')
    plt.xlabel('Steering angle')
    plt.title('Training examples Visualization')
    plt.minorticks_off()
    plt.tight_layout()
    plt.savefig('./report/training_visualization.png')

    end = datetime.datetime.now()
    diff = end - start
    elapsed_ms = (diff.days * 86400000) + (diff.seconds * 1000) + (diff.microseconds / 1000)
    print('Finish Visualation data in %d ms' %elapsed_ms)

def pre_processing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width, channels = img.shape
    crop_img = img[int(height/4):height-25, 0:width]
    scale_img = cv2.resize(crop_img, (IMG_COLS, IMG_ROWS))

    return scale_img

def getImage(batch_data_path, batch_data_full_path):
    # support Linux and Windows
    img_filename = (batch_data_full_path.split('/')[-1]).split('\\')[-1]
    img = cv2.imread(batch_data_path + 'IMG/' + img_filename)
    img = pre_processing(img)

    return img

# input  1 x batch_size
# output 6 x batch_size
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
                correction = 0.24
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
                augmented_angles.append(angle*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)

            yield shuffle(X_train, y_train)

def createNewModel():
    print('Create New Model')

    model = Sequential()
    # Normalization
    model.add(Lambda(lambda x: x / 255.0 - 0.5,
                    input_shape=(IMG_ROWS,IMG_COLS,IMG_CH)))
    # Cropping
    print(model.output_shape)
    # 66x200@3

    # NVIDIA Architecture
    if (keras_ver == b'1.2.1'):
        model.add(Convolution2D(24,5,5,subsample=(2,2)))
        model.add(ELU())
        model.add(Dropout(0.2))
        print(model.output_shape) # 31x98@24

        model.add(Convolution2D(36,5,5,subsample=(2,2)))
        model.add(ELU())
        model.add(Dropout(0.2))
        print(model.output_shape) # 14x47@36

        model.add(Convolution2D(48,5,5,subsample=(2,2)))
        model.add(ELU())
        model.add(Dropout(0.2))
        print(model.output_shape) # 5x22@48

        model.add(Convolution2D(64,3,3))
        model.add(ELU())
        model.add(Dropout(0.2))
        print(model.output_shape) # 3x20@64

        model.add(Convolution2D(64,3,3))
        model.add(ELU())
        model.add(Dropout(0.2))
        print(model.output_shape) # 1x18@64
    elif (keras_ver == b'2.0.4'):
        '''
        default:
        Conv2D(strides=(1, 1), 
                padding='valid', 
                activation=None)
        advanced_activations:
        ELU()
        '''
        model.add(Convolution2D(filters=24,kernel_size=(5,5),strides=2))
        model.add(ELU())
        model.add(Dropout(0.3))
        print(model.output_shape) # 31x98@24

        model.add(Convolution2D(filters=36,kernel_size=(5,5),strides=2))
        model.add(ELU())
        model.add(Dropout(0.3))
        print(model.output_shape) # 14x47@36

        model.add(Convolution2D(filters=48,kernel_size=(5,5),strides=2))
        model.add(ELU())
        model.add(Dropout(0.3))
        print(model.output_shape) # 5x22@48

        model.add(Convolution2D(filters=64,kernel_size=(3,3)))
        model.add(ELU())
        model.add(Dropout(0.2))
        print(model.output_shape) # 3x20@64

        model.add(Convolution2D(filters=64,kernel_size=(3,3)))
        model.add(ELU())
        model.add(Dropout(0.2))
        print(model.output_shape) # 1x18@64

    model.add(Flatten())
    print(model.output_shape) # 1152

    # default: activation=None
    model.add(Dense(100))
    model.add(ELU())
    model.add(Dropout(0.5))
    print(model.output_shape)

    model.add(Dense(50))
    model.add(ELU())
    model.add(Dropout(0.4))
    print(model.output_shape)

    model.add(Dense(10))
    model.add(ELU())
    model.add(Dropout(0.3))
    print(model.output_shape)

    model.add(Dense(1))
    model.add(ELU())
    print(model.output_shape)

    model.compile(loss='mse', optimizer='adam')

    # 27 million connections and 250 thousand parameters.
    return model

def main():
    model = None
    batch_size_val = 132
    num_epochs = 5

    ### Read all driving log from all folder
    samples = getAllSamples(data_paths)

    ### Visualization data
    visualation_data(samples)

    ### Create training samples and validation samples
    train_samples, validation_samples = train_test_split(samples, test_size=0.3)

    ### Create training set and validation set
    train_generator = generator(train_samples, batch_size=int(batch_size_val/6))
    validation_generator = generator(validation_samples, batch_size=int(batch_size_val/6))

    ### Define model
    if os.path.exists(model_file_path) == True: # Load old model
        print('Load model ' + model_file_path)
        model = load_model(model_file_path)
    else: # Create new model
        model = createNewModel()
    # Summary representation of the model.
    model.summary()

    ### Train the model using the generator function
    tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)

    if (keras_ver == b'1.2.1'):
        history_object = model.fit_generator(train_generator, \
                    #samples_per_epoch=int(len(train_samples)/batch_size_val)+1, \
                    samples_per_epoch=len(train_samples), \
                    nb_epoch=num_epochs, \
                    verbose=1, \
                    validation_data=validation_generator, \
                    #nb_val_samples=int(len(validation_samples)/batch_size_val)+1)
                    nb_val_samples=len(validation_samples))
    elif (keras_ver == b'2.0.4'):
        history_object = model.fit_generator(train_generator, \
                    steps_per_epoch=int(len(train_samples)/batch_size_val)+1, \
                    epochs=num_epochs, \
                    verbose=1, \
                    validation_data=validation_generator, \
                    validation_steps=int(len(validation_samples)/batch_size_val)+1, \
                    callbacks=[tbCallBack])

    ### Save model
    model.save(model_file_path)

    ### print the keys contained in the history object
    #print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    fig = plt.figure()
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')

    ### save data loss
    fig = plt.gcf()
    fig.savefig('./report/model_mean_squared_error_loss.png')
    #plt.show()

    ### Exit
    exit()

if __name__ == '__main__':
    main()
