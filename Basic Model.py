import os
from PIL import Image
from search_videos_in_directory import search_videos
from tensorflow.keras.preprocessing.image import ImageDataGenerator

input_direc = '/home/debanik/downloaded_videos'
output_direc = '/home/debanik/PycharmProjects/Detecting-Deepfakes/Face_images_100'
#
# input_direc = input("Enter the absolute path of the input directory: ")
# output_direc = input("Enter the absolute path of the output directory: ")
search_videos(input_direc, output_direc)


for root, dirs, files in os.walk(output_direc):
    for file in files:

        try:
            im = Image.open(root + '/' + file)
        except IOError:
            print(str(file))
            os.remove(root + '/' + file)

# create generator
datagen = ImageDataGenerator(samplewise_center=True, 
                             samplewise_std_normalization=True, 
                             rotation_range=10, 
                             horizontal_flip = True, 
                             validation_split=0.2)
# prepare an iterators for each dataset
training_images = datagen.flow_from_directory(output_direc, class_mode='binary', batch_size=64, subset='training')
validation_images = datagen.flow_from_directory(output_direc, class_mode='binary', batch_size=64, subset='validation')

import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.models import Model, Sequential


model = Sequential()
model.add(layers.Input(shape = (256, 256, 3)))

model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))
model.add(layers.AveragePooling2D())

model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(layers.AveragePooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(units=120, activation='relu'))

model.add(layers.Dense(units=84, activation='relu'))

model.add(layers.Dense(units=1, activation = 'sigmoid'))

model.summary()


model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
model.fit_generator(training_images, validation_data = validation_images, epochs = 10)
