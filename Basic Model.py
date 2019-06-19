import os
from PIL import Image
from search_videos_in_directory import search_videos
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# input_direc = '/home/debanik/downloaded_videos'
# output_direc = '/home/debanik/PycharmProjects/Detecting-Deepfakes/temp_n'

input_direc = input("Enter the absolute path of the input directory: ")
output_direc = input("Enter the absolute path of the output directory: ")
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
# # confirm the iterator works
# batchX, batchy = train_it.next()
# print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))


from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.models import Model


x = Input(shape = (256, 256, 3))

x1 = Conv2D(160, (3, 3), strides = 1, padding='same', activation = 'relu')(x)
x1 = Conv2D(40, (1, 1), padding='same', activation = 'relu')(x1)
x1 = BatchNormalization()(x1)
x1 = MaxPooling2D(pool_size=(8, 8), padding='same')(x1)

y = Flatten()(x1)
y = Dropout(0.5)(y)
y = Dense(100, activation='relu')(y)
y = Dense(1, activation='sigmoid')(y)


model = Model(inputs = x, outputs = y)

model.summary()


model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
model.fit_generator(training_images, validation_data = validation_images, epochs = 10)
