import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = Sequential()
#creates the neural network layers and decides the size of the image that it will take.
model.add(Conv2D(64, (9, 9), input_shape=(100,100,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#creates more layers
model.add(Conv2D(128, (5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
#uses binary/it either has glaucoma or it doesn't
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#creates an image crop for the photo
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range=30,
                                   shear_range=0.2,
                                   zoom_range=[0.8, 1.2],
                                    horizontal_flip=True,
                                    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale = 1./255)
#sets the size of the data/number of photos used for training
training_set = train_datagen.flow_from_directory('/content/drive/My Drive/data/train',
                                                 target_size = (100, 100),
                                                 batch_size = 200,
                                                 class_mode = 'binary')
#sets the number of photos used for testing
test_set = test_datagen.flow_from_directory('/content/drive/My Drive/data/val',
                                            target_size = (100, 100),
                                            batch_size = 50,
                                            class_mode = 'binary')
my_callbacks = [
   
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    #invalid code below, will change later
    #tf.keras.callbacks.ModelCheckpoint('my_model2.h5', 
    verbose=1, save_best_only=True, save_weights_only=False) 
    ]

model.fit(training_set, epochs=200, validation_data = test_set, callbacks=my_callbacks)

#invalid code below, will change later
#model.save('my_model2.h5')
