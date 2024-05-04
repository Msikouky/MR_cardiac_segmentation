import os
import datetime
import numpy as np
from skimage.io import imshow
from matplotlib import pyplot as plt

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.optimizers import Adam
from keras.metrics import MeanIoU
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping

import model
import metrics_and_losses

DATA_PATH = 'C:/Users/pc/OneDrive/Bureau/Segmentation'
TRAIN_FRAME_PATH = DATA_PATH + '/train_frames'
VAL_FRAME_PATH = DATA_PATH + '/val_frames'
TRAIN_MASK_PATH = DATA_PATH + '/train_masks'
VAL_MASK_PATH = DATA_PATH + '/val_masks'

##########################################
# Image generation and augmentation part #
##########################################

# Instances of ImageDataGenerator for training and validation datasets 
# Geometrical data augmentation is only considered for the training dataset
# Image gray levels are normalized between 0 and 1
# outputs = train_datagen and val_datagen
train_datagen = ImageDataGenerator(
    width_shift_range = 0.1,
    height_shift_range = 0.05,
    zoom_range = 0.2,
    rescale=1./255,
    rotation_range=10
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

# Build image generators for training / validation images and masks
# using the flow_from_directory Keras subroutine
BATCH_SIZE = 64  # Power of two: 4/8/16/32/64

# The images must be found in a subdir of TRAIN_FRAME_PATH for instance /Train
# Output = train_image_generator (images from TRAIN_FRAME_PATH)
train_image_generator = train_datagen.flow_from_directory(TRAIN_FRAME_PATH,
    batch_size=BATCH_SIZE, color_mode='grayscale', class_mode = None, seed = 42)

# The images must be found in a subdir of TRAIN_MASK_PATH for instance /Train
# Use the same seed as training images so that augmentation is the same
# Output = train_mask_generator (images from TRAIN_MASK_PATH)
train_mask_generator = train_datagen.flow_from_directory(TRAIN_MASK_PATH, 
    batch_size=BATCH_SIZE, color_mode='grayscale', class_mode = None, seed = 42)

# The images must be found in a subdir of VAL_FRAME_PATH for instance /Val
# Output = val_image_generator (images from VAL_FRAME_PATH)
val_image_generator = val_datagen.flow_from_directory(VAL_FRAME_PATH, 
    batch_size=BATCH_SIZE, color_mode='grayscale', class_mode = None, seed = 63)

# The images must be found in a subdir of VAL_MASK_PATH for instance /Val
# Use the same seed as validation images so that aumentation is the same
# Output = val_mask_generator (images from VAL_MASK_PATH)
val_mask_generator = val_datagen.flow_from_directory(VAL_MASK_PATH, 
    batch_size=BATCH_SIZE, color_mode='grayscale', class_mode = None, seed = 63)  

train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)

# Display the first generated image for the four image generators
# img = np.squeeze(train_image_generator.next()[0])
# imshow(img)
# plt.show()
# img = np.squeeze(train_mask_generator.next()[0])
# imshow(img)
# plt.show()
# img = np.squeeze(val_image_generator.next()[0])
# imshow(img)
# plt.show()
# img = np.squeeze(val_mask_generator.next()[0])
# imshow(img)
# plt.show()

img = load_img("C:/Users/pc/OneDrive/Bureau/Segmentation/train_frames/Train/11_0.538462_0.413793.png")
x = img_to_array(img)
x = x.reshape((1,) + x.shape)
#Générer des images et les afficher
fig, ax = plt.subplots(1, 5, figsize=(20, 4))
ax[0].imshow(img)
ax[0].set_title('Original Image')

i = 1
for batch in train_datagen.flow(x, batch_size=1):
    ax[i].imshow(np.squeeze(batch))
    ax[i].set_title(f'Augmented {i}')
    i += 1
    if i % 5 == 0:
        break

plt.show()
#################
# Training part #
#################

NO_OF_TRAINING_IMAGES = len(os.listdir(TRAIN_FRAME_PATH + '/Train'))
NO_OF_VAL_IMAGES = len(os.listdir(VAL_FRAME_PATH + '/Val'))

NO_OF_EPOCHS = 100 # Between 30 and 100 for small or medium datasets

# Load the CNN model
m = model.get_UNet(256, 256)
m.summary()

# Optimizer
adam = Adam(learning_rate=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# Compile the model with the chosen optimizer, loss function and monitoring metrics
# Choices for loss function are Dice coefficient or binary cross-entropy 
#m.compile(optimizer=opt, loss='binary_crossentropy', metrics=[MeanIoU(num_classes=1), #metrics_and_losses.dice_coef])
m.compile(optimizer=adam, loss='binary_crossentropy', metrics=[metrics_and_losses.dice_coef])

# Parameters for optimization
# Model weights are saved to weights.h5 when validation loss is improved
# Loss and metrics for training and validation datasets are stored in log.out at every epochs
# Optimization is stopped if there is no improvement in validation loss for at least 10 epochs
weights_path = './weights_CE_66_64_0.0258_endo.h5'
checkpoint = ModelCheckpoint(weights_path, verbose=1, save_best_only=True)
csv_logger = CSVLogger('./log.out', append=True, separator=';')
earlystopping = EarlyStopping(verbose = 1, patience = 10)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
callbacks_list = [checkpoint, csv_logger, earlystopping, tensorboard_callback]

# Training of model m using previously defined generators
results = m.fit_generator(train_generator, epochs=NO_OF_EPOCHS, steps_per_epoch=NO_OF_TRAINING_IMAGES//BATCH_SIZE, validation_data=val_generator, validation_steps=NO_OF_VAL_IMAGES//BATCH_SIZE, callbacks=callbacks_list)

# Final model weights are saved
m.save_weights('Final_weights.h5')
