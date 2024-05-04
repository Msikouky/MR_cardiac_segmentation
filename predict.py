import os
import numpy
from skimage.io import imread, imsave
from skimage.util import img_as_float
from skimage.transform import resize
from skimage.io import imshow
from matplotlib import pyplot as plt

import model
import metrics_and_losses
from medpy.metric.binary import asd, hd

DATA_PATH = 'C:/Users/pc/OneDrive/Bureau/Segmentation'

TRAIN_FRAME_PATH = DATA_PATH + '/train_frames/Train'
VAL_FRAME_PATH = DATA_PATH + '/val_frames/Val'
TEST_FRAME_PATH = DATA_PATH + '/test_frames/Test'

TRAIN_MASK_PATH = DATA_PATH + '/train_masks/Train'
VAL_MASK_PATH = DATA_PATH + '/val_masks/Val'
TEST_MASK_PATH = DATA_PATH + '/test_masks/Test'

# Due to image geometrical augmentation, generated images are twice as big as the initial ones
IMG_WIDTH = 256
IMG_HEIGHT = 256

# Load model with optimized weights
m = model.get_UNet(256, 256)
m.summary()
m.load_weights('weights_CE_66_64_0.0258_endo.h5')

# Build a 4D numpy matrix to store all training images from TRAIN_FRAME_PATH
# output = img_train with indices 0 for image number, 1 ordinate, 2 abscissa, 3 channel (0 for gray level images)
train_frames = os.listdir(TRAIN_FRAME_PATH)
img_train = numpy.empty((len(train_frames), IMG_HEIGHT, IMG_WIDTH, 1))
for n, frame in enumerate(train_frames):
    img = imread(TRAIN_FRAME_PATH + '/' + frame)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img_train[n, :, :, 0] = img

# Build a 4D numpy matrix to store all validation images from VAL_FRAME_PATH
# output = img_val with indices 0 for image number, 1 ordinate, 2 abscissa, 3 channel (0 for gray level images)
val_frames = os.listdir(VAL_FRAME_PATH)
img_val = numpy.empty((len(val_frames), IMG_HEIGHT, IMG_WIDTH, 1))
for n, frame in enumerate(val_frames):
    img = imread(VAL_FRAME_PATH + '/' + frame)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img_val[n, :, :, 0] = img

# Build a 4D numpy matrix to store all the test images from TEST_FRAME_PATH
# output = img_test with indices 0 for image number, 1 ordinate, 2 abscissa, 3 channel (0 for gray level images)
test_frames = os.listdir(TEST_FRAME_PATH)
img_test = numpy.empty((len(test_frames), IMG_HEIGHT, IMG_WIDTH, 1))
for n, frame in enumerate(test_frames):
    img = imread(TEST_FRAME_PATH + '/' + frame)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img_test[n, :, :, 0] = img

# Prediction from model m
pred_train = m.predict(img_train, verbose=1) > 0.5
pred_val = m.predict(img_val, verbose=1) > 0.5
pred_test = m.predict(img_test, verbose=1) > 0.5

# Fusion between initial images and predicted masks for display purpose
display_train = pred_train + img_train
display_val = pred_val + img_val
display_test = pred_test + img_test

# Paths to save predicted images
TRAIN_PRED_PATH = DATA_PATH + '/train_preds'
if not os.path.exists(TRAIN_PRED_PATH):
    os.makedirs(TRAIN_PRED_PATH)
VAL_PRED_PATH = DATA_PATH + '/val_preds'
if not os.path.exists(VAL_PRED_PATH):
    os.makedirs(VAL_PRED_PATH)
TEST_PRED_PATH = DATA_PATH + '/test_preds'
if not os.path.exists(TEST_PRED_PATH):
    os.makedirs(TEST_PRED_PATH)

# Save the fusion of training images and corresponding predicted masks to TRAIN_PRED_PATH
# Compute the average Dice coefficient for the training dataset
train_mean_dice = 0
train_mean_asd = 0
train_mean_hd = 0
m = 0
for n, frame in enumerate(train_frames):
    #print("Image #{} -> {}".format(n, frame), flush=True)
    img = 127 * display_train[n, :, :, 0]
    img = img.astype(numpy.uint8)
    imsave(TRAIN_PRED_PATH + '/' + frame, img)
    mask = imread(TRAIN_MASK_PATH + '/' + frame)
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH)).astype(numpy.float64)
    pred = pred_train[n, :, :, 0].astype(numpy.float64)
    if not numpy.amax(pred) == 0:
        dice = metrics_and_losses.dice_coef(mask, pred)
        train_mean_dice += dice
        if not numpy.amax(mask) == 0:
            euclidian = asd(pred, mask)
            train_mean_asd += euclidian
            hausdorff = hd(pred, mask)
            train_mean_hd += hausdorff
            m += 1
train_mean_dice /= m
print("Mean Dice coefficient for training database= {}".format(train_mean_dice))
train_mean_asd /= m
print("Mean euclidian distance for training database= {}".format(train_mean_asd))
train_mean_hd /= m
print("Mean Hausdorff distance for training database= {}".format(train_mean_hd))

# Save the fusion of validation images and corresponding predicted masks to VAL_PRED_PATH
# Compute the average Dice coefficient for the validation dataset
val_mean_dice = 0
val_mean_asd = 0
val_mean_hd = 0
m = 0
for n, frame in enumerate(val_frames):
    #print("Image #{} -> {}".format(n, frame), flush=True)
    img = 127 * display_val[n, :, :, 0]
    img = img.astype(numpy.uint8)
    imsave(VAL_PRED_PATH + '/' + frame, img)
    mask = imread(VAL_MASK_PATH + '/' + frame)
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH))
    pred = pred_val[n, :, :, 0].astype(numpy.float64)
    if not numpy.amax(pred) == 0:
        dice = metrics_and_losses.dice_coef(mask, pred)
        val_mean_dice += dice
        if not numpy.amax(mask) == 0:
            euclidian = asd(pred, mask)
            val_mean_asd += euclidian
            hausdorff = hd(pred, mask)
            val_mean_hd += hausdorff
            m += 1
val_mean_dice /= m
print("Mean Dice coefficient for validation database= {}".format(val_mean_dice))
val_mean_asd /= m
print("Mean euclidian distance for validation database= {}".format(val_mean_asd))
val_mean_hd /= m
print("Mean Hausdorff distance for validation database= {}".format(val_mean_hd))

# Save the fusion of test images and corresponding predicted masks to TEST_PRED_PATH
# Compute the average Dice coefficient for the test dataset
test_mean_dice = 0
test_mean_asd = 0
test_mean_hd = 0
m = 0
for n, frame in enumerate(test_frames):
    #print("Image #{} -> {}".format(n, frame), flush=True)
    img = 127 * display_test[n, :, :, 0]
    img = img.astype(numpy.uint8)
    imsave(TEST_PRED_PATH + '/' + frame, img)
    mask = imread(TEST_MASK_PATH + '/' + frame)
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH))
    pred = pred_test[n, :, :, 0].astype(numpy.float64)
    if not numpy.amax(pred) == 0:
        dice = metrics_and_losses.dice_coef(mask, pred)
        test_mean_dice += dice
        if not numpy.amax(mask) == 0:
            euclidian = asd(pred, mask)
            test_mean_asd += euclidian
            hausdorff = hd(pred, mask)
            test_mean_hd += hausdorff
            m += 1
test_mean_dice /= m
print("Mean Dice coefficient for test database= {}".format(test_mean_dice))
test_mean_asd /= m
print("Mean euclidian distance for test database= {}".format(test_mean_asd))
test_mean_hd /= m
print("Mean Hausdorff distance for test database= {}".format(test_mean_hd))
