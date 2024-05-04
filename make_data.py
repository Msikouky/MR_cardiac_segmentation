import os
import random
import re
from skimage.io import imread, imshow, imsave
from skimage.transform import resize

DATA_PATH = 'C:/Users/pc/OneDrive/Bureau/Segmentation'
FRAME_PATH = DATA_PATH + '/frames'
MASK_PATH = DATA_PATH + '/masks'

# Min and max values for slice level and time
zmin = 0.1
zmax = 0.9
tmin = 0.0
tmax = 1.0

# Image size
IMG_WIDTH = 128
IMG_HEIGHT = 128

# Create folders to hold images and masks
folders = ['train_frames/Train', 'train_masks/Train', 'val_frames/Val', 'val_masks/Val', 'test_frames/Test', 'test_masks/Test']

for folder in folders:
    if not os.path.exists(DATA_PATH + '/' + folder):
        os.makedirs(DATA_PATH + '/' + folder)
  
# Get all frames and masks file names
all_frames = os.listdir(FRAME_PATH)
all_masks = os.listdir(MASK_PATH)

# Select masks in the expected range of z and t using Python regular expression
# You may use https://regex101.com/ website to test your regular expression
# output = list of filenames selected_masks
regex = "\d\.\d+"
selected_masks = []
for mask in all_masks:
  match = re.findall(regex, mask)
  z = float(match[0])
  t = float(match[1])
  if z > zmin and z < zmax and t > tmin and t < tmax:
    selected_masks.append(mask)

# Randomly shuffle the selected mask list
random.shuffle(selected_masks)

# Generate lists of mask filenames for train, val, and test sets
# train list corresponds to the first 70%
# val list corresponds to the next 20%
# test list corresponds to the last 10%
end_train = int(0.7 * len(selected_masks))
end_val = int(0.9 * len(selected_masks))
train = selected_masks[:end_train]
val = selected_masks[end_train:end_val]
test = selected_masks[end_val:]

# Add train, val, test frames and masks to relevant folders
frame_folders = [(train, 'train_frames/Train'), (val, 'val_frames/Val'), 
                 (test, 'test_frames/Test')]

mask_folders = [(train, 'train_masks/Train'), (val, 'val_masks/Val'), 
                (test, 'test_masks/Test')]
               
# Subroutine that reads image from FRAME_PATH,
# resize it to IMG_HEIGHT x IMG_WIDTH
# and stores it to dir_name subdirectory of DATA_PATH
def add_frames(dir_name, image):
  array = imread(FRAME_PATH + '/' + image)
  array = resize(array, (IMG_HEIGHT, IMG_WIDTH))
  imsave(DATA_PATH + '/' + dir_name + '/' + image, array)
  
# Subroutine that reads mask from MASK_PATH,
# resize it to IMG_HEIGHT x IMG_WIDTH
# and stores it to dir_name subdirectory of DATA_PATH
def add_masks(dir_name, mask):
  array = imread(MASK_PATH + '/' + mask)
  array = resize(array, (IMG_HEIGHT, IMG_WIDTH))
  imsave(DATA_PATH + '/' + dir_name + '/' + mask, array)

# Add frames
for folder in frame_folders:
  array = folder[0]
  name = [folder[1]] * len(array)
  list(map(add_frames, name, array))   

# Add masks
for folder in mask_folders:
  array = folder[0]
  name = [folder[1]] * len(array)
  list(map(add_masks, name, array))
