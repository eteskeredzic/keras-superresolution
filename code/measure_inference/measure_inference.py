# dont forget to move the saved models into this folder before running



import time
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import numpy as np
import cv2 

img80 = cv2.imread('lr80.bmp')
img80 = np.asarray(img80)
img80 = img80.astype(np.float32)
img80 /= 255.0


img40 = cv2.imread('lr40.bmp')
img40 = np.asarray(img40)
img40 = img40.astype(np.float32)
img40 /= 255.0

img20 = cv2.imread('lr20.bmp')
img20 = np.asarray(img20)
img20 = img20.astype(np.float32)
img20 /= 255.0

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303

def SSIM(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0)) 

fsrcnn2x = keras.models.load_model('./fsrcnn2x.hdf5', custom_objects={"PSNR": PSNR, "SSIM": SSIM})
fsrcnn4x = keras.models.load_model('./fsrcnn4x.hdf5', custom_objects={"PSNR": PSNR, "SSIM": SSIM})

ednet2x = keras.models.load_model('./ednet2x.hdf5', custom_objects={"PSNR": PSNR, "SSIM": SSIM})
ednet4x = keras.models.load_model('./ednet4x.hdf5', custom_objects={"PSNR": PSNR, "SSIM": SSIM})

srednet2x = keras.models.load_model('./srednet2x.hdf5', custom_objects={"PSNR": PSNR, "SSIM": SSIM})
srednet4x = keras.models.load_model('./srednet4x.hdf5', custom_objects={"PSNR": PSNR, "SSIM": SSIM})

mslapsrn2x = keras.models.load_model('./mslapsrn2x.hdf5', custom_objects={"PSNR": PSNR, "SSIM": SSIM})
mslapsrn4x = keras.models.load_model('./mslapsrn4x.hdf5', custom_objects={"PSNR": PSNR, "SSIM": SSIM})


l40= list()
l40.append(img40)
l40 = np.asarray(l40)


l20 = list()
l20.append(img20)
l20 = np.asarray(l20)


l80= list()
l80.append(img80)
l80 = np.asarray(l80)

# we dont measure the first inference, because of additional operations that
# python (TF) performs when you run predict for the first time
fsrcnn2x.predict(l40)
sum = 0
for i in range(100):
    start = time.time()
    fsrcnn2x.predict(l40)
    end = time.time()
    sum += end-start

print('FSRCNN2x = {}'.format(sum/100.00))


fsrcnn4x.predict(l20)
sum = 0
for i in range(100):
    start = time.time()
    fsrcnn4x.predict(l20)
    end = time.time()
    sum += end-start

print('FSRCNN4x = {}'.format(sum/100.00))


ednet2x.predict(l80)
sum = 0
for i in range(100):
    start = time.time()
    ednet2x.predict(l80)
    end = time.time()
    sum += end-start

print('EDNET2x = {}'.format(sum/100.00))


ednet4x.predict(l80)
sum = 0
for i in range(100):
    start = time.time()
    ednet4x.predict(l80)
    end = time.time()
    sum += end-start

print('EDNET4x = {}'.format(sum/100.00))


srednet2x.predict(l80)
sum = 0
for i in range(100):
    start = time.time()
    srednet2x.predict(l80)
    end = time.time()
    sum += end-start

print('SREDNET2x = {}'.format(sum/100.00))


srednet4x.predict(l80)
sum = 0
for i in range(100):
    start = time.time()
    srednet4x.predict(l80)
    end = time.time()
    sum += end-start

print('SREDNET4x = {}'.format(sum/100.00))


mslapsrn2x.predict(l40)
sum = 0
for i in range(100):
    start = time.time()
    mslapsrn2x.predict(l40)
    end = time.time()
    sum += end-start

print('MSLAPSRN2x = {}'.format(sum/100.00))


mslapsrn4x.predict(l20)
sum = 0
for i in range(100):
    start = time.time()
    mslapsrn4x.predict(l20)
    end = time.time()
    sum += end-start

print('MSLAPSRN4x = {}'.format(sum/100.00))
