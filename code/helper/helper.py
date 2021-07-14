# helper functions
# don't forget the imports

# function to generate LR images
def crappify(img, scaleBack=True, newSize=40):
  # apply gaussian blur
  img = copy.deepcopy(cv2.GaussianBlur(img, 
                                      (5,5), 
                                      cv2.BORDER_DEFAULT))
  # reduce dimensions
  img = cv2.resize(img, 
                   (newSize, newSize), 
                   interpolation = cv2.INTER_AREA)
  
  # scale back if needed
  if scaleBack == True:
    img = cv2.resize(img, 
                     (80, 80), 
                     interpolation = cv2.INTER_AREA)
  return img
  

# PSNR metric  
def PSNR(y_true, y_pred):
  max_pixel = 1.0
  return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303


# SSIM metric
def SSIM(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))  
