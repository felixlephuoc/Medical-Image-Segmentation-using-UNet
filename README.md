
# Finding Nuclei in Divergent Images using U-Net


## Outline
* **Problem description**
* **Example: Stack all masks images for one single training image**
* **Creating our image masks of dimension 128 x 128 (black images)**
* **Visualize random training images and mask**
* **Define a custrom Metric called Intersection over Union (IoU)**
* **Build U-Net Model**
* **Train model**
* **Generating predictions for training and validation data**
* **Generating a Classification Report**

## Problem Description

This is Part of Kaggle's 2018 Data Sciene Bowl: https://www.kaggle.com/c/data-science-bowl-2018

#### Spot Nuclei. Speed Cures.

Imagine speeding up research for almost every disease, from lung cancer and heart disease to rare disorders. The 2018 Data Science Bowl offers our most ambitious mission yet: create an algorithm to automate nucleus detection.

We’ve all seen people suffer from diseases like cancer, heart disease, chronic obstructive pulmonary disease, Alzheimer’s, and diabetes. Many have seen their loved ones pass away. Think how many lives would be transformed if cures came faster.

By automating nucleus detection, you could help unlock cures faster—from rare disorders to the common cold. Want a snapshot about the 2018 Data Science Bowl? [View this video.](https://www.youtube.com/watch?v=eHwkfhmJexs&feature=youtu.be)

- Alot of the work in this notework was provided by Kjetil Åmdal-SævikKeras who made the "U-Net starter - LB 0.277" the top rated Kernel on Kaggle.
- https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277


```python
import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = './U_NET/train/'
TEST_PATH = './U_NET/validation/'

# warnings.filterwarnings('ignore', catergory=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed
```

    Using TensorFlow backend.
    /home/deeplearningcv/anaconda3/envs/cv/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:523: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /home/deeplearningcv/anaconda3/envs/cv/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:524: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /home/deeplearningcv/anaconda3/envs/cv/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /home/deeplearningcv/anaconda3/envs/cv/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:526: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /home/deeplearningcv/anaconda3/envs/cv/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:527: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /home/deeplearningcv/anaconda3/envs/cv/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:532: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])


**Collect our file names for training and test date**


```python
train_ids = next(os.walk(TRAIN_PATH))[1] # return the list of names of all subfolder in train/validation path
test_ids = next(os.walk(TEST_PATH))[1]
```

## Example: Stack all masks images for one single training image


```python
%matplotlib inline
import cv2
from matplotlib import pyplot as plt

mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
path = './U_NET/train/0acd2c223d300ea55d0546797713851e818e5c697d073b7f4091b96ce0f3d2fe/masks/'
for i, mask_file in enumerate(next(os.walk(path))[2]):
    mask_ = cv2.imread(path + mask_file)
    mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True)
    plt.imshow(mask_)
    plt.suptitle("Individual mask no. " + str(i), fontsize=14)
    plt.show()
    #print(mask_.shape)
    #mask_ = np.expand_dims(mask_, axis=-1)
    #print(mask_.shape)
    mask = np.maximum(mask, mask_)
    #print(mask)
final_mask = mask
final_mask.shape
plt.imshow(final_mask)
plt.suptitle("Final Mask Combination", fontsize=18, fontweight='bold')
plt.show()
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).



![png](19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_files/19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_7_1.png)


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).



![png](19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_files/19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_7_3.png)


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).



![png](19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_files/19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_7_5.png)


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).



![png](19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_files/19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_7_7.png)


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).



![png](19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_files/19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_7_9.png)


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).



![png](19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_files/19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_7_11.png)


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).



![png](19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_files/19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_7_13.png)


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).



![png](19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_files/19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_7_15.png)


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).



![png](19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_files/19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_7_17.png)


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).



![png](19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_files/19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_7_19.png)


## Creating our image masks of dimension 128 x 128 (black images)

Stack all the mask images in subfolder to one single mask image

Note that we only have mask images in training folder. In validation folder we only have original image, which we need to predict mask on it.


```python
print('Getting and resizing training images ... ')

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)


# Re-sizing our training images to 128 x 128
# Note sys.stdout prints info that can be cleared unlike print.
# Using TQDM allows us to create progress bars
sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH,1), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH,1), dtype=np.bool) # Create a matrix with all False value
    
    #  Now we take all masks associated with that image and combine them into one single mask
    for mask_file in next(os.walk(path + '/masks/'))[2]: # return all mask images names in /mask/ folder
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        mask_ = np.expand_dims(mask_, axis=-1)
        mask = np.maximum(mask, mask_) # this line of code merge all the mask images by selecting the 'maximum' value
    # Y_train is now our single mask associated with our image
    Y_train[n] = mask
    
# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
size_test = []
print('Getting and resizing test images ...')
sys.stdout.flush()

# here we resize our test images
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    size_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode = 'constant', preserve_range=True)
    X_test[n] = img
    # Not create mask (Y_test) at this point
    
print("Done!")
```

    Getting and resizing training images ... 


    100%|██████████| 670/670 [01:03<00:00, 17.02it/s]

    Getting and resizing test images ...


    
    100%|██████████| 65/65 [00:00<00:00, 110.12it/s]

    Done!


    



```python
print("Number of training samples: ", X_train.shape[0])
print("Number of testing samples: ", X_test.shape[0])
print("Number of masks using as training labels: ", Y_train.shape[0])

print("Shape of training image: ", X_train.shape)
print("Shape of training mask: ", Y_train.shape)
```

    Number of training samples:  670
    Number of testing samples:  65
    Number of masks using as training labels:  670
    Shape of training image:  (670, 128, 128, 3)
    Shape of training mask:  (670, 128, 128, 1)


## Visualize random training images and mask


```python
## Illustrate the train images and masks
plt.figure(figsize=(20,16)) # width of 20 inches and 16 inches of height
x,y = 12, 4 # 12 columns, x to use for 8 rows
for i in range(y):
    for j in range(x):
        # display original image at odd rows
        plt.subplot(y*2, x, i*2*x + j + 1) #plt.subplot(8, 12, [])
        pos = i*120 + j*20
        plt.imshow(X_train[pos])
        plt.title('Image #{}'.format(pos))
        plt.axis('off')
        
        # Display the mask image in the even rows
        plt.subplot(y*2, x, (i*2+1)*x + j +1)
        plt.imshow(np.squeeze(Y_train[pos])) # y_train is at shape (670, 128, 128, 1), need to squeeze to remove last diementions
        plt.title('Mask #{}'.format(pos))
        plt.axis('off')
        
plt.show()       

```


![png](19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_files/19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_12_0.png)


## Define a custrom Metric called Intersection over Union (IoU)


```python
# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2, y_true)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
```


### Custom IoU

#### Another IoU Metric  : https://www.kaggle.com/aglotero/another-iou-metric


```python
def iou_metric(y_true_in, y_pred_in, print_table=False):
    # Binary image with pixels inside convex hull set to True . 
    # This function uses skimage. morphology. label to define unique objects,
    # finds the convex hull of each using convex_hull_image , and combines these regions with logical OR.
    labels = label(y_true_in > 0.5)
    y_pred = label(y_pred_in > 0.5)
    
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))
    
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]
    
    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    
    # Commute union
    union = area_true + area_pred - intersection
    

    # Exclude backgound from analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9
    
    # Compute intersection over union
    iou = intersection / union
    
    # Precision helper function: compare iou with threshold to determine precision
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1 # correct objects
        false_positives = np.sum(matches, axis=0) == 0 # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0 # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn
    
    # Loop over thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp /(tp + fp + fn) # customzied precision, not the same as in classification 
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
        
    if print_table:
            print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)
    
def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch]) # Compute precision by batch
        metric.append(value)
    return np.array(np.mean(metric), dtype=np.float32)

def my_iou_metric(label, pred):
    # Using tf.py_func we can use a python function inside Tensorflow environment.
    metric_value = tf.py_func(iou_metric_batch, [label, pred], tf.float32) # ******
    return metric_value
```

## Build U-Net Model


```python
# Build U-Net model
# Note we make our layers varaibles so that we can concatenate or stack
# This is required so that we can re-create our U-Net Model

inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x:x/255)(inputs) # normalize input

# Contraction Path
c1 = Conv2D(16,(3,3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(16, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
p1 = MaxPooling2D((2,2))(c1)

c2 = Conv2D(32, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(32, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
p2 = MaxPooling2D((2,2))(c2)

c3 = Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
c3 = Dropout(0.1)(c3)
c3 = Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
p3 = MaxPooling2D((2,2))(c3)


c4 = Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
c4 = Dropout(0.2)(c4)
c4 = Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
p4 = MaxPooling2D((2,2))(c4)

c5 = Conv2D(256, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
c5 = Dropout(0.3)(c5)
c5 = Conv2D(256, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

# Expansion Path
u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5) # no activation
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
c6 = Dropout(0.2)(c6)
c6 = Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

u7 = Conv2DTranspose(64, (2,2) ,strides=(2,2), padding='same')(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
c7 = Dropout(0.2)(c7)
c7 = Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal',padding='same')(c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)


u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)


# Note our output is effectively a mask of 128x128
# c9 shape is (128, 128, 16). Doing Conv2D(1, (1,1)) and 'sigmoid' activation result in (128, 128,1) shape
outputs = Conv2D(1,(1,1), activation='sigmoid')(c9)

model = Model(inputs=[inputs], outputs = [outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])
model.summary()
```

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            (None, 128, 128, 3)  0                                            
    __________________________________________________________________________________________________
    lambda_1 (Lambda)               (None, 128, 128, 3)  0           input_1[0][0]                    
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, 128, 128, 16) 448         lambda_1[0][0]                   
    __________________________________________________________________________________________________
    dropout_1 (Dropout)             (None, 128, 128, 16) 0           conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, 128, 128, 16) 2320        dropout_1[0][0]                  
    __________________________________________________________________________________________________
    max_pooling2d_1 (MaxPooling2D)  (None, 64, 64, 16)   0           conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    conv2d_3 (Conv2D)               (None, 64, 64, 32)   4640        max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    dropout_2 (Dropout)             (None, 64, 64, 32)   0           conv2d_3[0][0]                   
    __________________________________________________________________________________________________
    conv2d_4 (Conv2D)               (None, 64, 64, 32)   9248        dropout_2[0][0]                  
    __________________________________________________________________________________________________
    max_pooling2d_2 (MaxPooling2D)  (None, 32, 32, 32)   0           conv2d_4[0][0]                   
    __________________________________________________________________________________________________
    conv2d_5 (Conv2D)               (None, 32, 32, 64)   18496       max_pooling2d_2[0][0]            
    __________________________________________________________________________________________________
    dropout_3 (Dropout)             (None, 32, 32, 64)   0           conv2d_5[0][0]                   
    __________________________________________________________________________________________________
    conv2d_6 (Conv2D)               (None, 32, 32, 64)   36928       dropout_3[0][0]                  
    __________________________________________________________________________________________________
    max_pooling2d_3 (MaxPooling2D)  (None, 16, 16, 64)   0           conv2d_6[0][0]                   
    __________________________________________________________________________________________________
    conv2d_7 (Conv2D)               (None, 16, 16, 128)  73856       max_pooling2d_3[0][0]            
    __________________________________________________________________________________________________
    dropout_4 (Dropout)             (None, 16, 16, 128)  0           conv2d_7[0][0]                   
    __________________________________________________________________________________________________
    conv2d_8 (Conv2D)               (None, 16, 16, 128)  147584      dropout_4[0][0]                  
    __________________________________________________________________________________________________
    max_pooling2d_4 (MaxPooling2D)  (None, 8, 8, 128)    0           conv2d_8[0][0]                   
    __________________________________________________________________________________________________
    conv2d_9 (Conv2D)               (None, 8, 8, 256)    295168      max_pooling2d_4[0][0]            
    __________________________________________________________________________________________________
    dropout_5 (Dropout)             (None, 8, 8, 256)    0           conv2d_9[0][0]                   
    __________________________________________________________________________________________________
    conv2d_10 (Conv2D)              (None, 8, 8, 256)    590080      dropout_5[0][0]                  
    __________________________________________________________________________________________________
    conv2d_transpose_1 (Conv2DTrans (None, 16, 16, 128)  131200      conv2d_10[0][0]                  
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 16, 16, 256)  0           conv2d_transpose_1[0][0]         
                                                                     conv2d_8[0][0]                   
    __________________________________________________________________________________________________
    conv2d_11 (Conv2D)              (None, 16, 16, 128)  295040      concatenate_1[0][0]              
    __________________________________________________________________________________________________
    dropout_6 (Dropout)             (None, 16, 16, 128)  0           conv2d_11[0][0]                  
    __________________________________________________________________________________________________
    conv2d_12 (Conv2D)              (None, 16, 16, 128)  147584      dropout_6[0][0]                  
    __________________________________________________________________________________________________
    conv2d_transpose_2 (Conv2DTrans (None, 32, 32, 64)   32832       conv2d_12[0][0]                  
    __________________________________________________________________________________________________
    concatenate_2 (Concatenate)     (None, 32, 32, 128)  0           conv2d_transpose_2[0][0]         
                                                                     conv2d_6[0][0]                   
    __________________________________________________________________________________________________
    conv2d_13 (Conv2D)              (None, 32, 32, 64)   73792       concatenate_2[0][0]              
    __________________________________________________________________________________________________
    dropout_7 (Dropout)             (None, 32, 32, 64)   0           conv2d_13[0][0]                  
    __________________________________________________________________________________________________
    conv2d_14 (Conv2D)              (None, 32, 32, 64)   36928       dropout_7[0][0]                  
    __________________________________________________________________________________________________
    conv2d_transpose_3 (Conv2DTrans (None, 64, 64, 32)   8224        conv2d_14[0][0]                  
    __________________________________________________________________________________________________
    concatenate_3 (Concatenate)     (None, 64, 64, 64)   0           conv2d_transpose_3[0][0]         
                                                                     conv2d_4[0][0]                   
    __________________________________________________________________________________________________
    conv2d_15 (Conv2D)              (None, 64, 64, 32)   18464       concatenate_3[0][0]              
    __________________________________________________________________________________________________
    dropout_8 (Dropout)             (None, 64, 64, 32)   0           conv2d_15[0][0]                  
    __________________________________________________________________________________________________
    conv2d_16 (Conv2D)              (None, 64, 64, 32)   9248        dropout_8[0][0]                  
    __________________________________________________________________________________________________
    conv2d_transpose_4 (Conv2DTrans (None, 128, 128, 16) 2064        conv2d_16[0][0]                  
    __________________________________________________________________________________________________
    concatenate_4 (Concatenate)     (None, 128, 128, 32) 0           conv2d_transpose_4[0][0]         
                                                                     conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    conv2d_17 (Conv2D)              (None, 128, 128, 16) 4624        concatenate_4[0][0]              
    __________________________________________________________________________________________________
    dropout_9 (Dropout)             (None, 128, 128, 16) 0           conv2d_17[0][0]                  
    __________________________________________________________________________________________________
    conv2d_18 (Conv2D)              (None, 128, 128, 16) 2320        dropout_9[0][0]                  
    __________________________________________________________________________________________________
    conv2d_19 (Conv2D)              (None, 128, 128, 1)  17          conv2d_18[0][0]                  
    ==================================================================================================
    Total params: 1,941,105
    Trainable params: 1,941,105
    Non-trainable params: 0
    __________________________________________________________________________________________________


## Train our model


```python
# Initialize our callbacks
model_path = "/home/deeplearningcv/DeepLearningCV/Trained Models/nuclei_finder_unet_1.h5"

checkpoint = ModelCheckpoint(model_path,
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 5,
                          verbose = 1,
                          restore_best_weights = True)

# Fit our model
results = model.fit(X_train, Y_train, validation_split = 0.1, # Split 10% of training data to become validation data
                   batch_size=16, epochs=10,
                   callbacks=[earlystop, checkpoint])
```

    Train on 603 samples, validate on 67 samples
    Epoch 1/10
    603/603 [==============================] - 56s 93ms/step - loss: 0.3699 - my_iou_metric: 0.0449 - val_loss: 0.2526 - val_my_iou_metric: 0.1074
    
    Epoch 00001: val_loss improved from inf to 0.25257, saving model to /home/deeplearningcv/DeepLearningCV/Trained Models/nuclei_finder_unet_1.h5
    Epoch 2/10
    603/603 [==============================] - 54s 89ms/step - loss: 0.1882 - my_iou_metric: 0.1972 - val_loss: 0.2672 - val_my_iou_metric: 0.2896
    
    Epoch 00002: val_loss did not improve from 0.25257
    Epoch 3/10
    603/603 [==============================] - 53s 87ms/step - loss: 0.1427 - my_iou_metric: 0.3215 - val_loss: 0.1430 - val_my_iou_metric: 0.2593
    
    Epoch 00003: val_loss improved from 0.25257 to 0.14296, saving model to /home/deeplearningcv/DeepLearningCV/Trained Models/nuclei_finder_unet_1.h5
    Epoch 4/10
    603/603 [==============================] - 53s 88ms/step - loss: 0.1245 - my_iou_metric: 0.3563 - val_loss: 0.1305 - val_my_iou_metric: 0.4077
    
    Epoch 00004: val_loss improved from 0.14296 to 0.13050, saving model to /home/deeplearningcv/DeepLearningCV/Trained Models/nuclei_finder_unet_1.h5
    Epoch 5/10
    603/603 [==============================] - 53s 88ms/step - loss: 0.1091 - my_iou_metric: 0.4038 - val_loss: 0.1066 - val_my_iou_metric: 0.4272
    
    Epoch 00005: val_loss improved from 0.13050 to 0.10658, saving model to /home/deeplearningcv/DeepLearningCV/Trained Models/nuclei_finder_unet_1.h5
    Epoch 6/10
    603/603 [==============================] - 53s 88ms/step - loss: 0.1022 - my_iou_metric: 0.4253 - val_loss: 0.1069 - val_my_iou_metric: 0.4484
    
    Epoch 00006: val_loss did not improve from 0.10658
    Epoch 7/10
    603/603 [==============================] - 54s 90ms/step - loss: 0.0979 - my_iou_metric: 0.4351 - val_loss: 0.1548 - val_my_iou_metric: 0.4313
    
    Epoch 00007: val_loss did not improve from 0.10658
    Epoch 8/10
    603/603 [==============================] - 54s 90ms/step - loss: 0.0915 - my_iou_metric: 0.4540 - val_loss: 0.1016 - val_my_iou_metric: 0.4626
    
    Epoch 00008: val_loss improved from 0.10658 to 0.10160, saving model to /home/deeplearningcv/DeepLearningCV/Trained Models/nuclei_finder_unet_1.h5
    Epoch 9/10
    603/603 [==============================] - 56s 92ms/step - loss: 0.0916 - my_iou_metric: 0.4530 - val_loss: 0.0978 - val_my_iou_metric: 0.4350
    
    Epoch 00009: val_loss improved from 0.10160 to 0.09784, saving model to /home/deeplearningcv/DeepLearningCV/Trained Models/nuclei_finder_unet_1.h5
    Epoch 10/10
    603/603 [==============================] - 55s 91ms/step - loss: 0.0905 - my_iou_metric: 0.4702 - val_loss: 0.1068 - val_my_iou_metric: 0.4581
    
    Epoch 00010: val_loss did not improve from 0.09784


## Generating predictions for training and validation data


```python
# Predict on training and validation data
# Note our use of my_iou metric
model = load_model('/home/deeplearningcv/DeepLearningCV/Trained Models/nuclei_finder_unet_1.h5',
                  custom_objects={'my_iou_metric':my_iou_metric})

# the first 90% was used for training
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)

# the last 10% used as validation
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8) #
preds_val_t = (preds_val > 0.5).astype(np.uint8)
```

    603/603 [==============================] - 14s 24ms/step
    67/67 [==============================] - 2s 23ms/step


### Showing our predicted mask on training data


```python
# Ploting our predicted masks
ix = random.randint(0, 602)
plt.figure(figsize=(20,20))

# Original training image
plt.subplot(131)
imshow(X_train[ix])
plt.title("Image",  fontsize=20, fontweight='bold')

# Original combined mask
plt.subplot(132)
imshow(np.squeeze(Y_train[ix]))
plt.title("Mask",fontsize=20, fontweight='bold')

# The mask our U-Net model predicts
plt.subplot(133)
imshow(np.squeeze(preds_train_t[ix] > 0.5))
plt.title("Predictions",fontsize=20, fontweight='bold')
plt.show()
```


![png](19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_files/19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_24_0.png)



### Show our predicted mask on validation data


```python
# Plot predicted masks
ix = random.randint(602, 668)
plt.figure(figsize=(20,20))

# Our original training image
plt.subplot(121)
imshow(X_train[ix])
plt.title("Image",fontsize=24, fontweight='bold')

# The mask our U-Net model predicts
plt.subplot(122)
ix = ix - 603
imshow(np.squeeze(preds_val_t[ix] > 0.5))
plt.title("Predictions",fontsize=24, fontweight='bold')
plt.show()
```


![png](19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_files/19.%20Medical%20Imaging%20Segmentation%20using%20U-Net_26_0.png)


## Generating a Classification Report


```python
iou_metric(np.squeeze(Y_train[ix]), np.squeeze(preds_train_t[ix]), print_table=True) 
```

    Thresh	TP	FP	FN	Prec.
    0.500	39	12	21	0.542
    0.550	39	12	21	0.542
    0.600	37	14	23	0.500
    0.650	33	18	27	0.423
    0.700	30	21	30	0.370
    0.750	22	29	38	0.247
    0.800	12	39	48	0.121
    0.850	6	45	54	0.057
    0.900	4	47	56	0.037
    0.950	1	50	59	0.009
    AP	-	-	-	0.285





    0.2848800703032563


