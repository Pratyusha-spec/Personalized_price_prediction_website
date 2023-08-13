# USAGE
# python train_covid19.py --d dataset

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


classes = {0:"Bad quality Banana:-{ Bad quality Banana}",1:"Good Quality Banana:-{ Good Quality Banana}",2:"Mixed Quality Banana:-{ Mixed Quality Banana}"}
img_width, img_height = 224,224

# load the model we saved
model = load_model('model.h5')
# predicting images
#img = image.load_img('plant/Train/BANYAN TREES/1.jpg', target_size=(img_width, img_height))
image = load_img('3.jpg',target_size=(224,224))
image = img_to_array(image)
image = image/255
image = np.expand_dims(image,axis=0)
result = np.argmax(model.predict(image))
print(result)

prediction = classes[result]
print(prediction)