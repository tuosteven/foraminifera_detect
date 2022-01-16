# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 00:53:00 2021

@author: tu i fan
"""





from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.models import load_model
from keras import models
from tensorflow import keras
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, Conv2D, MaxPooling2D
import numpy as np
# In[]:
    



model = Sequential()
model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=(150,150,3), activation='relu'))
model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# In[]:
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))



# In[]:

from keras.optimizers import Adam
opt = Adam(lr=0.01)
model.compile(loss='binary_crossentropy',
            optimizer=RMSprop(lr=0.0001),
            metrics=['accuracy'])
model.summary()

# In[]:
model.load_weights("../foraminiera_detect.h5")
# In[]:
    


# In[]:
import time
def predict(path, thresh=0.04):
  start=time.time()
  img = image.load_img(path, target_size=(150,150), interpolation='lanczos')
  img = image.img_to_array(img) / 255.
  #img = np.asarray(img, dtype='float32')/255.
  conf = model.predict(img.reshape(-1, *img.shape))[0][0]
  if conf < thresh:
      animal ='星星'
  elif thresh < conf <(1-thresh) :
      animal ='喳喳喳 '
  else :animal ='武漢'
  
  #plt.imshow(img)
  end=time.time()
  return (animal, '%.60f'%conf,end-start)



