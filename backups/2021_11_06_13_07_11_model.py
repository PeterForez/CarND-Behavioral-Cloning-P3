#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import csv
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import plot_model

from sklearn.utils import shuffle
from math import ceil

import warnings
warnings.filterwarnings("ignore")

import os


# # Global Variables

# In[2]:


BASE_PATH   = 'data/'
EPOCH       = 10
BATCH_SIZE  = 64    # Set our batch size
CORRECTION  = 0.2
RECORD_DROP = 0.8


# In[3]:


def save_backups():
    from IPython.display import display, Javascript
    import shutil
    display(Javascript('IPython.notebook.save_checkpoint();'))         # Save the Notebook
    os.system('jupyter nbconvert --to script model.ipynb')             # Save the Notebook to as py file
    os.system('jupyter nbconvert --to html model.ipynb')               # Save the Notebook to as html file

    shutil.copyfile('model.h5',    f'backups/'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+ '_model.h5')    #Backup for models.h5
    shutil.copyfile('model.py',    f'backups/'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+ '_model.py')    #Backup for model.py
    shutil.copyfile('model.ipynb', f'backups/'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+ '_model.ipynb') #Backup for model.ipnb
    shutil.copyfile('model.html',  f'backups/'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+ '_model_.html') #Backup for model.html
    
    with open('backups/'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+ '_Libraries.txt', 'w') as f:
        package_used = os.popen('pip list').read()
        f.write(package_used)


# # Nvidia Model

# ![Nvidia Model](https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)

# In[4]:


def Nvidia():
    model = Sequential()
    model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape = (160,320,3)))
    model.add(Cropping2D(cropping = ((70, 25), (0,0))))
    model.add(Conv2D(24,(5,5), subsample = (2,2), activation = 'relu'))
    model.add(Conv2D(36,(5,5), subsample = (2,2), activation = 'relu'))
    model.add(Conv2D(48,(5,5), subsample = (2,2), activation = 'relu'))
    model.add(Conv2D(64,(3,3), activation = 'relu'))
    model.add(Conv2D(64,(3,3), activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss = 'mse', optimizer = 'adam')
    plot_model(model, to_file='Images/Nvidia_Model.png',show_shapes=True, rankdir='TB');
    return model


# # Plot the MSE

# In[5]:


def plot_mse(history):
    plt.figure(figsize=(12,6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.grid()
    plt.show()
    plt.savefig('Images/MSE')


# # Generator

# In[6]:


def generator(samples, batch_size=32, ADD_SIDE = 0, ADD_FLIP = 0):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                c_path  = batch_sample[0]
                c_image = cv2.imread(c_path)
                c_angle = float(batch_sample[3])
                images.append(c_image)
                angles.append(c_angle)
                                                  
                if ADD_SIDE:
                    l_path  = batch_sample[1]
                    r_path  = batch_sample[2]
                    
                    l_image = cv2.imread(l_path)
                    r_image = cv2.imread(r_path) 
                    
                    l_angle = float(batch_sample[3]) + CORRECTION
                    r_angle = float(batch_sample[3]) - CORRECTION
                    
                    images.append(l_image)
                    images.append(r_image)
                    
                    angles.append(l_angle)
                    angles.append(r_angle)
                    
                if ADD_FLIP:
                    images.append(cv2.flip(c_image,1))
                    angles.append(c_angle*-1.0)
                    
                    if ADD_SIDE:
                        images.append(cv2.flip(l_image,1))
                        images.append(cv2.flip(r_image,1))

                        angles.append(l_angle*-1.0)
                        angles.append(r_angle*-1.0)

            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


# # Import the Data

# In[7]:


df = pd.read_csv(BASE_PATH + '\driving_log.csv', skipinitialspace=True)
df.head()


# # Drop Unused columns

# In[8]:


df_0 = df.drop(['throttle', 'brake', 'speed'], axis=1)
df_0.head()


# # Update the path 

# In[9]:


df_0.center = BASE_PATH + df_0.center
df_0.left   = BASE_PATH + df_0.left
df_0.right  = BASE_PATH + df_0.right
df_0.head()


# # Distribution of the Sterring Angle

# In[10]:


print("The original data contains {} images".format(df_0.shape[0]))


# In[11]:


plt.figure(figsize=(15,6))
df_0.steering.hist(bins = 50);
plt.title('Original Distribution of the sterring angle')
plt.ylabel('Count')
plt.xlabel('Sterring Angle');
plt.savefig('Images/SteeringAngle')


# **Most of the distribution of the steering angle is around zero.**
# 
# **So, I decided to drop some of those pictures**

# In[12]:


df_0.drop(df_0[df['steering'] == 0].sample(frac=RECORD_DROP).index, inplace = True)


# In[13]:


print("The updated data contains {} images".format(df_0.shape[0]))


# In[14]:


plt.figure(figsize=(15,6))
df_0.steering.hist(bins = 50);
plt.title('Updated Distribution of the sterring angle')
plt.ylabel('Count')
plt.xlabel('Sterring Angle');
plt.savefig('Images/SteeringAngle_Updated')


# In[15]:


samples = df_0.values.tolist()


# In[16]:


plt.figure(figsize = (20,4))
i = 0
plt.subplot(1,3,1)
image = cv2.imread(samples[i][1])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.xticks([]);plt.yticks([])
plt.title("Original Left Camera")

plt.subplot(1,3,2)
image = cv2.imread(samples[i][0])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.xticks([]);plt.yticks([])
plt.title("Original Middle Camera")

plt.subplot(1,3,3)
image = cv2.imread(samples[i][2])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.xticks([]);plt.yticks([])
plt.title("Original Right Camera");


plt.savefig('Images/CameraImages_Original')


# In[17]:


plt.figure(figsize = (20,4))
i = 0
plt.subplot(2,3,4)
image = cv2.imread(samples[i][1])
image = image[70:-25, :]
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.xticks([]);plt.yticks([])
plt.title("Cropped Left Camera")

plt.subplot(2,3,5)
image = cv2.imread(samples[i][0])
image = image[70:-25, :]
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.xticks([]);plt.yticks([])
plt.title("Cropped Middle Camera")

plt.subplot(2,3,6)
image = cv2.imread(samples[i][2])
image = image[70:-25, :]
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.xticks([]);plt.yticks([])
plt.title("Cropped Right Camera");

plt.savefig('Images/CameraImages_Cropped')


# ## Model Training

# In[18]:


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# In[19]:


# compile and train the model using the generator function
train_generator      = generator(train_samples,      batch_size = BATCH_SIZE, ADD_SIDE = 1, ADD_FLIP = 1)
validation_generator = generator(validation_samples, batch_size = BATCH_SIZE, ADD_SIDE = 0, ADD_FLIP = 0)


# In[20]:


from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
from datetime import datetime

# https://keras.io/api/callbacks/
my_callbacks = [
    ModelCheckpoint(filepath='backups/'+ datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '_model.{epoch:02d}-{val_loss:.4f}.h5'),
    CSVLogger(filename = 'backups/'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'_training.log', separator=",", append=False)
]


# In[21]:


model = Nvidia()
model.summary()


# In[22]:


history = model.fit_generator(train_generator, 
                              steps_per_epoch  = ceil(len(train_samples) * 6 / BATCH_SIZE), 
                              validation_data  = validation_generator, 
                              validation_steps = ceil(len(validation_samples) / BATCH_SIZE), 
                              callbacks        = my_callbacks,
                              epochs           = EPOCH)

model.save('model.h5')
plot_mse(history)


# # Plot the MSE for training and validation set

# In[23]:


plot_mse(history)


# # Save backups from the files

# In[ ]:


save_backups()


# # END 
