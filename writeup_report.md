# **Behavioral Cloning** 
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](model.py) containing the script to create and train the model
* [drive.py](drive.py) for driving the car in autonomous mode
* [model.h5](model.h5) containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my [drive.py](drive.py) file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The [model.py](model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. In addition to that, I included a jupyter notebook containing the images about the procedure.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
At the beggining, we normalized the data in the image to be between -0.5 to 0.5. After that we cropped the upper portion of the pictures that contains sky and lower portion that contains a part of the car.

In the following image, you can find the image captured from the three cameras:
![](Images/CameraImages_Original.png)

After cropping the images, the model will only see the part of the road.
![](Images/CameraImages_Cropped.png)


The normalized cropped image will be fed to Nvidia convolution neural network as desribed in the [link](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/). 

The architecture consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers.

![](Images/Nvidia_Model.png)

The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model
I did not include any droppout layer in the architecture as the car is driving smooth in the track. In fact, the validation loss is increasing. This is future work that will be done to make sure that the model is well overfitting.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
![](Image/MSE)

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road after correction factor 0.2 in the steering angle.
```python
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
```

In addition to that, I found that the input data is not balanced and the distribution contains a lot of zero steering angle as shown.
![](Images/SteeringAngle.png) 

So, I decided to drop 80% of the images that have zero sterring angle as shown.
```python
df_0.drop(df_0[df['steering'] == 0].sample(frac=RECORD_DROP).index, inplace = True)
```
![](Images/SteeringAngle_Updated.png) 

In fact this is change that make the car successfully pass the sharp turning on the bridge.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
I used Nvidia architecture without changing anything in it.
![](https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)

```python
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
```
#### 2. Choose the appropriate model
When I fit the model with data, I choose to have 10 epoch. After the completion of each epoch, we have a callback function to save the model generated with validation loss information. All these models are saved in the folder [backups](backups).


#### 3. Creation of the Training Set & Training Process
I used the data set given to me from udacity in the [link](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip). This was sufficient to complete the track.

