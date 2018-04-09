---
published: true
---
![an image alt text]({{ leandroagudelo189.github.io/tree/master }}/images/neural-transfer.jpg "an image title") 


## Study notes convolutional neural networks 


![an image alt text]({{ leandroagudelo189.github.io/tree/master }}/images/cnn1.jpg "an image title")
----
****

![an image alt text]({{ leandroagudelo189.github.io/tree/master }}/images/cnn2.jpg "an image title")


## Further reading

[Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf), by LeCun et al. 

[Introduction to CNNs](https://cs.nju.edu.cn/wujx/paper/CNN.pdf), by Wu

[Understanding CNNs with a mathematical model](https://arxiv.org/pdf/1609.04112.pdf), by Kuo

[ Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852.pdf), by He et al.

[Evaluation of Pooling Operations in Convolutional Architectures for Object Recognition](http://ais.uni-bonn.de/papers/icann2010_maxpool.pdf), by Scherer et al.

[A convolutional recursive modified Self Organizing Map for handwritten digits recognition](http://www.sciencedirect.com/science/article/pii/S0893608014001968?via%3Dihub), by Mohebi et al.

## Implementation

```python
### Convolutional Neural Networks
# Computer recognition of categorical data

### 1. Build the CNN

# importing libraries and packages

from keras.models import Sequential # to initialize the NN as a sequence of layers
from keras.layers import Convolution2D # for step 1 in convolutional layers
from keras.layers import MaxPooling2D 
from keras.layers import Flatten # convert the pooled features into one single vectors
from keras.layers import Dense # to add the fully connected layers

# start the CNN
model = Sequential() # create an object of our CNN

# Add the convolutional layers 
model.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))   # we will apply a method on this object (filters = feature_maps with #of rows and columns)

# Reduce the filters by Max pooling
model.add(MaxPooling2D(pool_size=(2,2)))

# add an additional or second convolutional layer after the max pooling
model.add(Convolution2D(32, (3, 3),  activation='relu')) # we don't need to include the input_shape since we have the pooled features (keras will notice it)
model.add(MaxPooling2D(pool_size=(2,2)))

# Flattening= take our pooled maps and convert them into a vector
model.add(Flatten())

# make a classic ANN "fully connected"
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# compile the CNN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# fit the CNN to the images
# use image augmentation to preprocess images to avoid overfitting
# use flow_from_directory if you have your dataset organized in folders
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64), # to get better accuracy one can increase the size here (more pixels)
                                                batch_size=32,
                                                class_mode='binary')


test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
        

model.fit_generator( training_set,
                    steps_per_epoch=8000,  #total samples
                    epochs=2,
                    validation_data=test_set,
                    validation_steps=2000)


# to improve the accuracy of the model we can do different things
# 1. Add an additional convolutional layer
# 2. Tune the ANN. Hyperparameter tuning or add an additional fully-connected layer


### now predict your own pictures

import numpy as np
from keras.preprocessing import image
test_new_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
# now we need to add a new dimension to the image to match the input shape of our training dataset
test_new_image = image.img_to_array(test_new_image)
# we need to add an additional dimension to our 3D-array     
# because the conv2d method only accepts input in a batch (index) therefore 4 dimensions
test_new_image = np.expand_dims(test_new_image, axis=0)

prediction = model.predict(test_new_image)
training_set.class_indices

if prediction[0][0] == 1:
    pred_final = 'dog'
else:
    pred_final = 'cat'
````

### Additional implementation
The following implementation gives higher accuracy. If training using GPUs is possible, the accuracy will increase much more. 

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 150, 150


# CNN model
# with dropout
# input shape

def create_model(p, input_shape=(32, 32, 3)):
    model = Sequential() # initialize
    
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    
    
    model.add(Flatten())
    
    # fully connected-NN
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p))
    
    model.add(Dense(1, activation='sigmoid'))
    
    
    #compiling
    optimizer = Adam(lr=1e-3)
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss = 'binary_crossentropy', metrics=metrics)
    return model


def run_training(bs=32, epochs=10):
    
    train_datagen = ImageDataGenerator(rescale = 1./255, 
                                       shear_range = 0.2, 
                                       zoom_range = 0.2, 
                                       horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    
    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (img_width, img_height),
                                                 batch_size = bs,
                                                 class_mode = 'binary')
                                                 
    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (img_width, img_height),
                                            batch_size = bs,
                                            class_mode = 'binary')
    
    
    
    
    model = create_model(p=0.6, input_shape=(img_width, img_height, 3))                                  
    model.fit_generator(training_set,
                         steps_per_epoch=8000/bs,
                         epochs = epochs,
                         validation_data = test_set,
                         validation_steps = 2000/bs)
    
    
    
def main_function():
    run_training(bs=32, epochs=100)
   
    
if __name__ == '__main_function__':
    main_function()
    
````
