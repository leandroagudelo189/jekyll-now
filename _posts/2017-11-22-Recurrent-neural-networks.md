---
published: true
---
![an image alt text]({{ leandroagudelo189.github.io/tree/master }}/images/rnn-model2.jpg "an image title")

### Supervised deep learning (sequence models)

Intuition on recurrent neural networks (Study notes)

----
****

![an image alt text]({{ leandroagudelo189.github.io/tree/master }}/images/rnn1.jpg "an image title")
----
****

![an image alt text]({{ leandroagudelo189.github.io/tree/master }}/images/rnn2.jpg "an image title")

----
****


### Further reading

[Learning long-term dependencies with gradient descent is difficult](http://ai.dinfo.unifi.it/paolo//ps/tnn-94-gradient.pdf), by Bengio et al. 

[On the difficulty of training recurrent neural networks](http://proceedings.mlr.press/v28/pascanu13.pdf), by Pascanu et al.

[Long Short-Term memory](http://www.bioinf.jku.at/publications/older/2604.pdf), by Hochreiter et al.

[Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/), by Olah

[LSTMs and its diagrams](https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714), by Shi Yan

[RNN's unreasonble effect](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), by Karpathy

[LSTM: A Search Space Odyssey](https://arxiv.org/pdf/1503.04069.pdf), by Greff et al.

[Deep Sparse Rectifier Neural Networks](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf), by Glorot et al.

----
****


### Implementation of RNN

```python
### 1. PREPROCESSING

# LIBRABRIES
%matplotlib inline 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

%matplotlib inline 
# IMPORT TRAINING SET
# only train on the training set; no need of test set

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv') # next we need a np.array for kears NN
training_set = dataset_train.iloc[:,1:2].values   # take a range since ranges are excluded so the column 2 will be out (.values creates a np array)

# RNN feature scaling unsing normalization 

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set) # next is to create a datastructure to remember (# of time steps)

# creating a datastructure with 60 timesteps and 1 output
# it will read 60 steps and then try to understand the next step (t+1) in this case the 60 previous financial days
# let's create the x_train with the 60 previous days as well as y_train with the result

X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# reshaping (add the unit= the number of predictior we need to predict the stock price)
# with this we can add more indicator to our prediction (become multidimesional)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



### 2. RNN building
# Libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# initialise 
regressor = Sequential()

# add the LSTM layer and dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))  # since they are stacked lstm sequences we set it to True
regressor.add(Dropout(0.2)) # 20% of neurons will be ignored

# add more LSTM layer with dropout reg
regressor.add(LSTM(units=50, return_sequences=True))  # no need to specify the input_shape since it recognizes automatically
regressor.add(Dropout(0.2)) 

# add more LSTM layer with dropout reg
regressor.add(LSTM(units=50, return_sequences=True))  # no need to specify the input_shape since it recognizes automatically
regressor.add(Dropout(0.2)) 

# add more LSTM layer with dropout reg
regressor.add(LSTM(units=50))  # no need to specify the input_shape since it recognizes automatically
regressor.add(Dropout(0.2)) 

# output layer
regressor.add(Dense(units=1))

# compiling
regressor.compile(optimizer='adam', loss= 'mean_squared_error')

# fit the RNN
regressor.fit(X_train, y_train, epochs=20, batch_size=64) # next time use 100 epochs or more and bs of 32



### 3. Prediction and visualization

# stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv') # next we need a np.array for kears NN
real_stock_price = dataset_test.iloc[:,1:2].values   # take a range since ranges are excluded so the column 2 will be out (.values creates a np array)


# getting the predicted stock price of 2017
# the keypoint to understand is that in order to predic the stock price of 2017 january we need the last 60 financial days 
# the algorithm train on how the stock behaved during 5 years in LSTMs of 60 days plus real value and predicted value

# we need to concatenate the training set and test set 
# we need to keep the test real values
# so we concatenate the two original dataset without scaling
# then we take the 60 inputs we need to predict the last month
# and we scale these inputs
# so we are not changing the real values of the prediction

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0 ) # for vertical axis=0 and horizontal axis=1
# get the 60 previous stock prices  for each new financial day
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)

# scale the inputs
inputs = sc.transform(inputs)


# now repeat the process to 3D expected by the NN

X_test = []


for i in range(60, 80): # the range goes to 80 since we only need 20 finantial days
    X_test.append(inputs[i-60:i, 0])
   

X_test = np.array(X_test)
# reshaping (add the unit= the number of predictior we need to predict the stock price)
# with this we can add more indicator to our prediction (become multidimesional)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)

# inverse the scaling
predicted_stock_price = sc.inverse_transform(predicted_stock_price)



### VISUALISING

plt.plot(real_stock_price, color= 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color= 'blue', label = 'LSTM-Predicted Stock Price')
plt.title('LSTM Stock Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()



### to evaluate the model (regression) we use Root Mean Squared Error 
# it is caluculated as the root of the mean of the squared differences between the predictions and the real values

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))


# how to improve the model
"""
1. more training data
2. increase the nomber of timesteps for example to 6 months
3. adding new indicators
4. adding more LSTM layers
5. adding more neurons to the LSTM layers

Tuning
you need to replace 
scoring = 'accuracy'
scoring = 'neg_mean_squared_error' 
in the GridSearchCV class parameters
"""

````
