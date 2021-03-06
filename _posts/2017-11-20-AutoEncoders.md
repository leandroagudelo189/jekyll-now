---
published: true
---
<meta name="viewport" content="width=device-width, initial-scale=1.0">
![an image alt text]({{ leandroagudelo189.github.io/tree/master }}/images/sdae.png "an image title")

# Study notes AutoEncoders 


![an image alt text]({{ leandroagudelo189.github.io/tree/master }}/images/AE1.jpg "an image title")

![an image alt text]({{ leandroagudelo189.github.io/tree/master }}/images/AE2.jpg "an image title")



# Further reading

- [Neural Networks Are Impressively Good At Compression](https://probablydance.com/2016/04/30/neural-networks-are-impressively-good-at-compression/), by Skarupke 

- [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html), by Chollet

- [Sparse Autoencoders](http://mccormickml.com/2014/05/30/deep-learning-tutorial-sparse-autoencoder/), by McCormick  

- [Sparse Autoencoders](http://www.ericlwilkinson.com/blog/2014/11/19/deep-learning-sparse-autoencoders), by Wilkinson

- [K-sparse Autoencoders](https://arxiv.org/pdf/1312.5663.pdf), by Makhzani et al.

- [Denoising Autoencoders](http://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf), by Vincent et al.

- [Contractive Autoencoders](http://machinelearning.wustl.edu/mlpapers/paper_files/ICML2011Rifai_455.pdf), by Rifai et al.

- [Stacked Denoising Autoencoders](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf), by Vincent et al.

- [Reducing the Dimensionality of
Data with Neural Networks](https://www.cs.toronto.edu/~hinton/science.pdf), by Hinton et al.


# Implementation of AE for recommendation systems

[Dataset](https://github.com/leandroagudelo189/AutoEncoders/tree/master/ml-1m)

```python
# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))
````
