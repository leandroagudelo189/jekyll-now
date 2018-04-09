---
published: true
---
![an image alt text]({{ leandroagudelo189.github.io/tree/master }}/images/facialexp-rbm.png "an image title")


# Study notes RBMs  


![an image alt text]({{ leandroagudelo189.github.io/tree/master }}/images/Boltzmann-machines.jpg "an image title")

![an image alt text]({{ leandroagudelo189.github.io/tree/master }}/images/boltzmann-machines-2.jpg "an image title")

![an image alt text]({{ leandroagudelo189.github.io/tree/master }}/images/boltzmann-machines-3.jpg "an image title")



# Further reading

[Energy-based learning](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf), by Yann LeCun et al. 

[A fast learning algorithm for deep belief networks](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf), by Hinton et al.

[Contrasting divergence](http://www.robots.ox.ac.uk/~ojw/files/NotesOnCD.pdf), by Woodford et al.

[Training of deep belief networks](http://www.iro.umontreal.ca/~lisa/pointeurs/BengioNips2006All.pdf), by Bengio et al.

[The wake-sleep algorithm for unsupervised neural networks](http://www.gatsby.ucl.ac.uk/~dayan/papers/hdfn95.pdf), by Hinton et al.

[Deep boltzmann machines](http://www.utstat.toronto.edu/~rsalakhu/papers/dbm.pdf), by Salakhutdinov et al.




# Implementation of RBM for recommendation-systems


- import the corresponding libraries
- in case of error during torch import, update numpy to 1.13 version
- test your version and path as follows:

```python
import numpy as np
print (np.__version__)
print (np.__path__)
```

We will try to build a recommender system with data from visuallens which contains users ratings ([Dataset](https://github.com/leandroagudelo189/AutoEncoders/blob/master/ml-1m.zip))

The dataset is also available at ([Grouplens](https://grouplens.org/datasets/movielens/))

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
````
```python 

# dataset
# importing all of the datasets
# engine is to make sure the data is imported correctly
# the encoding is different than utf because of some especial characters
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine = 'python', encoding = 'latin-1')


# preparing the dataset 
# training set and test set
# we will have 5 sets of training and test sets called u1...u5
# so we have 5 k-fold cross validation 
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype='int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype='int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# now we will organize our data so we can use it in 4k-fold cross validation
# it will be users in lines and films in columns with their respective ratings
# so the ratings will be our observations (features)

# we will create a list of lists
# it will be a list per users so, 943 lists (horizontal lists) with all movies-ratings
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users] # a var with all the movies rated and we use another braquets for the condition 
        id_ratings = data[:,2][data[:,0] == id_users] # the ratings per id_user
        ratings = np.zeros(nb_movies) # backbone to replace with the ratings and if not it will remain a zero
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

# we will apply this function to the training and test set
training_set = convert(training_set)
test_set = convert(test_set)
'''
now we need to convert our list into Torch tensors
the lines are gonna be the observations into the networks and the columsn are the features that are the input nodes of the networks
let's do our multidimesional matrix using pytorch'''
training_set =  torch.FloatTensor(training_set)
test_set =  torch.FloatTensor(test_set)

'''if you want to create a classifier of whether it belongs to a specific group like 0 or 1. 
0 = not belong and 1 =  belongs to 
you need to convert your data into binary
we will use RBMs to do this
they need to have the same format the output and input'''
# the unrated movies will have a -1 instead of zero
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# creating the architecture of the RBM
# we will create the probabilistic graphical model with a Class 
# Class as an ensemble of instructions of what we want to buil 
# it is guide, and we create objects to call these instructions
# our class will have all the info we need to build the RBM (hidden nodes, weights, bias...)
# 3 functions in our class: 1, To initialize our RBM. 2, sample the probability of the hidden nodes given the visible nodes
# 3, sample b from hidden to visible nodes probabilities

class RBM():
    def __init__(self, nv, nh): 
        '''it always defines the parameters of the object once the class is made
           it is the object that will be when we call the class. It helps specifying the variable
           Define all the parameters such as the weight and the variance'''
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh) # bias for the hidden nodes
        self.b = torch.randn(1, nv) # bias for the visible nodes

    ''' the second function will sample the probabilities of h given v (visible nodes) 
    and it is the sigmoid function. During the training we will approximate 
    the log likelihood gradient through gibb sampling. We compute the probabilities of the hidden node
    given the inputs. Then we sample the activation'''
    def sample_h(self, x): # x is the visible neurons v in ph given v
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx) # to make sure that the bias is applied to each side of the minibatch
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v) # remember this is a bernouille RBM so it is a binary outcome. Then it will return some bernouille samples of that distribution
# in other workds we are sample the probabilities of a hidden node being activated given v

    ''' now we need to sample our probabilities from the hidden to visible nodes. This is 
    to estimate the prob that each node is 1'''
    
    def sample_v(self, y): # y is the hidden node
        wy = torch.mm(y, self.W) 
        activation = wy + self.b.expand_as(wy) # to make sure that the bias is applied to each side of the minibatch
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    ''' in the last function we will do the contrastive divergence (CD) that we will use to approximate 
    the log-likelihood gradient. We are optimizing the weights to decrease the energy of the system.
    So to minimize the energy we need to maximize the log-likelihood gradient. We compute the gradient by
    approximation using the CD. We carried out gibbs chain in k-step by sampling in k-times the h and v nodes
    We sample the visible nodes to obtain the hidden node, then we sample the hidden to obtain the visible
    we do this process k-times 
    v0 = input vector
    vk = visible node after k-sampling
    ph0 = vector of probabilities at v0
    phk = probabilities after gibbs sampling given vk
    '''
    def train(self, v0, vk, ph0, phk): # we will update the weights and the biases
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0-vk), 0) # we add the 0 to keep the format of a tensor of 2 dimensions
        self.a += torch.sum((ph0-phk), 0)

# call the class RBM with the two parameters nh and nv 
# when we create the RBM object only the init function will take action. So the parameters w, b, a will
# be initialized. The other function will go in action during training
# it happens since the init is the default function and from this object we can use different functions defined in the class

''' so the first object will be nv, which are the movies. So one visible node for each movie.
   in other words it is the input 
make sure it corresponds to the number of feature from the input vector
'''

nv = len(training_set[0]) # number of elements in the first line. number of features
nh = 100 # number of features we want to detect. We can hyperparameterize
batch_size = 100 # we can change it
rbm = RBM(nv, nh) # object class

''' train the RBM
create a for loop where our ten observations (epochs) will go throough the network
and update the weight every time it finishes.
remember we need a loss function to compare prediction vs real
'''
nb_epoch = 10 # because we have few observations, number of users (we can use same model for genes- more epochs)

for epoch in range(1, nb_epoch + 1):
    # let's use the absolute difference
    train_loss = 0 # we need a counter. To normalize the train_loss. Divide the train_loss by the counter s
    s = 0. # counter type float
    # let's get the batches of users 
    for id_user in range(0, nb_users - batch_size, batch_size): # the third argument is the step
        vk = training_set[id_user:id_user + batch_size]  # the input of vectors
        v0 = training_set[id_user:id_user + batch_size]# the target is the batch of original ratings that we want to compare to our predicted ratings
        ph0,_ = rbm.sample_h(v0)# initial probabilities so we need p_h_given_v. We add a come and underscore to get the first value of the function
        # let's loop the k-steps of contrastive divergence
        for k in range(10): # k-steps of the random walk- gibbs sampling. It is a markov chain and monte carlo technique MCMC
            # in every steps the visible and hidden nodes are updated so we get closer to the predicted ratings
            # we need to call the sample_h on the visible node to get the first sampling of nv
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            # now we need to not learn from the nodes containing the -1. To do so, we freeze these values
            vk[v0<0] = v0[v0<0]
        # let's compute phk before start training
        phk,_ = rbm.sample_h(vk) # taking the probabilities of the last step
        rbm.train(v0, vk, ph0, phk)
        # update the train loss 
        train_loss += torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0]))
        s += 1.
    print('epoch: ' + str(epoch)+ ' loss: '+ str(train_loss/s))     
    
# testing the RBMs 
'''
   We need to test the RBM. we dont need the epoch or the loops for it.
   Batch_size can alse be removed
   we will compare our target input from the test_set to the training set predictions
   we dont need the ph0 'cause we use it to train our model
'''
    # let's use the absolute difference
test_loss = 0 # we need a counter. To normalize the train_loss. Divide the train_loss by the counter s
s = 0. # counter type float
# let's get the batches of users 
for id_user in range(nb_users): # the third argument is the step
    v = training_set[id_user:id_user + 1]  # We cant replace it, because the training set is the input that will be use to activate the hidden neurons of the RBM
    vt = test_set[id_user:id_user + 1] # target vector from the test_set
# we need only one step for the blind walk. We were trained to do 100 k-steps of the random walk- gibbs sampling. It is a markov chain and monte carlo technique MCMC
    if len(vt[vt>=0]) > 0: # we will filter the non existing ratings of the test_set   
        _,h = rbm.sample_h(v) # we remove the k because we are not in a loop of k steps
        _,v = rbm.sample_v(h)
    # let's compute phk before start training
    # update the train loss 
        test_loss += torch.mean(torch.abs(vt[vt>=0]-v[vt>=0]))
        s += 1.
print('test loss: '+ str(test_loss/s)) 
```
