---
published: true
---



![an image alt text]({{ leandroagudelo189.github.io/tree/master}}/images/deep-q-implementation.jpg "an image title")


# Requirements

- Anaconda (run jupiter lab or spyder)
- Python 2.7 or 3.5 (for this, we will use the 2.7 version)
- Create a new environment in anaconda with $ conda create --name autonomous-agents
- You can also create a environments.yml file and install all the dependencies with $ conda env create -f environments.yml
- Install pytorch and torch vision [(installation here)](http://pytorch.org/)
- For the agent environment, install the kivy application [(installation here)](https://kivy.org/#home)

If you're all set-up, let's first create the classes for the agent. 

# Autonomous Agent

We start by importing the libraries we will be using.

```python

import numpy as np
import random # random samples from our batches when carrying out experience replay
import os # this is to load the model and save it in a pth file
import torch  
import torch.nn as nn

# this functional package conatins the different functions we will use when implementing the NN
import torch.nn.functional as F 
import torch.optim as optim  # to improve the SGD

# we need to converge our tensor to a variable that containg our gradients
import torch.autograd as autograd 
from torch.autograd import Variable
```

# 1. The architecture

Now, we will create the network class. This class contains the architecture of the neural network, the number of hidden layers and their activation function. 

```python

class Network(nn.Module): # we are doing inheritance of a the parent class Module from torch
    
    def __init__(self, input_size, nb_action):
        '''this will be our model that can be modified when we create objects of this class.
        Remember to always use the self
        Here, we have 5 inputs from our sensors 
        [left, right, front, orientation and - orientation].
        And 3 actions to take with the softmax function. '''
        
        # we now use the super function from torch 
        # this in order to use all the tools from the nn.module
        super(Network, self).__init__()
        
        # now we will create a variables that will be attached to the objects. 
        #These variables contain the input neurons and actions to take
        self.input_size = input_size
        self.nb_action = nb_action
        
        # let's create the first full connection between the inputs and the hidden layers
        # this can be changed to try to improve performance of the model
        
        self.fc1 = nn.Linear(input_size, 60) # let's add another hidden layer
        self.fc2 = nn.Linear(60, 40)
        self.fc3 = nn.Linear(40,25)
        self.fc4 = nn.Linear(25, nb_action)
        
        def forward(self, state):
        '''now we make the function that will carry out forward propagation. 
        It will activate the neurons. It will also take the q values for each state'''
        
        # we will now activate the hidden layers using nn.functional from torch
        x = F.relu(self.fc1(state))
        x2 = F.relu(self.fc2(x))
        x3 = F.relu(self.fc3(x2))
        q_values = self.fc4(x3)
        return q_values
        
```

