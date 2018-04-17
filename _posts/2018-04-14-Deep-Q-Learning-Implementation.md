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

let's import the libraries we will be using in this file

```python

import numpy as np
import random # random samples from our batches when carrying out experience replay
import os # this is to load the model and save it in a pth file
import torch  
import torch.nn as nn
import torch.nn.functional as F # this functional package conatins the different functions we will use when implementing the NN
import torch.optim as optim  # to improve the SGD
import torch.autograd as autograd # we need to converge our tensor to a variable that containg our gradients
from torch.autograd import Variable
```

# 1. The architecture

Now, we create the network class, containing the architecture of the neural network, the amount of hidden layers and their activation functions. 

```python

class Network(nn.Module): # we are doing inheritance of a the parent class Module from torch
    
    def __init__(self, input_size, nb_action):
        '''this will be our model that can be modified when we create objects of this class. Remember to always use the self
        # we have 5 inputs from our sensors [left, right, front, orientation and - orientation].
        # 3 actions to take with the softmax function. '''
        
        # we now use the super function from torch in order to use all the tools from the nn.module
        super(Network, self).__init__()
```

