---
published: true
---



![an image alt text]({{ leandroagudelo189.github.io/tree/master}}/images/deep-q-implementation.jpg "an image title")


# Requirements

- Anaconda (run jupiter lab or spyder)
- Python 2.7 or 3.5 (for this, we will use the 2.7 version)
- Create a new environment in anaconda with $ conda create --name autonomous-agents
- You can also create a environments.yml file and install all the dependencies with $ conda env create -f environments.yml
- Install pytorch and torch vision [installation here](http://pytorch.org/)
- For the agent environment, install the kivy application [installation here](https://kivy.org/#home)

If you are all set-up, let's create the classes for the agent first

# Libraries

let's import the libraries we will be using in this file

``` python

import numpy as np
import random # random samples from our batches when carrying out experience replay
import os # this is to load the model and save it in a pth file
import torch  
import torch.nn as nn
import torch.nn.functional as F # this functional package conatins the different functions we will use when implementing the NN
import torch.optim as optim  # to improve the SGD
import torch.autograd as autograd # we need to converge our tensor to a variable that containg our gradients
from torch.autograd import Variable
´´´


