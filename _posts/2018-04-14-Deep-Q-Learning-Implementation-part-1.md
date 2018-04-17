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

# A. Autonomous Agent

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

# 2. Experience replay

This class will help our agent to learn better the weights, even in the absence of variations from the incomming inputs. It will make an approximation (sample) of previous similar states as inputs (in batches or series of events). In sum, it saves state-transitions in an artificial memory.

```python

class ReplayMemory(object):
    
    def __init__(self, capacity):
        
        # the maximum number of transitions
        self.capacity = capacity
        
        # initialize the list of series
        self.memory = []
    
    
    '''now we will create a push function to append 
    the batches/transitions in the memory list. 
    it will also make sure the transitions are for example 100 (fixed number)
    The argument of this function (event) will contain 4 things. 
    1. last state, 2. new state, 3. last action, 4 last reward'''
    
    def push(self, event):
        self.memory.append(event)
        
        # we will try 100 000 transitions
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    
    ''' now let's make the function that determines 
    the sample input from the batch-transition or batch_size'''
    def sample(self, batch_size):
    
    	# we need the memory and the size of the batches
        samples = zip(*random.sample(self.memory, batch_size))
        
        '''
        the zip function changes the shape of the list.
        We cannot return the samples directly because 
        we need to put it in a pytorch variable.
        To do this, we use the map() function; it will take the samples
        and map them to torch variables containing the tensors and the gradients.
        It takes several arguments: the first one is the function we want to use 
        and the second the sequnce we want to applied the function on.
        In sum, we use the map() to create subfunctions for 
        additional manipulations of the data.
        We call it lambda x and it is followed by what we want it to return,
        in this case we want to transform our samples (previous line of code) 
        into a torch variables.
        So we use the Variable function from torch. 
        In order to get everything well aligned 
        we also need to concatenate the actions with the states.
        This way, each row corresponds to the action, the state and the reward for its "t"
        
        '''
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
        '''
        in sum: 
        1. the lambda function will take the samples, 
        concatenate them with respect of the first dimension
        
        2. convert those tensors into torch variables containing 
        both the tensors and the gradients
                 
        3. this torch varible[tensors, gradients] will be used 
        for stochastic gradient descent'''


```


# 3. Deep Q-learning Network

This is the class that integrates our model making use of the previous classes. We will have 7 different functions including:
 1. The __init__ function that creates the objects of our model
 2. A function to select the best action (softmax())
 3. A function where we carry out backpropagation
 4. A function where we update the weights
 5. A function to score the mean of the rewards
 6. A function to save the model in a pth file for re-training
 7. A funtion to load the last saved model
 
 ```python
 
 class Dqn():
    
    #  first create objects of what we need in our model
    def __init__(self, input_size, nb_action, gamma):
        
        self.gamma = gammma
        
        # the following object will be the reward of th batch
        self.reward_window = []
        
        # now an object of our neural network class
        self.model = Network(input_size, nb_action)
        
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        
        # now we create the variables composing the transition events
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
        
    '''now we create the function of selecting the best 
       action depending on the states/batches'''
    def select_action(self, state):   
    	# takes the input-state as an argument. The signals comming from the sensors 
        
        # we use the softmax function to determine the distribution 
        # of probabilities of any possible action
        probs = F.softmax( self.model( Variable( state, volatile = True ) ) * 7 )
        # temperature = 7. It helps the results from the softmax function 
        #to undertake the decision with the highest probability.
        
        # now wetake a random prob fro our distribution softmax
        # function by using the multinomial method
        action = probs.multinomial()
        
        # now we return the action without the batch fake dimension, so we use .data[index]
        return action.data[0,0]
    
    
    '''now we will train our neural network carrying out forwrd- and backpropagation '''    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        
        # we want the outputs from our inputs in batch, so we use the self.model
        # then we use the gather function to obtain the action that was chosen 
        # remember that the batch-state has a fake dimmension added with the unsqueeze function
        # we need to do the same with the batch action
        # we will add the fake dimmension corresponding to the actions with unsqueeze function
        # and we will make sure the total output is a commun variable 
        # not a tensor so we squeeze the whole at the index of the actions so index 1
        
        outputs = self.model( batch_state ).gather( 1, batch_action.unsqueeze(1) ).squeeze(1) 
        # 1 since it is the indice of the fake dimmension of the action 
        
        # now we need to calculate the next output in order to perform the loss function 
        # just remember that we need the maximum of the Q value of all possible next-states. 
        # to do so we use the detach function to extract all those possible values. 
        # We then can calculate the max
        # the correct index for that is 1 since it corresponds to the actions
        # we also need to specify we are taking the Q value for the next state
        # (q(a,st+1) and the index for that is 0 (the index for states is 0))
        
        next_outputs = self.model( batch_next_state ).detach().max(1)[0]
        
        # with the next-output calculated we can now compute the target,
        # which is comming from the reward (state-t) plus 
        # gamma times the next-output variable we just computed
        
        target = self.gamma * next_outputs + batch_reward
        
        # now wa can compute the loss function (empirical-output - predicted output)
        # we will use the hubble loss from our functional module
        # recommended function to calculate the loss in deep-q-learning
        # is the smooth_l1_loss
        
        td_loss = F.smooth_l1_loss(outputs, target)
        
        
        # now we will compute the optimizer that we chose previously, the adam optimizer
        # we need to apply in the last error in order to perform SGD and update the weights
        # in pytorch we need to re-initialize it at each iteration of the loop
        # to do so we use the zero_grad method
        
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True) 
        # we set it to True to improve the performance of the algorithm 
        
        # now we update the weights
        self.optimizer.step()
        
 
 ```
In the update function the main goal is to obtain the weights. They are calulated from the loss function with respect to the actions giving the highest reward. Therefore, anytime the agent reaches a new state we need to update the model. This means that once an action is selected we need to update all the actions of the transition/batch.
We then append this to our memory and follow-up the reward to see how the training is going.
In sum, we use the select-function and we integrate it with the update function to choose the best action.

```python

def update(self, reward, new_signal):

        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        
        # update the memory 
        # append the transition
        self.memory.push( ( self.last_state, new_state, torch.LongTensor( [ int( self.last_action ) ] ), torch.Tensor( [ self.last_reward ] ) ) )
        
        # we now play an action by using the select action function
        # then we use the new state as the argument
        action = self.select_action(new_state)
        
        # now we need to learn from the selected action
        # we make the ai learn from the last 100 events
        # we make sure we are selecting 100 events by using the "if" function
        # we have a batch of 100 000 and we then take a random sample of 100 events
        # we use the memory (object of the replaymemory class) 
        # and the second memory which is the object of the attribute
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        
        # update with the new parameters
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        
        # append the reward to the reward window (reward of the batch)
        # make sure the length is correct
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    
    # compute the mean of all the rewards stored in the reward window
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.) 
        # the + 1 is to make sure the denominator is not 0
    
    
    # let's save the model for any future models
    # save the brain of the car
    # we can then load the last version of the model
    # we want to save the optimizer and the weights
    # we will use a dictionary
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),  
        # the .state_dict() is to save the parameters 
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth') 
        # we use a pth file to save the model. 
        # Pth files add additional locations sys.path
    
    
    # take what we save and use it again. Useful for long training experiments
    # make sure the file exists
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            # now we need to update the files
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
    
    
    

````
# [Deep Q-learning implementation part 2]({{ site.baseurl }}/Deep-Q-Learning-Implementation-part-2/)  [<img src="{{ site.baseurl }}/images/404.jpg" alt="Constructocat by https://github.com/jasoncostello" style="width: 200px;"/>]({{ site.baseurl }}/Deep-Q-Learning-Implementation-part-2/)
