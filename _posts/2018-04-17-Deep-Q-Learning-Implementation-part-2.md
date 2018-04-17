---
published: true
---


![an image alt text]({{ leandroagudelo189.github.io/tree/master}}/images/deep-q-implementation.jpg "an image title")


# B. Agent Environment

Let's import the libraries and the kivy packages to create the environment

```python

import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

```

# 1. Import the Deep Q-learning class

Now, we import the DQN class and set the parameters for some conditions in the game. For more information on kivy applications please visit [Kivy tutorials](https://kivy.org/docs/tutorials/pong.html/"title" target="_blank").
```python
# Importing the Dqn object 
from ai import Dqn

# Adding this line to customize the window of application
# and the user interface with the mouse
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'width', '1200')
Config.set('graphics', 'height', '800')
# Introducing last_x and last_y, used to keep the last point in memory

last_x = 0
last_y = 0
n_points = 0
length = 0

# Create the object of the DQN class. Called here "brain"
# contains our neural network that represents our Q-function
# arguments are: 5 is the number of sensors, 3 the actions and gamma = 0.9
# please see bellman equation and MDP in "Deep q learning notes"
brain = Dqn(5,3,0.9)

# action = 0 => no rotation, action = 1 => rotate 20 degres, 
# action = 2 => rotate -20 degres
action2rotation = [0,20,-20]
last_reward = 0

# initialize the mean score curve or the window
# of rewards with respect to time
scores = []

# Initializing the map only once
# get coordinates and destination
first_update = True
def init():

    # sand is an array that has as many cells as our graphic interface has pixels. 
    # Each cell has a one if there is sand, 0 otherwise.
    global sand
    
    # x and y-coordinates of the goal 
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur,largeur))
    # upper left of the map
    goal_x = 20
    goal_y = largeur - 20
    first_update = False

# Initializing the last distance
# It gives the current distance from the car to the road
last_distance = 0

```
