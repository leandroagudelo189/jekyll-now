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

Now, we import the DQN class and set the parameters for some conditions in the game. For more information on kivy applications please visit [Kivy tutorials](https://kivy.org/docs/tutorials/pong.html){:target="_blank"}. 


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


# 2. Class for the car/agent and the game

```python

# "NumericProperty" and "ReferenceListProperty", see kivy tutorials
class Car(Widget):
    
    '''parameters for the car'''
    angle = NumericProperty(0) 
    # initializing the angle of the car (angle between the x-axis of the map 
    # and the axis of the car)
    rotation = NumericProperty(0) 
    # initializing the last rotation of the car (after playing the action, 
    # the car does a rotation of 0, 20 or -20 degrees)
    velocity_x = NumericProperty(0) 
    # initializing the x-coordinate of the velocity vector
    velocity_y = NumericProperty(0) 
    # initializing the y-coordinate of the velocity vector
    velocity = ReferenceListProperty(velocity_x, velocity_y) # velocity vector
    sensor1_x = NumericProperty(0) 
    # initializing the x-coordinate of the first sensor (the one that looks forward)
    sensor1_y = NumericProperty(0) 
    # initializing the y-coordinate of the first sensor 
    # (the one that looks forward sensor1 = ReferenceListProperty(sensor1_x, sensor1_y) 
    # first sensor vector
    sensor2_x = NumericProperty(0)
    # initializing the x-coordinate of the second sensor 
    # (the one that looks 30 degrees to the left)
    sensor2_y = NumericProperty(0)
    # initializing the y-coordinate of the second sensor 
    # (the one that looks 30 degrees to the left)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y) 
    # second sensor vector
    sensor3_x = NumericProperty(0) 
    # initializing the x-coordinate of the third sensor
    # (the one that looks 30 degrees to the right)
    sensor3_y = NumericProperty(0)
    # initializing the y-coordinate of the third sensor 
    # (the one that looks 30 degrees to the right)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y) # third sensor vector
    
    '''Now, we will get the density of sand in each sensor.
    we take into account the squares of each sensor. The squares are 200by200.
    for each square we divide the number of ones in the square by the total number of cells in the square.
    This will be the density of signals in each sensors represented in each square'''
    signal1 = NumericProperty(0) # initializing the signal received by sensor 1
    signal2 = NumericProperty(0) # initializing the signal received by sensor 2
    signal3 = NumericProperty(0) # initializing the signal received by sensor 3


    # functionality
    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos 
        # updating the position of the car according to its last position and velocity
        self.rotation = rotation
        # getting the rotation of the car
        self.angle = self.angle + self.rotation
        # updating the angle
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos 
        # updating the position of sensor 1. 30 is the distance between 
        # the car and what the car detects
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos 
        # updating the position of sensor 2
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        # updating the position of sensor 3
        
        # once the sensors are updated we need to update the signals
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400. 
        # getting the signal received by sensor 1 (density of sand around sensor 1)
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400. 
        # getting the signal received by sensor 2 (density of sand around sensor 2)
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400. 
        # getting the signal received by sensor 3 (density of sand around sensor 3)
        
        ''' a penalty when reaching the edges of the car'''
        if self.sensor1_x > longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10: 
        # if sensor 1 is out of the map (the car is facing one edge of the map)
            self.signal1 = 1. # sensor 1 detects full sand or penalty
        if self.sensor2_x > longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10: 
        # if sensor 2 is out of the map (the car is facing one edge of the map)
            self.signal2 = 1. 
            # sensor 2 detects full sand
        if self.sensor3_x > longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
        # if sensor 3 is out of the map (the car is facing one edge of the map)
            self.signal3 = 1. 
            # sensor 3 detects full sand


class Ball1(Widget): 
# sensor 1 (see kivy tutorials: kivy https://kivy.org/docs/tutorials/pong.html)
    pass
class Ball2(Widget): 
# sensor 2 (see kivy tutorials: kivy https://kivy.org/docs/tutorials/pong.html)
    pass
class Ball3(Widget):
# sensor 3 (see kivy tutorials: kivy https://kivy.org/docs/tutorials/pong.html)
    pass

"ObjectProperty", see kivy tutorials

class Game(Widget):
    
    '''let's creat the object of the car'''
    car = ObjectProperty(None) # getting the car object from our kivy file
    ball1 = ObjectProperty(None) # getting the sensor 1 object from our kivy file
    ball2 = ObjectProperty(None) # getting the sensor 2 object from our kivy file
    ball3 = ObjectProperty(None) # getting the sensor 3 object from our kivy file

    '''starting the car when launchig the application'''
    def serve_car(self): 
    # starting the car when we launch the application
        self.car.center = self.center 
        # the car will start at the center of the map
        self.car.velocity = Vector(6, 0) 
        # the car will start to go horizontally to the right with a speed of 6
        
        
        
        
    '''one of the most important functions. It updates everythig in each state comming to the sensors '''   
    def update(self, dt): 
    # the big update function that updates everything that needs 
    # to be updated at each discrete time t when reaching a new state 
    # (getting new signals from the sensors)

        global brain 
        # specifying the global variables (the brain of the car, that is our AI)
        global last_reward 
        # specifying the global variables (the last reward)
        global scores 
        # specifying the global variables (the means of the rewards)
        global last_distance 
        # specifying the global variables (the last distance from the car to the goal)
        global goal_x 
        # specifying the global variables (x-coordinate of the goal)
        global goal_y 
        # specifying the global variables (y-coordinate of the goal)
        global longueur
        # specifying the global variables (width of the map)
        global largeur 
        # specifying the global variables (height of the map)

        longueur = self.width 
        # width of the map (horizontal edge)
        largeur = self.height
        # height of the map (vertical edge)
        if first_update: 
        # trick to initialize the map only once
            init()

        xx = goal_x - self.car.x 
        # difference of x-coordinates between the goal and the car
        yy = goal_y - self.car.y 
        # difference of y-coordinates between the goal and the car
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180. 
        # direction of the car with respect to the goal 
        #(if the car is heading perfectly towards the goal, then orientation = 0)
        
        '''the - orientation makes sure it explores in both directions'''
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        # our input state vector, composed of the three signals received 
        # by the three sensors, plus the orientation and -orientation
        # important as the inputs go into the neural network. 
        # After the action will take place followed by the corresponding update
        action = brain.update(last_reward, last_signal)
        # playing the action from our ai (the object brain of the dqn class)
        
        '''now we fully update the states and parameters of the car and the game'''
        scores.append(brain.score())
        # appending the score (mean of the last 100 rewards to the reward window)
        rotation = action2rotation[action]
        # converting the action played (0, 1 or 2) into the rotation angle (0°, 20° or -20°)
        self.car.move(rotation) 
        # moving the car according to this last rotation angle
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2) 
        # getting the new distance between the car and the goal right after the car moved
        self.ball1.pos = self.car.sensor1 
        # updating the position of the first sensor (ball1) right after the car moved
        self.ball2.pos = self.car.sensor2 
        # updating the position of the second sensor (ball2) right after the car moved
        self.ball3.pos = self.car.sensor3 
        # updating the position of the third sensor (ball3) right after the car moved

        
        '''this is where the penalty takes place'''
        if sand[int(self.car.x),int(self.car.y)] > 0: 
        # if the car is on the sand
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            # it is slowed down (speed = 1)
            last_reward = -2 # and it gets a low reward = -1
        else: # otherwise
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            # it goes to a normal speed (speed = 6)
            last_reward = -0.1
            # if it is getting away from the goal, it gets bad reward (-0.2)
            if distance < last_distance: 
            # however if it getting close to the goal
                last_reward = 0.5 
                # it still gets slightly positive reward 0.1
        
        
        '''when the car is close to the edges'''       
        if self.car.x < 10: 
        # if the car is in the left edge of the frame
            self.car.x = 10 
            # it is not slowed down
            last_reward = -1 
            # but it gets bad reward -1
        if self.car.x > self.width-10: 
        # if the car is in the right edge of the frame
            self.car.x = self.width-10
            # it is not slowed down
            last_reward = -1
            # but it gets bad reward -1
        if self.car.y < 10: 
        # if the car is in the bottom edge of the frame
            self.car.y = 10 
            # it is not slowed down
            last_reward = -1 
            # but it gets bad reward -1
        if self.car.y > self.height-10: 
        # if the car is in the upper edge of the frame
            self.car.y = self.height-10 
            # it is not slowed down
            last_reward = -1 
            # but it gets bad reward -1

        
        '''the goal is reached''' 
        if distance < 100: 
        # when the car reaches its goal
            goal_x = self.width - goal_x 
            # the goal becomes the bottom right corner of the map (the downtown), 
            # and vice versa (updating of the x-coordinate of the goal)
            goal_y = self.height - goal_y 
            # the goal becomes the bottom right corner of the map 
# (the downtown), and vice versa (updating of the y-coordinate of the goal)

        # Updating the last distance from the car to the goal
        last_distance = distance

```
