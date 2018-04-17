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

Now, we import the DQN class and set the parameters for some conditions in the game. For more information on kivy applications please visit 