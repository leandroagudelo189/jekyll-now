---
published: true
---

Intuition on resticted boltzmann machines (study notes)

----
****

![an image alt text]({{ leandroagudelo189.github.io/tree/master }}/images/Boltzmann-machines.jpg "an image title")
----
****

![an image alt text]({{ leandroagudelo189.github.io/tree/master }}/images/boltzmann-machines-2.jpg "an image title")
----
****

![an image alt text]({{ leandroagudelo189.github.io/tree/master }}/images/boltzmann-machines-3.jpg "an image title")

----
****


### Further reading

[Energy-based learning](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf), by Yann LeCun et al. 

[A fast learning algorithm for deep belief networks](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf), by Hinton et al.

[Contrasting divergence](http://www.robots.ox.ac.uk/~ojw/files/NotesOnCD.pdf), by Woodford et al.

[Training of deep belief networks](http://www.iro.umontreal.ca/~lisa/pointeurs/BengioNips2006All.pdf), by Bengio et al.

[The wake-sleep algorithm for unsupervised neural networks](http://www.gatsby.ucl.ac.uk/~dayan/papers/hdfn95.pdf), by Hinton et al.

[Deep boltzmann machines](http://www.utstat.toronto.edu/~rsalakhu/papers/dbm.pdf), by Salakhutdinov et al.

----
****



### Implementation of RBM for recommendation-systems


- import the corresponding libraries
- in case of error during torch import, update numpy to 1.13 version
- test your version and path as follows:

```python
import numpy as np
print (np.__version__)
print (np.__path__)
```

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
