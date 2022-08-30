#%%
import sys
sys.path.insert(0, '../layers')
from ast import Assert
from fuzzy_layer import FuzzyLayer
from fuzzy_layer_indep import FuzzyLayerIndep
from keras.models import Sequential
import numpy as np
import tensorflow as tf


fuzzy_layer = FuzzyLayer(output_dim=4, input_dim=2)

x = tf.random.uniform((500, 2))
y = fuzzy_layer(x)
# %%
assert fuzzy_layer.weights == [fuzzy_layer.c, fuzzy_layer.a]
# %%
print("weights:", len(fuzzy_layer.weights))
print("non-trainable weights:", len(fuzzy_layer.non_trainable_weights))
print("trainable_weights:", fuzzy_layer.trainable_weights)

# %%
model = Sequential()
model.add(fuzzy_layer)
model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'acc'])
model.fit(x,  np.array( [([1,0,0,0] if a[0]<0.5 and a[1]<0.5 else
                          [0,1,0,0] if a[0]<0.5 and a[1]>0.5 else
                          [0,0,1,0] if a[0]>0.5 and a[1]<0.5 else 
                          [0,0,0,1]) for a in x]),
          epochs=1000,
          verbose=0,
          batch_size=10)

print("trainable_weights:", fuzzy_layer.trainable_weights)
# %%
assert np.argmax(fuzzy_layer([0.0, 0.0])) == 0
# %%
assert np.argmax(fuzzy_layer([0.0, 1.0])) == 1
# %%
assert np.argmax(fuzzy_layer([1.0, 0.0])) == 2
# %%
assert np.argmax(fuzzy_layer([1.0, 1.0])) == 3
# %%

x = tf.random.uniform((500, 2))
r = tf.ones([2])

xx = x * (1.0 - x) * r
# %%
from logistic_map_layer import LogisticMapLayer

logmap_layer = LogisticMapLayer()

y = logmap_layer(x)
# %%

import sys
sys.path.insert(0, '../layers')
from ast import Assert
from fuzzy_layer_indep import FuzzyLayerIndep
from keras.models import Sequential
import numpy as np
import tensorflow as tf

x = tf.Variable([[1.0, 1.0],
                 [2.0, 2.0],
                 [3.0, 3.0]])


min_max = [1, 3]


findep = FuzzyLayerIndep(4, min_max)
y = findep(x)
y