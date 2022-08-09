Main goal of this project is to provide trainable representations of real-word input data to regular neural networks. 
There are two classes implemented so far: `FuzzyLayer` and `DefuzzyLayer`.

## FuzzyLayer

This layer is suitable for cases when you working with data that can be clustered into interpretable  groups e.g. spatial coordinates, multi-function values and etc.

Membership function for this layer have form:

<img id="image" src="http://mathurl.com/ybkw2ohp.png" alt="\mu_{{j}} \left( x,c,a \right) ={e}^{-\sum _{i=0}^{\dim}1/4\,{\frac {
 \left( x_{{i}}-c_{{i}} \right) ^{2}}{{a_{{i}}}^{2}}}}
" style="border: 0; padding: 1ex 2ex 1ex 2ex">

where `x` is an input vector of `dim` length, `c` is centroid of j-th membership function and `a` is vector of scaling factors.

## DefuzzyLayer

`DefuzzyLayer` can be trained to transform output of an model to continuos values. In other words this layer can be interpreted as an ruleset and input to this layer - firing levels for rules. 

<img id="image" src="http://mathurl.com/yabcgzn9.png" alt="d \left( x,r \right) =\sum _{i=0}^{{\it input\_dim}}x_{{i}}r_{{i}}" style="border: 0; padding: 1ex 2ex 1ex 2ex">

## FuzzyLayer 2

Membership function for layer `FuzzyLayer2` have form $\mu(x, A) = e^{ -|| \[A . \~x\]_{1 \cdots m} ||^2}$ where $m$ is task dimension,  $A$ is transformation matrix in form 

![image](https://user-images.githubusercontent.com/6205671/170839478-2c80ba81-1ea5-40c3-a9cb-350f4cf1f9d5.png)

with $c_{1\cdots m}$ - centroid, 
$s_{1\cdots m}$ - scaling factor, 
$a_{1\cdots m, 1\cdots m}$ - alignment coefficients and 
$x$ is an extended with $1$ vector 
$x = [x_1, x_2, \cdots, x_m, 1]$.

Main benefit of `FuzzyLayer2` over `FuzzyLayer` is that fuzzy centroids are aligned in arbitrary direction to cover cluster structures more preciesly.

## Basic sample

```python
import keras
from FuzzyLayer import FuzzyLayer
from DefuzzyLayer import DefuzzyLayer
from keras.models import Sequential

...

model = Sequential()
model.add(FuzzyLayer(20, input_dim=2))
model.add(DefuzzyLayer(1))

model.compile(loss='logcosh',
              optimizer='rmsprop',
              metrics=['mae'])

model.fit(x_train, y_train,
          epochs=500,
          verbose=1,
          batch_size=100)
```

## MNIST classification with neuro-fuzzy model

```python
latent_dim = 3

mnist_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(64, 3, activation="relu", padding="same")(mnist_inputs)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.Conv2D(32, 5, activation="relu", strides=2, padding="same")(x)
shape_before_flattening = K.int_shape(x)
x = layers.Flatten()(x)
z = layers.Dense(latent_dim, name="z")(x)
base_model = keras.Model(mnist_inputs, z)

x = base_model(mnist_inputs)
x = FuzzyLayer2(10, name="fuzzy")(x)
x = DefuzzyLayer(10, name="defuzzy")(x)
x = tf.keras.layers.Softmax()(x)
fuzzy_model = keras.Model(mnist_inputs, x)
```

Accuracy achieved by training `fuzzy_model` is about 0.995 and presented model has nice clustered latent layer `z` structure:

![b1](https://user-images.githubusercontent.com/6205671/173923018-b2edecfe-dcab-4da7-83e8-9eb79fd61b36.png)

## Usage examples

[MNIST fuzzy anomaly detection with VAE](https://github.com/kenoma/KerasFuzzy/blob/master/KerasFuzzy/experiments/digit-recognizer-anomaly-detection-fae.ipynb)
[MNIST fuzzy anomaly detection with CVAE](https://github.com/kenoma/KerasFuzzy/blob/master/KerasFuzzy/experiments/digit-recognizer-anomaly-detection.ipynb)
