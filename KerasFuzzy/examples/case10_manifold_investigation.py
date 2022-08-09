#%%
import sys
sys.path.insert(0, '../layers')
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from fuzzy_layer import FuzzyLayer
from tensorflow.python.client import device_lib
from keras.utils import to_categorical
print(device_lib.list_local_devices())
#%%

input_img = Input(shape=(784,))
model = Dense(256)(input_img)
model = Dense(2)(model)
f_layer = FuzzyLayer(100)
model = f_layer(model)
model = Dense(10, activation='softmax')(model)
mnist_classifier = Model(input_img, model)

#%%
mnist_classifier.compile(
        optimizer='adagrad',
        loss='categorical_crossentropy',
        metrics=['mae','acc'])
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print(x_train.shape)
print(x_test.shape)

mnist_classifier.fit(x_train, y_train,
                epochs=10,
                batch_size=48,
                shuffle=True,
                validation_data=(x_test, y_test))

#%%
weights = f_layer.get_weights()

plt.ion()
plt.show()
plt.clf()
plt.title('f')
plt.ylabel('s')
plt.xlabel('x')

tmpy = []
tmpx = []
for i in range(0, 100):
        tmpy.append(weights[0][0][i])
        tmpx.append(weights[0][1][i])

plt.scatter(tmpx, tmpy)   
plt.show()


#%%
