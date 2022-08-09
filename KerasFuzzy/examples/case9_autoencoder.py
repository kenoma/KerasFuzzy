#%%
import sys
sys.path.insert(0, '../layers')
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from fuzzy_layer import FuzzyLayer
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
#%%
fuzzy_kernels = 10
fuzzy_inputs = 2
centroids_init_values= tf.random_uniform_initializer(0, 1)(shape=(fuzzy_inputs, fuzzy_kernels), dtype="float32")    
sigma_init_values = tf.constant_initializer(1e-1)(shape=(fuzzy_inputs, fuzzy_kernels), dtype="float32")    

input_img = Input(shape=(784,))
encoded = Dense(32, activation='sigmoid')(input_img)
encoded = Dense(fuzzy_inputs, activation='sigmoid')(encoded)

f_layer = FuzzyLayer(fuzzy_kernels,
                initial_centers=centroids_init_values,
                initial_sigmas=sigma_init_values)
encoded = f_layer(encoded)

decoded = Dense(fuzzy_inputs, activation='sigmoid')(encoded)
decoded = Dense(32, activation='sigmoid')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

encoded_input = Input(shape=(fuzzy_kernels,))
decoder_layer3 = autoencoder.layers[-3]
decoder_layer2 = autoencoder.layers[-2]
decoder_layer1 = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer1(decoder_layer2(decoder_layer3(encoded_input))))

#%%
autoencoder.compile(optimizer='adam', 
    loss='binary_crossentropy', metrics=['mae'])
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, x_train,
                epochs=300,
                batch_size=16                                                     ,
                shuffle=True,
                validation_data=(x_test, x_test))

#%% encode and decode some digits
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

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
for i in range(0, fuzzy_kernels):
        tmpy.append(weights[0][0][i])
        tmpx.append(weights[0][1][i])

plt.scatter(tmpx, tmpy)   
plt.show()


#%%
