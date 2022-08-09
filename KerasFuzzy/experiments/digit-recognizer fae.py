#%%
import sys
sys.path.insert(0, 'D:/projects/KerasFuzzy/KerasFuzzy/layers')
from fuzzy_layer_2 import FuzzyLayer2
from fuzzy_layer import FuzzyLayer
from defuzzy_layer_2 import DefuzzyLayer2
from defuzzy_layer import DefuzzyLayer

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
np.random.seed(2)
random_seed = 2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K 
import random
import itertools
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
#%%
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#%%
train = pd.read_csv("./digit-recognizer/train.csv")
test = pd.read_csv("./digit-recognizer/test.csv")

Y_train = train["label"]

X_train = train.drop(labels = ["label"], axis = 1) 
del train 
#%%
X_train = X_train / 255.0
test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
#%%
Y_train = to_categorical(Y_train, num_classes = 10)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

#%%
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

#%%
latent_dim = 3
fuzzy_centroids = 10

encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(64, 3, activation="relu", padding="same")(encoder_inputs)
#x = layers.Dropout(0.1)(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
#x = layers.Dropout(0.1)(x)
x = layers.Conv2D(32, 5, activation="relu", strides=2, padding="same")(x)
#x = layers.Dropout(0.1)(x)
shape_before_flattening = K.int_shape(x)
x = layers.Flatten()(x)
x = layers.Dense(512)(x)
#x = layers.Dropout(0.1)(x)
x = layers.Dense(256)(x)
#x = layers.Dropout(0.1)(x)
x = layers.Dense(64)(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
fz_c = FuzzyLayer2(fuzzy_centroids, initial_centers=[
    [0.8090169943, 0.5877852524,0], 
    [0.3090169938, 0.9510565165,0], 
    [-0.3090169938, 0.9510565165,0], 
    [-0.8090169943, 0.5877852524,0], 
    [-1., 0.,0], 
    [-0.8090169943, -0.5877852524,0], 
    [-0.3090169938, -0.9510565165,0], 
    [0.3090169938, -0.9510565165,0], 
    [0.8090169943, -0.5877852524,0], 
    [1., 0.,0]])

#fz = DefuzzyLayer2(3)(z_mean)
fz = fz_c(z_mean)
#fz = layers.Dense(10 ,activation="relu")(fz)
fz = tf.keras.layers.Softmax()(fz)
#fz = DefuzzyLayer2(fuzzy_centroids)(fz)

#fz = FuzzyLayer2(fuzzy_centroids)(fz)
#fz = DefuzzyLayer2(fuzzy_centroids)(fz)
#fz = tf.keras.layers.Softmax()(fz)
#fz = FuzzyLayer2(fuzzy_centroids, name="fuzzy")(z_mean)
#fz = layers.Dense(fuzzy_centroids, activation="softmax")(fz)
#z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
#z = Sampling()([z_mean, z_log_var])
#fz = DefuzzyLayer(fuzzy_centroids)(fz)

encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z, fz], name="encoder")
encoder.summary()
#%%
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(latent_dim)(latent_inputs)
x = layers.Dense(64)(x)
#x = layers.Dropout(0.1)(x)
x = layers.Dense(256)(x)
#x = layers.Dropout(0.1)(x)
x = layers.Dense(512)(x)
x = layers.Dense(np.prod(shape_before_flattening[1:]), activation="relu")(x)
x = layers.Reshape(shape_before_flattening[1:])(x)
x = layers.Conv2DTranspose(32, 5, activation="relu", strides=2, padding="same")(x)
#x = layers.Dropout(0.1)(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
#x = layers.Dropout(0.1)(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()
#%%
class FAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(FAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        #self.max_firing_loss_tracker = keras.metrics.Mean(name="mxf_loss")
        #self.sum_firing_loss_tracker = keras.metrics.Mean(name="sum_loss")
        self.fz_loss_tracker = keras.metrics.Mean(name="fz_loss")
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            #self.max_firing_loss_tracker,
            #self.sum_firing_loss_tracker,
            self.fz_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, fz = self.encoder(data[0])
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data[0], reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            fz_loss = K.sum(K.categorical_crossentropy(data[1], fz), axis=-1)
            total_loss =  kl_loss  + reconstruction_loss + fz_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.fz_loss_tracker.update_state(fz_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "fz_loss": self.fz_loss_tracker.result(),
        }

fae = FAE(encoder, decoder, name="fae")

def lr_scheduler(epoch, lr):
    if epoch % 5 == 1:
        lr = lr * 0.8
    return lr
    
fae.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.001))
log_dir = f"d:/projects/KerasFuzzy/logs/fae_{latent_dim}_{fuzzy_centroids}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}" #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#mnist_digits = np.concatenate([X_train, X_val], axis=0)
fae.fit(X_train, Y_train, 
    epochs=30, 
    batch_size=86, 
    callbacks=[
        tensorboard_callback, 
        LearningRateScheduler(lr_scheduler, verbose=1),
        EarlyStopping(monitor='reconstruction_loss', patience = 5)])
#%%
fae = keras.Model(encoder_inputs, z_mean)
fae_class = keras.Model(encoder_inputs, fz)

def plot_label_clusters(fae, data, labels):
    z_means = fae.predict(data)
    fig, ax = plt.subplots(ncols=2, nrows=int(latent_dim/2), figsize=(12, 6), squeeze=False)
    for odim in range(latent_dim - 1):
        ax[int(odim/2), odim % 2].scatter(z_means[:, 0], z_means[:, odim + 1], c=labels)
        ax[int(odim/2), odim % 2].set_xlabel("z[0]")
        ax[int(odim/2), odim % 2].set_ylabel(f"z[{odim + 1}]")
    
    plt.show()

z_classes = fae_class.predict(X_val)
z_classes = z_classes.argmax(axis=-1)

labels = [np.argmax(a) for a in Y_val]
dlab = [ (-1 if a[0] == a[1] else a[0]) for a in zip(labels, z_classes)]
plot_label_clusters(fae, X_val, labels)
plot_label_clusters(fae, X_val, dlab)

#%%
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(labels, z_classes)
print(cf_matrix)

#%%
fae_decoder = keras.Model(latent_inputs, decoder_outputs)

fig, ax = plt.subplots(int(fuzzy_centroids/2), 2, sharex=True, sharey=True, figsize = (5,12), squeeze=False)

c = tf.gather(fz_c.R, latent_dim, axis=2)
for odim in range(fuzzy_centroids):
    inp = c[odim].numpy()
    decoded_img = fae_decoder.predict(inp.reshape((-1,latent_dim)))
    ax[int(odim/2),odim % 2].imshow(decoded_img.reshape((28,28)), interpolation='nearest', aspect='auto')
    #ax[int(fc/2),fc % 2].set_title(f"Channel {fc}")
#%%
fae_class = keras.Model(encoder_inputs, fz)
fae_class.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
fae_class.evaluate(X_val, Y_val)
#%%