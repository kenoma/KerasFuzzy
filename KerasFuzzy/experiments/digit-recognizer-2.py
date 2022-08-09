#%%

import sys
sys.path.insert(0, 'D:/projects/KerasFuzzy/KerasFuzzy/layers')
from fuzzy_layer_2 import FuzzyLayer2
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
g = plt.imshow(X_train[0][:,:,0])
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

encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(64, 3, activation="relu", padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.Conv2D(32, 5, activation="relu", strides=2, padding="same")(x)
shape_before_flattening = K.int_shape(x)
x = layers.Flatten()(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()
#%%
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(np.prod(shape_before_flattening[1:]), activation="relu")(latent_inputs)
x = layers.Reshape(shape_before_flattening[1:])(x)
x = layers.Conv2DTranspose(32, 5, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()
#%%
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
#%%
vae = VAE(encoder, decoder, name="vae")
vae.compile(optimizer=keras.optimizers.Adam())
log_dir = "d:/projects/KerasFuzzy/logs/vae_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
mnist_digits = np.concatenate([X_train, X_val], axis=0)
vae.fit(mnist_digits, epochs=40, batch_size=140, callbacks=[tensorboard_callback])
#%%
def plot_label_clusters(vae, data, labels):
    z_means, _, _ = vae.encoder.predict(data)
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
    ax[0].scatter(z_means[:, 0], z_means[:, 1], c=labels)
    ax[0].set_xlabel("z[0]")
    ax[0].set_xlabel("z[1]")
    ax[1].scatter(z_means[:, 2], z_means[:, 1], c=labels)
    ax[1].set_xlabel("z[2]")
    ax[1].set_xlabel("z[1]")
    plt.show()

plot_label_clusters(vae, X_train, [np.argmax(a) for a in Y_train])
plot_label_clusters(vae, X_val, [np.argmax(a) for a in Y_val])

# %%
base_model = keras.Model(encoder_inputs, z_mean)
base_model.trainable = False

fuzzy_centroids = 81
z_means, _, _ = vae.encoder.predict(X_train)
init_c = random.sample(list(z_means), fuzzy_centroids)
init_s = np.empty((fuzzy_centroids, latent_dim))
init_s.fill(0.1)

x = base_model(encoder_inputs, training = False)
x = FuzzyLayer2(fuzzy_centroids, initial_centers=init_c, initial_scales = init_s, name="fuzzy")(x)
x = DefuzzyLayer(fuzzy_centroids, name="defuzzy")(x)
x = layers.Dense(10, activation="softmax")(x)
model = keras.Model(encoder_inputs, x)

optimizer = keras.optimizers.RMSprop(learning_rate=0.003, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
model.summary()
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.8, 
                                            min_lr=0.000001)
epochs = 2
batch_size = 86
datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=13,  
        zoom_range = 0.05, 
        width_shift_range=0.05,  
        height_shift_range=0.05,  
        horizontal_flip=False,  
        vertical_flip=False)  
datagen.fit(X_train)
log_dir = "d:/projects/KerasFuzzy/logs/main_phase_1_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
history = model.fit(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, 
                              validation_data = (X_val,Y_val),
                              verbose = 1, 
                              steps_per_epoch=X_train.shape[0] // batch_size, 
                              callbacks=[learning_rate_reduction, tensorboard_callback])
#%%
base_model.trainable = True
optimizer = keras.optimizers.RMSprop(learning_rate=0.003, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
model.summary()
epochs = 200
batch_size = 86
checkpoint_filepath = 'weights.{epoch:02d}-{val_loss:.2f}.h5'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
callback=keras.callbacks.EarlyStopping(
    monitor='val_accuracy', min_delta=0, patience=40, verbose=2, mode='auto',
    baseline=None, restore_best_weights=True)
log_dir = "d:/projects/KerasFuzzy/logs/main_phase_2_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
history = model.fit(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, 
                              validation_data = (X_val,Y_val),
                              verbose = 1, 
                              steps_per_epoch=X_train.shape[0] // batch_size, 
                              callbacks=[
                                  learning_rate_reduction, 
                                  tensorboard_callback, 
                                  #model_checkpoint_callback, 
                                  callback])
#%%
plot_label_clusters(vae, X_train, [np.argmax(a) for a in Y_train])
plot_label_clusters(vae, X_val, [np.argmax(a) for a in Y_val])
#%%
learned_centroids = []
weights = model.get_layer('fuzzy').get_weights()
for odim in range(fuzzy_centroids):
    origin = np.dot(np.vstack([weights[0][odim], np.array([0,0,0, 1])]), np.array([0,0,0, 1]))
    e1 = np.dot(np.vstack([weights[0][odim], np.array([0,0,0, 1])]), np.array([1,0,0, 1]))
    e2 = np.dot(np.vstack([weights[0][odim], np.array([0,0,0, 1])]), np.array([0,1,0, 1]))
    e3 = np.dot(np.vstack([weights[0][odim], np.array([0,0,0, 1])]), np.array([0,0,1, 1]))
    me1 = np.dot(np.vstack([weights[0][odim], np.array([0,0,0, 1])]), np.array([-1,0,0, 1]))
    me2 = np.dot(np.vstack([weights[0][odim], np.array([0,0,0, 1])]), np.array([0,-1,0, 1]))
    me3 = np.dot(np.vstack([weights[0][odim], np.array([0,0,0, 1])]), np.array([0,0,-1, 1]))
    plt.plot([-origin[0], -e1[0]], [-origin[1], -e1[1]], c = 'b', linewidth=2)
    plt.plot([-origin[0], -e2[0]], [-origin[1], -e2[1]],  c = 'b',linewidth=2)
    plt.plot([-origin[0], -e3[0]], [-origin[1], -e3[1]],  c = 'b',linewidth=2)
    plt.plot([-origin[0], -me1[0]], [-origin[1], -me1[1]],  c = 'b',linewidth=2)
    plt.plot([-origin[0], -me2[0]], [-origin[1], -me2[1]],  c = 'b',linewidth=2)
    plt.plot([-origin[0], -me3[0]], [-origin[1], -me3[1]],  c = 'b',linewidth=2)
    learned_centroids.append(origin)

plt.scatter([a[0] for a in learned_centroids], [a[1] for a in learned_centroids], alpha=0.9, s=2)
#%%
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

Y_pred = model.predict(X_val)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.argmax(Y_val,axis = 1) 
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
plot_confusion_matrix(confusion_mtx, classes = range(10)) 

errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)
most_important_errors = sorted_dela_errors[-6:]
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
results = model.predict(test)

results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"), results],axis = 1)
submission.to_csv("cnn_mnist_fuzzy_b.csv", index=False)
# %%
