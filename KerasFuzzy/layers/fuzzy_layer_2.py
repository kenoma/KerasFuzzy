from keras import backend as K
from tensorflow import keras
import tensorflow as tf

class FuzzyLayer2(keras.layers.Layer):
    """Fuzzy layer with scaling and rotations
       mu(x,a,c) = exp(-|| a . x ||^2)
       initial_centers - each row is fuzzy term centroid
    """
    def __init__(self, 
                 output_dim, 
                 initial_centers=None,
                 initial_scales=None, 
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.output_dim = output_dim
        self.initial_centers = initial_centers
        self.initial_scales = initial_scales
        super(FuzzyLayer2, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dimensions = input_shape[-1]
        self.batch_size = input_shape[0]
        c_init =[]
        
        if self.initial_centers is None:
            c_init= tf.random_uniform_initializer(-1, 1)(shape=(self.output_dim, input_shape[-1]), dtype="float32") #, trainable=False
        else:
            c_init = -tf.convert_to_tensor(self.initial_centers, dtype="float32")
        
        s_init = []
        if self.initial_scales is None:
            s_init = tf.ones_initializer()(shape=(self.output_dim, input_shape[-1]), dtype="float32")    
        else:
            s_init = tf.convert_to_tensor(self.initial_scales, dtype="float32")
        
        s = s_init
        c = c_init

        index = 0
        translate_and_rotate = []
        for diag_scale in tf.unstack(tf.linalg.diag(s)):
            translate_and_rotate.append(tf.concat([diag_scale, tf.reshape(c[index,:],(self.input_dimensions,1))], 1))
            index = index + 1
        self.R =tf.Variable(tf.convert_to_tensor(translate_and_rotate, dtype="float32"), trainable=True)

        self.dummy_row = tf.concat([tf.zeros(shape=(1, input_shape[-1]), dtype="float32"), tf.ones(shape=(1, 1), dtype="float32")], 1)
        self.dummy_one = tf.ones(shape=(1))
        self.paddings = tf.constant([[0, 0,], [0, 1]])
        super(FuzzyLayer2, self).build(input_shape)  

    def call(self, batch_x):
        padded_batch = tf.pad(batch_x, self.paddings, "CONSTANT",constant_values = 1)
        retval = []
        for r in tf.unstack(self.R):
            padded_r = tf.concat([r, self.dummy_row],0)
            tran_and_rot = tf.linalg.matvec(padded_r, padded_batch)
            upper = -(tf.reduce_sum(tf.square(tran_and_rot),1)-1)
            exponent =tf.transpose(tf.exp(upper))
            retval.append(exponent)

        return tf.transpose(retval)


    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.output_dim,)

    def get_config(self):
        return {"R": self.R.numpy()}