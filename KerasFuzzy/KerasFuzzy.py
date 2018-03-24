from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.batch = 1 if input_shape[0] is None else input_shape[0]
        self.c = self.add_weight(name='c', 
                                 shape=(input_shape[1], self.output_dim),
                                 initializer='ones',
                                 trainable=True)
        self.a = self.add_weight(name='a', 
                                 shape=(input_shape[1], self.output_dim),
                                 initializer='ones',
                                 trainable=True)

        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):

        aligned_x = K.repeat_elements(K.expand_dims(x, axis = 2), self.output_dim, 2)
        aligned_c = K.repeat_elements(K.expand_dims(self.c, 0), self.batch, 0)
        aligned_a = 2 * K.square(K.repeat_elements(K.expand_dims(self.a, 0), self.batch, 0))
        xc = K.exp(-K.sum(K.square(aligned_x - aligned_c) / aligned_a, axis=1, keepdims=False))
        
        return xc
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

