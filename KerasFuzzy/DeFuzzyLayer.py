from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class DefuzzyLayer(Layer):

    def __init__(self, 
                 output_dim, 
                 initializer_centers = None,
                 initializer_sigmas = None, 
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.output_dim = output_dim
        self.initializer_centers = initializer_centers
        self.initializer_sigmas = initializer_sigmas
        super(DefuzzyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.batch = 1 if input_shape[0] is None else input_shape[0]
        self.rules_outcome = self.add_weight(name='rules_outcome', 
                                 shape=(input_shape[1], self.output_dim),
                                 initializer= self.initializer_centers if self.initializer_centers is not None else 'uniform',
                                 trainable=True)
        super(DefuzzyLayer, self).build(input_shape)  

    def call(self, x):
        aligned_x = K.repeat_elements(K.expand_dims(x, axis = 2), self.output_dim, 2)
        aligned_rules_outcome = K.repeat_elements(K.expand_dims(self.rules_outcome, 0), self.batch, 0)
        
        xc = K.sum((aligned_x * aligned_rules_outcome), axis=1, keepdims=False)
        return xc
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

