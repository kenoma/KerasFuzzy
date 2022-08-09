import tensorflow as tf
from tensorflow import keras
from keras import backend as K

class DefuzzyLayer2(keras.layers.Layer):

    def __init__(self, 
                 output_dim, 
                 initial_rules_outcomes=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.output_dim = output_dim
        self.initial_rules_outcomes = initial_rules_outcomes
        super(DefuzzyLayer2, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dimensions = list(input_shape)[:-1:-1]

        outcomes_init_values = []
        if self.initial_rules_outcomes is None:
            outcomes_init_values = tf.random_uniform_initializer()(shape=(input_shape[-1], self.output_dim), dtype="float32")    
        else:
            outcomes_init_values = tf.convert_to_tensor(self.initial_rules_outcomes, dtype="float32")
        
        self.rules_outcome = tf.Variable(initial_value = outcomes_init_values, trainable=True)
        self.bias = tf.Variable(initial_value = tf.zeros_initializer()(shape=(input_shape[-1], self.output_dim), dtype="float32"), trainable=True)
        self.rules_outcome_2 = tf.Variable(initial_value = tf.zeros_initializer()(shape=(input_shape[-1], self.output_dim), dtype="float32"), trainable=True)
        
        super(DefuzzyLayer2, self).build(input_shape)  

    def call(self, x):
        aligned_x = K.repeat_elements(K.expand_dims(x, axis = -1), self.output_dim, -1)
        aligned_bias = self.bias
        aligned_rules_outcome = self.rules_outcome
        aligned_rules_outcome_2 = self.rules_outcome_2
        
        for dim in self.input_dimensions:
            aligned_bias = K.repeat_elements(K.expand_dims(aligned_bias, 0), dim, 0)
            aligned_rules_outcome = K.repeat_elements(K.expand_dims(aligned_rules_outcome, 0), dim, 0)
            aligned_rules_outcome_2 = K.repeat_elements(K.expand_dims(aligned_rules_outcome_2, 0), dim, 0)
        
        xc = K.sum((aligned_x * aligned_x * aligned_rules_outcome_2 + aligned_x * aligned_rules_outcome + aligned_bias), axis=-2, keepdims=False)
        return xc
        
    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.output_dim,)

    def get_config(self):
        return {
            "rules_outcome": self.rules_outcome.numpy(),
            "aligned_rules_outcome_2":self.rules_outcome_2.numpy(),
            "aligned_bias":self.bias.numpy()
        }