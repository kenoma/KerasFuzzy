from DeFuzzyLayer import DefuzzyLayer
import unittest
from keras import backend as K

class DefuzzyLayerTest(unittest.TestCase):

    def defuzzy_test_layer_single(self):
        odim = 2
        input_dim = 2
        batch = 1
        sess = K.get_session()

        layer = DefuzzyLayer(odim)
        layer.build(input_shape=(batch,input_dim))
        
        x = K.placeholder(shape=(batch, input_dim))
        rules_outcome = K.placeholder(shape=(input_dim, odim))
        
        layer.rules_outcome = rules_outcome
        layer.a = a
        xc = layer.call(x)

        xx = [[1, 0]]
        rr = [[1, 0], [0, 1]]
        
        vals = sess.run(xc, feed_dict={x: xx, rules_outcome:rr})
        self.assertEqual(len(vals), batch)
        self.assertEqual(len(vals[0]), odim)
        self.assertAlmostEqual(vals[0][0], 1, 7)
        self.assertAlmostEqual(vals[0][1], 0, 7)
        

if __name__ == '__main__':
    unittest.main()