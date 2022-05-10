import unittest
from FuzzyLayer import FuzzyLayer
from DefuzzyLayer import DefuzzyLayer
import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


class FuzzyLayerTest(unittest.TestCase):

    def test_layer_output_single(self):
        odim = 2
        input_dim = 2
        batch = 1
        sess = K.get_session()

        layer = FuzzyLayer(odim)
        layer.build(input_shape=(batch,input_dim))
        
        x = K.placeholder(shape=(batch, input_dim))
        c = K.placeholder(shape=(input_dim, odim))
        a = K.placeholder(shape=(input_dim, odim))
        layer.c = c
        layer.a = a
        xc = layer.call(x)

        xx = [[1, 1]]
        cc = [[1, 0], [1, 0]]
        aa = [[1 / 10, 1 / 10], [1 / 10, 1 / 10]]
        vals = sess.run(xc, feed_dict={x: xx, c:cc, a:aa})
        self.assertEqual(len(vals), batch)
        self.assertEqual(len(vals[0]), odim)
        self.assertAlmostEqual(vals[0][0], 1, 1)
        self.assertAlmostEqual(vals[0][1], 0, 1)

    def test_layer_output_single2(self):
        odim = 2
        input_dim = 2
        batch = 1
        sess = K.get_session()

        layer = FuzzyLayer(odim)
        layer.build(input_shape=(batch,input_dim))
        
        x = K.placeholder(shape=(batch, input_dim))
        c = K.placeholder(shape=(input_dim, odim))
        a = K.placeholder(shape=(input_dim, odim))
        layer.c = c
        layer.a = a
        xc = layer.call(x)

        xx = [[0.5, 0.5]]
        cc = [[1, 0], [1, 0]]
        aa = [[1 / 2, 1 / 2], [1 / 2, 1 / 2]]
        vals = sess.run(xc, feed_dict={x: xx, c:cc, a:aa})
        self.assertEqual(len(vals), batch)
        self.assertEqual(len(vals[0]), odim)
        self.assertAlmostEqual(vals[0][0], 0.6065306597, 7)
        self.assertAlmostEqual(vals[0][1], 0.6065306597, 7)

    def test_layer_output_batched(self):
        odim = 2
        input_dim = 2
        batch = 3
        sess = K.get_session()

        layer = FuzzyLayer(odim)
        layer.build(input_shape=(batch,input_dim))
        
        x = K.placeholder(shape=(batch, input_dim))
        c = K.placeholder(shape=(input_dim, odim))
        a = K.placeholder(shape=(input_dim, odim))
        layer.c = c
        layer.a = a
        xc = layer.call(x)

        xx = [[1, 1], [0, 0], [0.5, 0.5]]
        cc = [[1, 0], [1, 0]]
        aa = [[1 / 10, 1 / 10], [1 / 10, 1 / 10]]
        vals = sess.run(xc, feed_dict={x: xx, c:cc, a:aa})
        self.assertEqual(len(vals), batch)
        self.assertEqual(len(vals[0]), odim)
        self.assertAlmostEqual(vals[0][0], 1, 7)
        self.assertAlmostEqual(vals[0][1], 0, 7)
        self.assertAlmostEqual(vals[1][0], 0, 7)
        self.assertAlmostEqual(vals[1][1], 1, 7)
        self.assertAlmostEqual(vals[2][0], 0.000003726653172, 7)
        self.assertAlmostEqual(vals[2][1], 0.000003726653172, 7)

    def test_layer_output_batched2(self):
        odim = 2
        input_dim = 2
        batch = 2
        sess = K.get_session()

        layer = FuzzyLayer(odim)
        layer.build(input_shape=(batch,input_dim))
        
        x = K.placeholder(shape=(batch, input_dim))
        c = K.placeholder(shape=(input_dim, odim))
        a = K.placeholder(shape=(input_dim, odim))
        layer.c = c
        layer.a = a
        xc = layer.call(x)

        xx = [[0.5, 0.8], [0.8, 0.5]]
        cc = [[1, 0.2], [0.8, 0]]
        aa = [[1 / 2, 1 / 4], [1, 1 / 8]]
        vals = sess.run(xc, feed_dict={x: xx, c:cc, a:aa})
        self.assertEqual(len(vals), batch)
        self.assertEqual(len(vals[0]), odim)
        self.assertAlmostEqual(vals[0][0], 0.7788007831, 7)
        self.assertAlmostEqual(vals[0][1], 0.00002491600973, 7)
        self.assertAlmostEqual(vals[1][0], 0.9394130628, 7)
        self.assertAlmostEqual(vals[1][1], 0.004339483271, 7)

    def test_layer_output_batched_and_context(self):

        input_dim = 1
        context = 2
        batch = 3
        odim = 4

        sess = K.get_session()

        layer = FuzzyLayer(odim)
        layer.build(input_shape=(batch, context, input_dim))
        
        x = K.placeholder(shape=(batch, context, input_dim))
        c = K.placeholder(shape=(input_dim, odim))
        a = K.placeholder(shape=(input_dim, odim))
        layer.c = c
        layer.a = a
        xc = layer.call(x)

        xx = [[[0.5], [0.8]],
              [[0.8], [0.6]],
              [[0.6], [0.4]]]

        cc = [[1, 0.8, 0.6, 0.4]]
        aa = [[1, 1, 1, 1]]
        vals = sess.run(xc, feed_dict={x: xx, c:cc, a:aa})
        self.assertEqual(len(vals), batch)
        self.assertEqual(len(vals[0]), context)
        self.assertEqual(len(vals[0][0]), odim)
        

    def test_defuzzy(self):
        odim = 1
        input_dim = 2
        batch = 1
        sess = K.get_session()

        layer = DefuzzyLayer(odim)
        layer.build(input_shape=(batch,input_dim))
        
        x = K.placeholder(shape=(batch, input_dim))
        rules_outcome = K.placeholder(shape=(input_dim, odim))
        layer.rules_outcome = rules_outcome
        xc = layer.call(x)

        xx = [[0.2, 0.3]]
        cc = [[1],[2]]
        vals = sess.run(xc, feed_dict={x: xx, rules_outcome:cc})
        self.assertEqual(len(vals), batch)
        self.assertEqual(len(vals[0]), odim)
        self.assertAlmostEqual(vals[0][0], 0.8, 7)

    def test_defuzzy2(self):
        odim = 1
        input_dim = 2
        batch = 2
        sess = K.get_session()

        layer = DefuzzyLayer(odim)
        layer.build(input_shape=(batch,input_dim))
        
        x = K.placeholder(shape=(batch, input_dim))
        rules_outcome = K.placeholder(shape=(input_dim, odim))
        layer.rules_outcome = rules_outcome
        xc = layer.call(x)

        xx = [[0.2, 0.3],[0.3, 0.2]]
        cc = [[1],[2]]
        vals = sess.run(xc, feed_dict={x: xx, rules_outcome:cc})
        self.assertEqual(len(vals), batch)
        self.assertEqual(len(vals[0]), odim)
        self.assertAlmostEqual(vals[0][0], 0.8, 7)
        self.assertAlmostEqual(vals[1][0], 0.7, 7)

    def test_defuzzy3(self):
        odim = 3
        input_dim = 2
        batch = 2
        sess = K.get_session()

        layer = DefuzzyLayer(odim)
        layer.build(input_shape=(batch,input_dim))
        
        x = K.placeholder(shape=(batch, input_dim))
        rules_outcome = K.placeholder(shape=(input_dim, odim))
        layer.rules_outcome = rules_outcome
        xc = layer.call(x)

        xx = [[0.2, 0.3],[0.3, 0.2]]
        cc = [[1, 2, 3],[0, 1, 0]]
        vals = sess.run(xc, feed_dict={x: xx, rules_outcome:cc})
        self.assertEqual(len(vals), batch)
        self.assertEqual(len(vals[0]), odim)
        self.assertAlmostEqual(vals[0][0], 0.2, 7)
        self.assertAlmostEqual(vals[0][1], 0.7, 7)
        self.assertAlmostEqual(vals[0][2], 0.6, 7)
        self.assertAlmostEqual(vals[1][0], 0.3, 7)        
        self.assertAlmostEqual(vals[1][1], 0.8, 7)        
        self.assertAlmostEqual(vals[1][2], 0.9, 7)        

if __name__ == '__main__':
    unittest.main()

