import unittest
from FuzzyLayers import FuzzyLayer
from FuzzyLayers import DefuzzyLayer
from keras import backend as K

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
        aa = [[1/10, 1/10], [1/10, 1/10]]
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
        aa = [[1/2, 1/2], [1/2, 1/2]]
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
        aa = [[1/10, 1/10], [1/10, 1/10]]
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
        aa = [[1/2, 1/4], [1, 1/8]]
        vals = sess.run(xc, feed_dict={x: xx, c:cc, a:aa})
        self.assertEqual(len(vals), batch)
        self.assertEqual(len(vals[0]), odim)
        self.assertAlmostEqual(vals[0][0], 0.7788007831, 7)
        self.assertAlmostEqual(vals[0][1], 0.00002491600973, 7)
        self.assertAlmostEqual(vals[1][0], 0.9394130628, 7)
        self.assertAlmostEqual(vals[1][1], 0.004339483271, 7)

    
    def test_defuzzy_simple(self):
        odim = 2
        input_dim = 2
        batch = 2
        sess = K.get_session()

        layer = DefuzzyLayer(odim)
        layer.build(input_shape=(batch,input_dim))
        
        x = K.placeholder(shape=(batch, input_dim))
        c = K.placeholder(shape=(input_dim, odim))
        a = K.placeholder(shape=(input_dim, odim))
        layer.c = c
        layer.a = a
        xc = layer.call(x)

        xx = [[0.5, 0.8], [0.8, 0.5]]
        cc = [[1, 0.2], [0.8, 0]]
        aa = [[1/2, 1/4], [1, 1/8]]
        vals = sess.run(xc, feed_dict={x: xx, c:cc, a:aa})
        self.assertEqual(len(vals), batch)
        self.assertEqual(len(vals[0]), odim)
        self.assertAlmostEqual(vals[0][0], 0.7788007831, 7)
        self.assertAlmostEqual(vals[0][1], 0.00002491600973, 7)
        self.assertAlmostEqual(vals[1][0], 0.9394130628, 7)
        self.assertAlmostEqual(vals[1][1], 0.004339483271, 7)


if __name__ == '__main__':
    unittest.main()

