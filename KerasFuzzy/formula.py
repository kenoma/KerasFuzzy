from keras import backend as K
import numpy as np

odim = 5
input_dim = 2
batch = 3
x = K.placeholder(shape=(batch, input_dim))
c = K.placeholder(shape=(input_dim, odim))
a = K.placeholder(shape=(input_dim, odim))

aligned_x = K.repeat_elements(K.expand_dims(x, axis = 2), odim, 2)
aligned_c = K.repeat_elements(K.expand_dims(c, 0), batch, 0)
aligned_a = 2 * K.square(K.repeat_elements(K.expand_dims(a, 0), batch, 0))
xc = K.exp(-K.sum(K.square(aligned_x - aligned_c) / aligned_a, axis=1, keepdims=False))

sess = K.get_session()

xx = [[1,1], 
      [0.5,0.5], 
      [0,0]]
cc = [[1, 0.75, 0.5, 0.25, 0],
      [1, 0.75, 0.5, 0.25, 0]]
aa = np.ones(shape=(input_dim, odim))/10
print(sess.run(xc, feed_dict={x: xx, c:cc, a:aa}))
#should be [[1,.....],[... ,...,1,...,...],[...,1]
print(xc)
