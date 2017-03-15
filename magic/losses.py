import tensorflow as tf
import numpy as np
from meta_restorer import restore_in_scope



class LayerExtractor:
    # todo extend to support adversarial loss family - ie model_func not fixed
    def __init__(self, model_func, model_ckpt, features, extractor_scope_name='contentlossextractor', reuse=None):
        # special layer name = #SELF#
        self.model_func = model_func
        self.model_ckpt = model_ckpt
        self.features = features
        self.extractor_scope_name = extractor_scope_name
        self.reuse = reuse
        self._to_init = []
        self.extracted_features = None
        self.__called = False
        self.sess = None
        self._losses = {}
        self.calculated_losses = {}

    def init_all(self, sess):
        self.sess = sess
        sess.run(*tuple(self._to_init))
        for name, loss in self._losses.items():
            if hasattr(loss, 'init'):
                loss.init(self)
            partial_losses = []
            for layer in loss._layers:
                genuine, fake = self.extracted_features[layer]
                partial_losses.append(loss._calc_loss(genuine, fake, layer))
                assert len(partial_losses[-1].get_shape().as_list()) == 0
            self.calculated_losses[name] = tf.reduce_mean(tf.stack(partial_losses))

    def __call__(self, genuine, fake):
        assert not self.__called
        self._genuine_inp, self._fake_inp = genuine, fake
        assert genuine.get_shape().as_list() == fake.get_shape().as_list()
        self._inp = tf.concat((genuine, fake), 0)
        with tf.variable_scope(self.extractor_scope_name, reuse=self.reuse):
            self.model_func(self._inp, trainable=False, reuse=self.reuse)
        self._to_init.append(restore_in_scope(self.model_ckpt, self.extractor_scope_name))

        # now get all the required tensors from extractor
        self.extracted_features = {}  # layer_name: (genuine_tensor, fake_tensor)
        for feature_name, tensor_name in self.features.items():
            if tensor_name == '#SELF#':
                self.extracted_features[feature_name] = (genuine, fake)
            else:
                cand = tf.get_default_graph().get_tensor_by_name(self.extractor_scope_name + '/' + tensor_name)
                self.extracted_features[feature_name] = tf.split(cand, 2)
        self.__called = True

    def add_loss(self, layers, loss_class, loss_name=None):
        assert self.__called
        assert isinstance(loss_class, LossClass), 'loss_class must be an instance of LossClass'
        if loss_name is None:
            loss_name = loss_class.__class__.__name__
        loss_class._layers = layers
        self._losses[loss_name] = loss_class



class LossClass:
    _layers = None
    def _calc_loss(self, genuine, fake, feature_name):  # takes tensors, returns float32 tensor ()
        raise NotImplementedError()



class LNLoss(LossClass):
    def __init__(self, n, epsilon=0.):
        self.n = n
        assert epsilon >= 0.
        self.epsilon = epsilon
        assert not self.n % 1 or epsilon, 'Epsilon must be provided when using non integer n'

    def _calc_loss(self, genuine, fake, feature_name):
        if self.n % 1 != 0:
            return tf.reduce_mean((tf.abs(genuine - fake) + self.epsilon)**self.n)
        else:
            if self.n==1:
                return tf.reduce_mean(tf.abs(genuine - fake))
            elif  self.n % 2 == 0:
                return tf.reduce_mean((genuine - fake)**self.n)
            else:
                return tf.reduce_mean(tf.abs(genuine - fake)**self.n)


class TVLoss(LossClass):
    def __init__(self, border_penalty=False):
        self.border_penalty = border_penalty


    def _calc_loss(self, genuine, fake, feature_name):
        # again we care only about fake
        if len(fake.get_shape().as_list())==3:
            fake = tf.expand_dims(fake, 3)
        n, x, y, c = fake.get_shape().as_list()

        if self.border_penalty:
            mx1 = tf.concat((fake, tf.zeros((n, 1, y, c))), 1)
            mx2 = tf.concat((tf.zeros((n, 1, y, c)), fake), 1)
            x_loss = tf.reduce_mean((mx1 - mx2) ** 2)
            my1 = tf.concat((fake, tf.zeros((n, x, 1, c))), 2)
            my2 = tf.concat((tf.zeros((n, x, 1, c)), fake), 2)
            y_loss = tf.reduce_mean((my1 - my2) ** 2)
        else:
            x_loss = tf.reduce_mean((fake[:, 1:, :, :] - fake[:, :-1, :, :]) ** 2)
            y_loss = tf.reduce_mean((fake[:, :, 1:, :] - fake[:, :, :-1, :]) ** 2)
        return (x_loss + y_loss) / 2.



class StyleLoss(LossClass):
    def __init__(self, normalised_image_np_array):
        assert len(normalised_image_np_array.shape) == 3
        self.normalised_image_np_array = normalised_image_np_array

    def init(self, feature_extractor):
        t = feature_extractor.extracted_features.items()
        keys, to_collect = list(e[0] for e in t), list(e[1] for e in t)
        sh = feature_extractor._inp.get_shape().as_list()
        temp_input = np.zeros(sh, np.float32) + np.expand_dims(self.normalised_image_np_array, 0)
        image_features = dict(zip(keys, feature_extractor.sess.run(to_collect, {feature_extractor._inp: temp_input})))
        self.style_features = {}
        for l, f in image_features.items():
            f = f[0][0]
            sh = f.shape
            f = np.reshape(f, (sh[0]*sh[1], sh[2]))
            self.style_features[l] = tf.constant(np.expand_dims(np.matmul(f.T, f) / (sh[0]*sh[1]), 0), tf.float32)

    def _calc_loss(self, genuine, fake, feature_name):
        # we just look at the fake one
        target = self.style_features[feature_name]
        # we just care about fake
        sh = fake.get_shape().as_list()
        target_shape = sh[0], sh[1]*sh[2], sh[3]
        temp = tf.reshape(fake, target_shape)
        styles = tf.matmul(tf.transpose(temp, (0, 2, 1)), temp) / (sh[1]*sh[2])

        return tf.reduce_mean((styles - target)**2)



STD_FEATURES_RES_NET = {
    'pixels': '#SELF#',
    'l1': 'scale2/block1/Relu:0',
    'l2': 'scale2/block3/Relu:0',
    'l3': 'scale3/block4/Relu:0',
    'l4': 'scale4/block6/Relu:0',
}
import imagenet
import cv2
import resnet


opt = tf.get_variable('opt', (1, 224, 224, 3), dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
target = tf.constant(np.expand_dims(imagenet.load_img('ox.jpg'), 0), tf.float32)



go = imagenet.load_img('gogh.jpg')
le = LayerExtractor(resnet.inference, resnet.CKPT, STD_FEATURES_RES_NET)
le(target, opt)
le.add_loss(['l1','l2', 'l3', 'l4'], LNLoss(2))
le.add_loss([ 'l2'], StyleLoss(go))
le.add_loss(['pixels'], TVLoss())

sess = tf.Session()
sess.run(tf.global_variables_initializer())
le.init_all(sess)

loss = le.calculated_losses['LNLoss'] + 30*le.calculated_losses['TVLoss'] + 5*le.calculated_losses['StyleLoss']
print loss
train_op = tf.train.AdamOptimizer(0.1).minimize(loss)

sess.run(tf.global_variables_initializer())
sess.run(le._to_init)


for e in xrange(1300):
    _, l, o = sess.run([train_op, loss, opt])
    print l
    if not e%300:
       # cv2.imshow('test', imagenet.to_bgr_img(o[0]))
        cv2.imwrite('test.jpg', imagenet.to_bgr_img(o[0]))
      #  cv2.waitKey(100000)

