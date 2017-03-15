import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers as tf_layers
from dense_net import DenseBlock, TrainsitionLayer, bottleneck_block
import random_hole
import tiny_imagenet



# Based on DenseNets but with an extra trick with dilated convolutions to increase receptive fields
# Basically, intelligently converts one texture to another and you can train it anything, image segmentation, super resolution,
# neural style, image inpainting, neural doodle, etc. The only drawback is that for every one you will need to train the model from scratch...
# I will try to make things more generic by adding additional input to the Magic network - so that the transformation is a
# function of an image and a goal.


class DownsampleBlockSpecs:
    def __init__(self, layers, growth_factors, dilate_after=None, bottleneck=0.25, keep_dim_fraction=0.5):
        ''' layers specifies how many layers given dense block should have
            growth_factors specifies growth_factor for each layer (if int constant for all)
            dilate_after is a list of layers after which dilation should be increased by 2, if None then no dilation'''
        self.layers = layers
        self.growth_factors = growth_factors if type(growth_factors)!=int else self.layers*[growth_factors]
        self.dilations = [1]*layers
        if dilate_after is not None:
            for k in dilate_after:
                while k<len(self.dilations):
                    self.dilations[k] *= 2
                    k += 1

        self.bottleneck = bottleneck
        self.keep_dim_fraction = keep_dim_fraction


class UpsableBlockSpecs:
    def __init__(self, kernel_size, channels, passthrough, passthrough_relative_size, follow_up_residual_block, activation=tf.nn.elu):
        self.kernel_size = kernel_size
        self.channels = channels
        self.passthrough = passthrough
        if passthrough:
            assert follow_up_residual_block, 'You must follow up with residual blocks if you use passthrough so that you get specified number of channels'
        self.passthrough_relative_size = passthrough_relative_size
        self.follow_up_residual_block = follow_up_residual_block
        self.activation = activation


class Specs:
    def __init__(self, downsample_blocks, upsample_blocks):
        self.downsample_blocks = downsample_blocks
        self.upsample_blocks = upsample_blocks
        print 'According to specs the resolution of the output will be x%f' % 2**(len(upsample_blocks)-len(downsample_blocks)+1)


class Magic:
    def __init__(self, specs, trainable=True, weights_collections=None):
        assert isinstance(specs, Specs)
        self.specs = specs
        self.trainable = trainable
        self.batch_norm_params = {'updates_collections': None, 'is_training': trainable, 'trainable': trainable, 'scale': False}
        assert not isinstance(weights_collections, basestring), 'Must be a list of collections!'
        self.variable_collections = None if weights_collections is None else {'weights': weights_collections}
        self.weight_collections = weights_collections
        self.d_res_maps = None
        self.u_res_maps = None
        self.own_scope_name = None

    def __call__(self, images):
        ''' transforms images '''
        resolution = 1.
        out = images
        self.d_res_maps = {}
        self.u_res_maps = {}
        with tf.variable_scope(None, default_name='MagicNet'):
            with tf.variable_scope('downsampler'):
                for dblock in self.specs.downsample_blocks:
                    out = DenseBlock(growth_rate=dblock.growth_factors, layers=dblock.layers,
                                     bottleneck=dblock.bottleneck, trainable=self.trainable,
                                     weights_collections=self.weight_collections, dilation_factors=dblock.dilations)(out)
                    out, res_map = TrainsitionLayer(keep_dim_fraction=dblock.keep_dim_fraction, trainable=self.trainable,
                                           weights_collections=self.weight_collections)(out)
                    self.d_res_maps[resolution] = res_map
                    resolution /= 2.

                resolution *= 2.
                out = self.d_res_maps[resolution]

            with tf.variable_scope('upsampler'):
                for ublock in self.specs.upsample_blocks:
                    # first standard deconv
                    out = tf_layers.conv2d_transpose(out, ublock.channels, ublock.kernel_size, stride=2,
                                                     activation_fn=ublock.activation,
                                                     normalizer_fn=tf_layers.batch_norm,
                                                     normalizer_params=self.batch_norm_params,
                                                     variables_collections=self.variable_collections,
                                                     trainable=self.trainable)

                    resolution *= 2.

                    if ublock.passthrough:
                        assert ublock.follow_up_residual_block
                        take_from = self.d_res_maps[resolution]

                        # the question is: should we add the passthrough or concat as extra channels?
                        # if concat then use batch_norm + activation, otherwise not but has to have the same num of channels
                        # will use concat for now
                        if ublock.passthrough_relative_size != 1:
                            ext = tf_layers.conv2d(take_from, int(take_from.get_shape().as_list()[-1] * ublock.passthrough_relative_size), 1,
                                                   stride=1,
                                                   activation_fn=ublock.activation,
                                                   normalizer_fn=tf_layers.batch_norm,
                                                   normalizer_params=self.batch_norm_params,
                                                   variables_collections=self.variable_collections,
                                                   trainable=self.trainable)
                        else:
                            ext = take_from

                        out = tf.concat((out, ext), 3)

                    if ublock.follow_up_residual_block:
                        if not isinstance(ublock.follow_up_residual_block, int):
                            blocks = 1
                        else:
                            blocks = ublock.follow_up_residual_block
                        for _ in xrange(blocks):
                            out = bottleneck_block(out, ublock.channels, stride=1, training=self.trainable, weights_collections=self.weight_collections, scale=False, activation=ublock.activation)
                    self.u_res_maps[resolution] = out
            scope = tf.get_variable_scope()
            self.own_scope_name = scope.name
        return out

    def get_own_variables(self):
        return tf.get_collection(tf.GraphKeys().GLOBAL_VARIABLES, scope=self.own_scope_name)

    def get_own_weights(self):
        assert self.weight_collections
        return tf.get_collection(self.weight_collections[-1])

    def get_num_params(self):
        s = 0
        for e in self.get_own_weights():
            s += np.prod(e.get_shape().as_list())
        return s

    def get_own_l2_loss(self):
        print 'Number of params in weights', self.get_num_params()
        return sum(map(tf.nn.l2_loss, self.get_own_weights()), tf.constant(0.))



def output_channel_ranges_from_mean_std(mean, std):
    new_mean = (255./2 - mean)/std
    new_range = 255. / std
    return np.concatenate((np.expand_dims(new_mean - new_range/2., 1), np.expand_dims(new_mean + new_range/2. , 1)), 1)

def to_image_channels(inp, num_channels, output_channel_ranges, trainable=True, nonlinearity=tf.nn.tanh, nonlinearity_range=(-1, 1)):
    # for each channel you must supply a range (min_val, max_val) as array CHANS X 2
    assert len(output_channel_ranges) == num_channels
    print output_channel_ranges
    with tf.variable_scope(None, default_name='ToImageChannels'):
        out = tf_layers.conv2d(inp, num_channels, 1,
                               stride=1,
                               activation_fn=nonlinearity,
                               trainable=trainable)
        nonlinearity_mean = sum(nonlinearity_range) / 2.
        nonlinearity_spread = float(nonlinearity_range[1]) - nonlinearity_range[0]
        output_channel_ranges = np.array(output_channel_ranges)
        output_channel_means = np.mean(output_channel_ranges, 1)
        output_channel_spreads = output_channel_ranges[:, 1] - output_channel_ranges[:, 0]
        return (out - nonlinearity_mean) / nonlinearity_spread * output_channel_spreads + output_channel_means



def to_classification_layer(inp, num_classes, trainable=True, weights_collections=None):
    variables_collections = None if weights_collections is None else {'weights': weights_collections}
    with tf.variable_scope(None, default_name='ClassificationLayer'):
        out = tf.reduce_mean(inp, (1,2))
        out = tf_layers.fully_connected(out, num_classes,
                               activation_fn=None,
                               variables_collections=variables_collections,
                               trainable=trainable)
    return out



StandardDownsample = [DownsampleBlockSpecs(7, 16, [3, 5]),
                      DownsampleBlockSpecs(11, 22, [5, 7]),
                      DownsampleBlockSpecs(14, 22, [7]),
                      DownsampleBlockSpecs(16, 22, None)]

DiscDownsample = [DownsampleBlockSpecs(6, 16, [3, 5]),
                  DownsampleBlockSpecs(8, 22, [4, 6]),
                  DownsampleBlockSpecs(10, 22, [6]),]

StandardUpsample = [UpsableBlockSpecs(2, 256, True, 1, 3),
                    UpsableBlockSpecs(2, 128, True, 1, 3),
                    UpsableBlockSpecs(2, 64, True, 1, 3)
                    ]


def get_spatial_feature_weights(mask, masked_weight):
    temp = masked_weight*(1.-mask) + mask
    return temp / tf.reduce_mean(temp)




BS = 4
MASKED_FEATURES_WEIGHT = 11.
IMG_SIZE = 80
import imagenet


StandardMagic = Specs(StandardDownsample, StandardUpsample)


def get_vars(scope):
    return tf.get_collection(tf.GraphKeys().GLOBAL_VARIABLES, scope=scope)


masks = tf.ones((BS, IMG_SIZE, IMG_SIZE, 1))* tf.expand_dims(random_hole.matrix_select(tf.ones((IMG_SIZE, IMG_SIZE, 1)), 8, 72, 8, 72), 0)
#masks = 1. - tf.expand_dims(random_hole.random_matrices_gen(BS, 20, 30, (IMG_SIZE, IMG_SIZE)), 3) # tf.ones((BS, 64, 64, 1), tf.float32)  # 0s or 1s, randomly generated every run!


masks1 = masks
masks2 = tf.image.resize_bilinear(masks, (IMG_SIZE/2, IMG_SIZE/2))
masks4 = tf.image.resize_bilinear(masks, (IMG_SIZE/4, IMG_SIZE/4))
masks8 = tf.image.resize_bilinear(masks, (IMG_SIZE/8, IMG_SIZE/8))


images = tf.placeholder(tf.float32, (BS, IMG_SIZE, IMG_SIZE, 3))
labels = tf.placeholder(tf.int32, (BS,))

masked_images = masks1*images

with tf.variable_scope('EncDec'):
    a = Magic(StandardMagic, weights_collections=['abc'], trainable=False)
    fake_imgs = to_image_channels(a(masked_images), 3,
                                output_channel_ranges=output_channel_ranges_from_mean_std(tiny_imagenet.IMAGE_NET_PIXEL_MEAN,
                                                                                          tiny_imagenet.IMAGE_NET_PIXEL_STD))
    raw_scores = to_classification_layer(a.d_res_maps[1/8.], 200, weights_collections=['abc'])
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=raw_scores, labels=labels))



with tf.variable_scope('Disc'):
    disc = Magic(Specs(DiscDownsample, []), weights_collections=['yub'], trainable=False)
    disc(tf.concat((images, fake_imgs), 0))
    disc_probs = tf_layers.conv2d(disc.d_res_maps[1 / 4.], 1, 1, activation_fn=tf.nn.sigmoid)


real_features1, fake_features1 = tf.split(disc.d_res_maps[1.], 2)
real_features2, fake_features2 = tf.split(disc.d_res_maps[1./2], 2)
real_features4, fake_features4 = tf.split(disc.d_res_maps[1./4], 2)


distance_loss = (tf.reduce_mean(get_spatial_feature_weights(masks1, MASKED_FEATURES_WEIGHT)*((images - fake_imgs) ** 2)) #+
                # tf.reduce_mean(get_spatial_feature_weights(masks2, MASKED_FEATURES_WEIGHT)*((real_features2 - fake_features2) ** 2)) +
                # tf.reduce_mean(get_spatial_feature_weights(masks4, MASKED_FEATURES_WEIGHT)*((real_features4 - fake_features4) ** 2))
                ) / 2.


real_probs_map, fake_probs_map = tf.split(disc_probs, 2)
# we have to use masks4, it may seem high res but actually they cover about 32x32 image patches thanks to dilated convolutions
i_masks4 = 1. - masks4
i_masks4_areas = tf.reduce_sum(i_masks4, (1,2,3))
real_probs, fake_probs = tf.reduce_sum(real_probs_map*i_masks4, (1,2,3)) / i_masks4_areas, \
                         tf.reduce_sum(fake_probs_map*i_masks4, (1,2,3)) / i_masks4_areas


trick_loss = -tf.reduce_mean(tf.log(1.-fake_probs))
disc_loss = (-tf.reduce_mean(tf.log(fake_probs)) - tf.reduce_mean(tf.log(1-real_probs))) / 2.
full_disc_loss = disc_loss + 0.0005*disc.get_own_l2_loss()



print len(a.get_own_variables())


full_loss =  0.0005*a.get_own_l2_loss() + 0.5*distance_loss + 0.1*trick_loss

import time, cv2
LAST = time.time()
def tick(extra_vars, batch):
    global LAST
    if time.time() - LAST < 10:
        return
    LAST = time.time()
    cv2.imwrite('xyz.jpg', tiny_imagenet.to_bgr_img(np.concatenate((extra_vars['fake_imgs'][0], extra_vars['masked_images'][0]), 0)))



print len(tf.global_variables())

print len(tf.trainable_variables())
print len(a.get_own_weights())
print a.get_num_params()

train_main = tf.train.MomentumOptimizer(0.01, 0.9, use_nesterov=True).minimize(full_loss, var_list=get_vars('EncDec'))

disc_train_every = 15
maybe_train_disc = tf.cond(tf.random_uniform((), 0., 1.) < 1./disc_train_every,
                           lambda : tf.train.MomentumOptimizer(0.05, 0.9, use_nesterov=True).minimize(full_disc_loss, var_list=get_vars('Disc')),
                           lambda : tf.no_op())

train_op = tf.group(train_main, maybe_train_disc)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    import tfutils
    train_bm = tiny_imagenet.get_train_bm(BS)
    val_bm = tiny_imagenet.get_train_bm(BS)
    saver = tf.train.Saver(tf.global_variables())

    nt = tfutils.NiceTrainer(sess, train_bm, [images, labels], train_op, bm_val=val_bm,
                             extra_variables={#'loss': loss,
                                              #'probs': tf.nn.softmax(raw_scores),
                                              'trick_loss': trick_loss,
                                              'disc_loss': disc_loss,
                                              'distance_loss': distance_loss,
                                              'fake_imgs': fake_imgs,
                                              'masked_images': masked_images,
                                               },
                             printable_vars=['distance_loss', 'disc_loss', 'trick_loss'],
                             computed_variables={#'acc': tfutils.accuracy_calc_op(),
                                                 'tick': tick},
                             saver=saver,
                             save_every=5000000,
                             save_dir='chuj',
                             smooth_coef=0.9)

    nt.restore(relaxed=True)
    # reinit = tf.get_collection(tf.GraphKeys().GLOBAL_VARIABLES, scope='Disc')
    # assert reinit
    # sess.run(tf.variables_initializer(reinit))


    while True:
        nt.train()
        nt.validate()
        nt.save()


