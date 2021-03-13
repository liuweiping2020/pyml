import collections

import tensorflow as tf

from modeler.tfmodel import TFModel


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.

    Its parts are:
      scope: The scope of the `Block`.
      unit_fn: The ResNet unit function which takes as input a `Tensor` and
        returns another `Tensor` with the output of the ResNet unit.
      args: A list of length equal to the number of units in the `Block`. The list
        contains one (depth, depth_bottleneck, stride) tuple for each unit in the
        block to serve as argument to unit_fn.
    """

class ResNetModel(TFModel):
    def __init__(self):
        self.batch_size = 32
        self.height, self.width = 224, 224
        self.num_batches= 100
        self.slim = tf.contrib.slim
        pass

    def add_placeholder(self):
        self.inputs = tf.random_uniform((self.batch_size, self.height, self.width, 3))
        pass

    def build(self):
        with self.slim.arg_scope(self.resnet_arg_scope(is_training=False)):
            self.net, self.end_points = self.resnet_v2_152(self.inputs, 1000)
        pass

    def subsample(self,inputs, factor, scope=None):

        if factor == 1:
            return inputs
        else:
            return self.slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

    def conv2d_same(self,inputs, num_outputs, kernel_size, stride, scope=None):
        if stride == 1:
            return self.slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                               padding='SAME', scope=scope)
        else:
            # kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            inputs = tf.pad(inputs,
                            [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
            return self.slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                               padding='VALID', scope=scope)

    @tf.contrib.slim.add_arg_scope
    def stack_blocks_dense(self,net, blocks,
                           outputs_collections=None):
        """Stacks ResNet `Blocks` and controls output feature density.

        First, this function creates scopes for the ResNet in the form of
        'block_name/unit_1', 'block_name/unit_2', etc.


        Args:
          net: A `Tensor` of size [batch, height, width, channels].
          blocks: A list of length equal to the number of ResNet `Blocks`. Each
            element is a ResNet `Block` object describing the units in the `Block`.
          outputs_collections: Collection to add the ResNet block outputs.

        Returns:
          net: Output tensor

        """
        for block in blocks:
            with tf.variable_scope(block.scope, 'block', [net]) as sc:
                for i, unit in enumerate(block.args):
                    with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                        unit_depth, unit_depth_bottleneck, unit_stride = unit
                        net = block.unit_fn(net,
                                            depth=unit_depth,
                                            depth_bottleneck=unit_depth_bottleneck,
                                            stride=unit_stride)
                net = self.slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

        return net

    def resnet_arg_scope(self,is_training=True,
                         weight_decay=0.0001,
                         batch_norm_decay=0.997,
                         batch_norm_epsilon=1e-5,
                         batch_norm_scale=True):
        """Defines the default ResNet arg scope.

        Args:
          is_training: Whether or not we are training the parameters in the batch
            normalization layers of the model.
          weight_decay: The weight decay to use for regularizing the model.
          batch_norm_decay: The moving average decay when estimating layer activation
            statistics in batch normalization.
          batch_norm_epsilon: Small constant to prevent division by zero when
            normalizing activations by their variance in batch normalization.
          batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
            activations in the batch normalization layer.

        Returns:
          An `arg_scope` to use for the resnet models.
        """
        batch_norm_params = {
            'is_training': is_training,
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
            'scale': batch_norm_scale,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
        }

        with self.slim.arg_scope(
                [self.slim.conv2d],
                weights_regularizer=self.slim.l2_regularizer(weight_decay),
                weights_initializer=self.slim.variance_scaling_initializer(),
                activation_fn=tf.nn.relu,
                normalizer_fn=self.slim.batch_norm,
                normalizer_params=batch_norm_params):
            with self.slim.arg_scope([self.slim.batch_norm], **batch_norm_params):
                with self.slim.arg_scope([self.slim.max_pool2d], padding='SAME') as arg_sc:
                    return arg_sc

    @tf.contrib.slim.add_arg_scope
    def bottleneck(self,inputs, depth, depth_bottleneck, stride,
                   outputs_collections=None, scope=None):
        """Bottleneck residual unit variant with BN before convolutions.

        This is the full preactivation residual unit variant proposed in [2]. See
        Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
        variant which has an extra bottleneck layer.

        When putting together two consecutive ResNet blocks that use this unit, one
        should use stride = 2 in the last unit of the first block.

        Args:
          inputs: A tensor of size [batch, height, width, channels].
          depth: The depth of the ResNet unit output.
          depth_bottleneck: The depth of the bottleneck layers.
          stride: The ResNet unit's stride. Determines the amount of downsampling of
            the units output compared to its input.
          rate: An integer, rate for atrous convolution.
          outputs_collections: Collection to add the ResNet unit output.
          scope: Optional variable_scope.

        Returns:
          The ResNet unit's output.
        """
        with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
            depth_in = self.slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
            preact = self.slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
            if depth == depth_in:
                shortcut = self.subsample(inputs, stride, 'shortcut')
            else:
                shortcut = self.slim.conv2d(preact, depth, [1, 1], stride=stride,
                                       normalizer_fn=None, activation_fn=None,
                                       scope='shortcut')

            residual = self.slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                                   scope='conv1')
            residual = self.conv2d_same(residual, depth_bottleneck, 3, stride,
                                   scope='conv2')
            residual = self.slim.conv2d(residual, depth, [1, 1], stride=1,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='conv3')

            output = shortcut + residual

            return self.slim.utils.collect_named_outputs(outputs_collections,
                                                    sc.name,
                                                    output)

    def resnet_v2(self,inputs,
                  blocks,
                  num_classes=None,
                  global_pool=True,
                  include_root_block=True,
                  reuse=None,
                  scope=None):
        with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with self.slim.arg_scope([self.slim.conv2d, self.bottleneck,
                                 self.stack_blocks_dense],
                                outputs_collections=end_points_collection):
                net = inputs
                if include_root_block:
                    with self.slim.arg_scope([self.slim.conv2d],
                                        activation_fn=None, normalizer_fn=None):
                        net = self.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                    net = self.slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                net = self.stack_blocks_dense(net, blocks)
                net = self.slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                if num_classes is not None:
                    net = self.slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                      normalizer_fn=None, scope='logits')
                # Convert end_points_collection into a dictionary of end_points.
                end_points = self.slim.utils.convert_collection_to_dict(end_points_collection)
                if num_classes is not None:
                    end_points['predictions'] = self.slim.softmax(net, scope='predictions')
                return net, end_points

    def resnet_v2_50(self,inputs,
                     num_classes=None,
                     global_pool=True,
                     reuse=None,
                     scope='resnet_v2_50'):
        """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
        blocks = [
            Block('block1', self.bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
            Block(
                'block2', self.bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
            Block(
                'block3', self.bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
            Block(
                'block4', self.bottleneck, [(2048, 512, 1)] * 3)]
        return self.resnet_v2(inputs, blocks, num_classes, global_pool,
                         include_root_block=True, reuse=reuse, scope=scope)

    def resnet_v2_101(self,inputs,
                      num_classes=None,
                      global_pool=True,
                      reuse=None,
                      scope='resnet_v2_101'):
        """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
        blocks = [
            Block(
                'block1', self.bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
            Block(
                'block2', self.bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
            Block(
                'block3', self.bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
            Block(
                'block4', self.bottleneck, [(2048, 512, 1)] * 3)]
        return self.resnet_v2(inputs, blocks, num_classes, global_pool,
                         include_root_block=True, reuse=reuse, scope=scope)

    def resnet_v2_152(self,inputs,
                      num_classes=None,
                      global_pool=True,
                      reuse=None,
                      scope='resnet_v2_152'):
        """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
        blocks = [
            Block(
                'block1', self.bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
            Block(
                'block2', self.bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
            Block(
                'block3', self.bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
            Block(
                'block4', self.bottleneck, [(2048, 512, 1)] * 3)]
        return self.resnet_v2(inputs, blocks, num_classes, global_pool,
                         include_root_block=True, reuse=reuse, scope=scope)

    def resnet_v2_200(self,inputs,
                      num_classes=None,
                      global_pool=True,
                      reuse=None,
                      scope='resnet_v2_200'):
        """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
        blocks = [
            Block(
                'block1', self.bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
            Block(
                'block2', self.bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
            Block(
                'block3', self.bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
            Block(
                'block4', self.bottleneck, [(2048, 512, 1)] * 3)]
        return self.resnet_v2(inputs, blocks, num_classes, global_pool,
                         include_root_block=True, reuse=reuse, scope=scope)
