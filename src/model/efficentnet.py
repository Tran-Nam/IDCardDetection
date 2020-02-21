import tensorflow as tf 
import numpy as np 
import collections 
from .module import *

BlockArgs = collections.namedtuple(
    'BlockArgs', [
        'kernel_size',
        'num_repeat',
        'input_filters',
        'output_filters',
        'strides',
        'expand_ratio'
    ]
)

DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16, strides=[1, 1], expand_ratio=1),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24, strides=[2, 2], expand_ratio=6),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40, strides=[2, 2], expand_ratio=6),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80, strides=[2, 2], expand_ratio=6),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112, strides=[1, 1], expand_ratio=6),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192, strides=[2, 2], expand_ratio=6),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320, strides=[1, 1], expand_ratio=6)
]

def round_filters(filters, width_coef, depth_divisor):
    """
    round num filters based on width multiplier
    """
    filters *= width_coef
    new_filters = int(filters + depth_divisor/2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    if new_filters < 0.9*filters:
        new_filters += depth_divisor
    return int(new_filters)

def round_repeats(repeats, depth_coef):
    """
    round num repeats each block based on depth coef
    """
    return int(np.ceil(repeats*depth_coef))

def MBConv_block(inputs,
    block_args,
    is_training=True,
    prefix=''):
    """
    Mobile Inverted Residual Bottleneck
    """
    filters = block_args.input_filters * block_args.expand_ratio
    # base = tf.layers.conv2d(
    #     inputs, 
    #     filters=block_args.output_filters, 
    #     kernel_size=block_args.kernel_size, 
    #     strides=block_args.strides, 
    #     padding='same')#@@@@
    with tf.variable_scope(prefix+'MBConv_block'):        
        if block_args.expand_ratio != 1:
            x = conv_block(
                inputs,
                filters,
                kernel_size=1,
                use_batchnorm=True,
                use_relu=True,
                is_training=is_training,
                prefix=prefix+'expand_conv'
            )
        else:
            x = inputs 
        
        # print(x.get_shape())
        x = DWConv_block(
            x,
            filters,
            kernel_size=block_args.kernel_size,
            use_batchnorm=True,
            use_relu=True,
            is_training=is_training
        )
        # print(x.get_shape())

        x = conv_block(
            x,
            output_dim=block_args.input_filters,
            kernel_size=1,
            use_batchnorm=True,
            use_relu=False,
            is_training=is_training,
            prefix=prefix+'invert_expand_conv'
        )
        # print(x.get_shape)
        # print(inputs.get_shape)
        # print('='*50)
        x = tf.add(inputs, x)

        x = conv_block(
            x,
            output_dim=block_args.output_filters,
            kernel_size=block_args.kernel_size,
            strides=block_args.strides,
            use_batchnorm=True, 
            use_relu=True,
            is_training=is_training
        )
        # print(base.get_shape(), x.get_shape())
        # print('-'*50)
        # x = tf.add(base, x)
        # x = tf.nn.relu(x)
        return x

def EfficentNet(inputs,
    width_coef,
    depth_coef,
    depth_divisor=8,
    blocks_args=DEFAULT_BLOCKS_ARGS,
    is_training=True,
    model_name='EfficentNet'):
    """
    EfficentNet architext respond to compound parameter
    """
    # first block
    
    output = {}

    with tf.variable_scope(model_name):
        x = conv_block(
            inputs, 
            round_filters(32, width_coef, depth_divisor),
            kernel_size=3,
            strides=(2, 2),
            use_batchnorm=True,
            use_relu=True,
            is_training=is_training
        )
        # print(x.get_shape)
        # print('='*50)

        num_blocks_total = sum(block_args.num_repeat for block_args in blocks_args)
        # print(num_blocks_total)
        block_num = 0
        for idx, block_args in enumerate(blocks_args):
            assert block_args.num_repeat > 0
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, width_coef, depth_divisor),
                output_filters=round_filters(block_args.output_filters, width_coef, depth_divisor),
                num_repeat=round_repeats(block_args.num_repeat, depth_coef)
            )
            # print(block_args.num_repeat)
            with tf.variable_scope('Block_{}'.format(idx+1)):
                x = MBConv_block(
                    x,
                    block_args,
                    is_training=is_training,
                    prefix='block_{}0_'.format(idx+1)
                )
                # print(x.get_shape)

                block_num += 1
                if block_args.num_repeat > 1:
                    block_args = block_args._replace(
                        input_filters=block_args.output_filters, 
                        strides=[1, 1]
                    )

                    for bidx in range(block_args.num_repeat - 1):
                        block_prefix = 'block{}{}_'.format(
                            idx+1,
                            bidx+1
                        )

                        x = MBConv_block(
                            x,
                            block_args,
                            is_training=is_training,
                            prefix=block_prefix
                        )

                        block_num += 1
                    # print(x.get_shape)
            output['block_{}'.format(idx+1)] = x
        
    return output

def EfficentNetB0(inputs,
    is_training=True):
    return EfficentNet(
        inputs,
        width_coef=1.0,
        depth_coef=1.0,
        is_training=is_training,
        model_name='EfficentNetB0'
    )

def EfficentNetB1(inputs,
    is_training=True):
    return EfficentNet(
        inputs,
        width_coef=1.0,
        depth_coef=1.1,
        is_training=is_training,
        model_name='EfficentNetB1'
    )

def EfficentNetB2(inputs,
    is_training=True):
    return EfficentNet(
        inputs,
        width_coef=1.1,
        depth_coef=1.2,
        is_training=is_training,
        model_name='EfficentNetB2'
    )

def EfficentNetB3(inputs,
    is_training=True):
    return EfficentNet(
        inputs, 
        width_coef=1.2,
        depth_coef=1.4,
        is_training=is_training,
        model_name='EfficentNetB3'
    )

def EfficentNetB4(inputs,
    is_training=True):
    return EfficentNet(
        inputs,
        width_coef=1.4,
        depth_coef=1.8,
        is_training=is_training,
        model_name='EfficentNetB4'
    )

def EfficentNetB5(inputs,
    is_training=True):
    return EfficentNet(
        inputs, 
        width_coef=1.6, 
        depth_coef=2.2,
        is_training=is_training,
        model_name='EfficentNetB5'
    )

def EfficentNetB6(inputs,
    is_training=True):
    return EfficentNet(
        inputs, 
        width_coef=1.8,
        depth_coef=2.6,
        is_training=is_training,
        model_name='EfficentNetB6'
    )

def EfficentNetB7(inputs,
    is_training=True):
    return EfficentNet(
        inputs, 
        width_coef=2.0,
        depth_coef=3.1,
        is_training=is_training,
        model_name='EffientNetB7'
    )