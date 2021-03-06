import tensorflow as tf 
import sys 
sys.path.append('..')
import config
from .module import *
from .efficentnet import EfficentNetB0, EfficentNetB1, EfficentNetB2, EfficentNetB3

num_classes = config.NUM_CLASSES

def backbone(inputs,
    is_training=True,
    name='Upconv'):
    """
    complete backbone for detection
    """
    out_effnet = EfficentNetB3(inputs, is_training=is_training)
    # for x in out_effnet.values():
    #     print(x.get_shape())
    block_1 = out_effnet['block_1']
    block_2 = out_effnet['block_2']
    block_3 = out_effnet['block_3']
    block_4 = out_effnet['block_4']
    block_5 = out_effnet['block_5']
    block_6 = out_effnet['block_6']
    block_7 = out_effnet['block_7']
    # print(block_7, block_6, block_5, block_4)

    with tf.variable_scope(name):
        x = UPConv_block(
            block_7,
            skip=block_5,
            kernel_size=3, 
            strides=(2, 2),
            is_training=is_training,
            prefix='upconv_1'
        )
        x = UPConv_block(
            x, 
            skip=block_3,
            kernel_size=3,
            strides=(2, 2),
            is_training=is_training,
            prefix='upconv_2'
        )
        x = UPConv_block(
            x, 
            skip=block_2,
            kernel_size=3,
            strides=(2, 2),
            is_training=is_training,
            prefix='upconv_3'
        )
    return x

def net(inputs,
    is_training=True,
    name='ABC'):
    """
    complete network
    """
    # print(is_training)
    # print('-'*22)
    fea_map = backbone(
        inputs,
        is_training=is_training
    )

    # aff_map = part_aff(
    #     fea_map,
    #     num_repeats=3,
    #     out_dim=8,
    #     is_training=is_training
    # )

    # com_map = tf.concat([fea_map, aff_map], 3)

    heatmap = heat(
        fea_map,
        out_dim=4*num_classes,
        is_training=is_training
    )

    offmap = offset(
        fea_map, 
        out_dim=2,
        is_training=is_training
    )

    aff_map = part_aff(
        fea_map,
        out_dim=8,
        is_training=is_training
    )

    # output = {
    #     'affi': aff_map,
    #     'heatmap': heatmap,
    #     'offset': offmap
    # }
    output = tf.concat([
        heatmap, aff_map, offmap
    ], axis=3)
    #print(output.shape)

    
    return output
