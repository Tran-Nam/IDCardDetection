import tensorflow as tf 

def conv_block(inputs, 
    output_dim, 
    kernel_size=3, 
    strides=(1, 1),
    use_batchnorm=True, 
    use_relu=True, 
    is_training=True, 
    prefix=''):
    """
    convolution block with 
        con2d
        batchnorm - option
        activation: relu - option
    """
    with tf.variable_scope(prefix+'conv_block'):
        x = tf.layers.conv2d(inputs, output_dim, kernel_size, strides, padding='same')
        if use_batchnorm:
            x = tf.contrib.layers.batch_norm(x, is_training=is_training)
        if use_relu:
            x = tf.nn.relu(x)
        return x 

def DWConv_block(inputs,
    output_dim, 
    kernel_size=3,
    use_batchnorm=True, 
    use_relu=True,
    is_training=True,
    prefix=''):
    """
    depthwise convolution block
    """
    with tf.variable_scope(prefix+'DWConv_block'):
        x = tf.contrib.layers.separable_conv2d(
            inputs,
            output_dim,
            kernel_size,
            depth_multiplier=1,
            activation_fn=None
        )
        if use_batchnorm:
            x = tf.contrib.layers.batch_norm(x, is_training=is_training)
        if use_relu:
            x = tf.nn.relu(x)
        return x

def UPConv_block(inputs,
    skip,
    kernel_size=3,
    strides=(2, 2),
    use_batchnorm=True, 
    use_relu=True,
    is_training=True,
    prefix=''):
    """
    Up convolution block
    similar architect UNet
    """
    output_dim = skip.get_shape().as_list()[3]
    with tf.variable_scope(prefix+'UPConv_block'):
        x = tf.layers.conv2d_transpose(
            inputs,
            output_dim,
            kernel_size=kernel_size,
            strides=strides,
            padding='same'
        )
        if use_batchnorm:
            x = tf.contrib.layers.batch_norm(x, is_training=is_training)
        x = tf.add(skip, x)
        if use_relu:
            x = tf.nn.relu(x)
        return x

def heat(inputs,
    out_dim=4, 
    is_training=True, 
    prefix=''): # tl, tr, br, bl
    """
    module detect 4 corner of idcard
    """
    in_dim = inputs.get_shape().as_list()[3]
    with tf.variable_scope(prefix+'heatmap'):
        x = conv_block(
            inputs,
            in_dim, 
            kernel_size=3, 
            use_batchnorm=False, 
            use_relu=True, 
            is_training=is_training
        )
        x = conv_block(
            x, 
            out_dim, 
            kernel_size=1, 
            use_batchnorm=False,
            use_relu=False,
            is_training=is_training,
            prefix='last_conv'
        )
        return x      
    
def offset(inputs, 
    out_dim=2, 
    is_training=True,
    prefix=''): # offset
    """
    module detect offset of point in heatmap
    """
    in_dim = inputs.get_shape().as_list()[3]
    with tf.variable_scope(prefix+'offset'):
        x = conv_block(
            inputs,
            in_dim, 
            kernel_size=3, 
            use_batchnorm=False, 
            use_relu=True, 
            is_training=is_training
        )
        x = conv_block(
            x, 
            out_dim, 
            kernel_size=1, 
            use_batchnorm=False,
            use_relu=False,
            is_training=is_training,
            prefix='last_conv'
        )
        return x 

def part_aff(inputs,
    num_repeats=3,
    out_dim=8,
    is_training=True,
    prefix=''):
    """
    module detect part affinity field
    """
    in_dim = inputs.get_shape().as_list()[3]
    with tf.variable_scope(prefix+'aff'):
        x = conv_block(
            inputs,
            in_dim,
            kernel_size=3,
            strides=(1, 1),
            use_batchnorm=True,
            use_relu=True,
            is_training=is_training,
            prefix='block_0'
        )
        if num_repeats > 1:
            for bidx in range(1, num_repeats):
                x = tf.concat([inputs, x], 3)
                x = conv_block(
                    x,
                    out_dim,
                    kernel_size=3,
                    strides=(1, 1),
                    use_batchnorm=True,
                    use_relu=True,
                    is_training=is_training,
                    prefix='block_{}'.format(bidx)
                )
        return x