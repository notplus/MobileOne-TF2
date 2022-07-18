import tensorflow as tf

from mobileone import MobileOneBlock, make_mobileone_s0

if __name__ == '__main__':
    xx = tf.random.normal((1, 128, 128, 4))

    block = MobileOneBlock(4, 4, 4, 1, deploy=False)
    block.build((1, 32, 32, 4))
    train_y = block(xx)

    block.switch_to_deploy()

    deploy_y = block(xx)
    print(tf.reduce_sum((train_y - deploy_y) ** 2).numpy())

    x = tf.random.normal((1, 128, 128, 3))
    model = make_mobileone_s0(width_mult=1, deploy=False)
    model.build(input_shape=(1, 128, 128, 3))
    model.summary()

    train_y = model(x)

    for layer in model.layers:
        for block in layer.layers:
            if hasattr(block, 'switch_to_deploy'):
                block.switch_to_deploy()

    deploy_y = model(x)
    print(tf.reduce_sum((train_y - deploy_y)**2).numpy())