import tensorflow as tf


def noise_robust_dice_loss(y_true, y_pred, beta=1.5, epsilon=1e-5) -> tf.Tensor:
    axis = list(range(1, len(y_true.get_shape().as_list())))
    numerator = tf.math.reduce_sum(tf.math.pow(tf.math.abs(y_true - y_pred), beta), axis=axis)
    denominator = tf.math.reduce_sum(tf.math.square(y_true) + tf.math.square(y_pred), axis=axis) + epsilon
    loss = numerator / denominator
    return loss


def main():
    import tensorflow as tf
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    tf.compat.v1.enable_eager_execution()

    h, w = (512, 512)
    y_true = tf.random.normal(shape=(1, h, w, 1), dtype=tf.float32)
    y_pred = tf.random.normal(shape=(1, h, w, 1), dtype=tf.float32)
    print(noise_robust_dice_loss(y_true=y_true, y_pred=y_pred))


if __name__ == '__main__':
    main()
