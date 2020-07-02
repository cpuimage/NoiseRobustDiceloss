# NoiseRobustDiceloss
Unofficial Tensorflow implementation of Noise-robust Dice loss for image segmentation

 
```python
import tensorflow as tf
def noise_robust_dice_loss(y_true, y_pred, beta=1.5, epsilon=1e-5) -> tf.Tensor:
    axis = list(range(1, len(y_true.get_shape().as_list())))
    numerator = tf.math.reduce_sum(tf.math.pow(tf.math.abs(y_true - y_pred), beta), axis=axis)
    denominator = tf.math.reduce_sum(tf.math.square(y_true) + tf.math.square(y_pred), axis=axis) + epsilon
    loss = numerator / denominator
    return loss

```

# Reference 
[A Noise-robust Framework for Automatic Segmentation of COVID-19 Pneumonia Lesions from CT Images](https://ieeexplore.ieee.org/document/9109297) 

# Donating

If you found this project useful, consider buying me a coffee

<a href="https://www.buymeacoffee.com/gaozhihan" target="_blank"><img src="https://img2018.cnblogs.com/blog/824862/201809/824862-20180930223603138-1708589189.png" alt="Buy Me A Coffee" style="height: auto !important;width: auto !important;" ></a>
