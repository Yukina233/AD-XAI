import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras import losses


### Average distance from the Center of Mass
def com_conv(G_out, beta ,power):
    def dispersion_loss(y_true, y_pred):
        loss_b = tf.reduce_mean(losses.binary_crossentropy(y_true, y_pred))
        
        center = tf.reduce_mean(G_out, axis=0, keepdims=True)
        distance_xy = tf.pow(tf.abs(tf.subtract(G_out,center)),power)
        distance = tf.reduce_sum(distance_xy, (1,2,3))
        avg_distance = tf.reduce_mean(tf.pow(tf.abs(distance), 1/power))
        loss_d = tf.reciprocal(avg_distance)
        
        loss = loss_b + beta*loss_d
        return loss
    return dispersion_loss


def com_conv_new(G_out, beta, power):
    def dispersion_loss(y_true, y_pred):
        loss_b = tf.reduce_mean(losses.binary_crossentropy(y_true, y_pred))

        center = tf.reduce_mean(G_out, axis=0, keepdims=True)
        distance_xy = tf.pow(tf.abs(tf.subtract(G_out, center)), power)
        distance = tf.reduce_sum(distance_xy, 1)
        avg_distance = tf.reduce_mean(tf.pow(tf.abs(distance), 1 / power))
        loss_d = tf.reciprocal(avg_distance)

        loss = loss_b + beta * loss_d
        return loss

    return dispersion_loss