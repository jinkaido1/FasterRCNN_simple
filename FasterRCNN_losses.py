import tensorflow as tf
import keras.backend as K

# Define custom loss
def custom_loss(layer):

    def loss(y_true,y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)
   
    # Return a function
    return loss

def bounding_box_loss(anchors):
#
    def loss(y_true,y_pred):
        epsilon = 1e-10
        keep_indexes = y_true[:,:,3]!=0
        y_t = y_true[keep_indexes]
        y_p = y_pred[keep_indexes]
        y_p += epsilon
        y_t += epsilon
        #a = anchors[tf.squeeze(keep_indexes, axis=0)]
        keep_indexes_where = tf.where(keep_indexes)
        a = tf.gather(anchors, keep_indexes_where[:,1])
        a = tf.dtypes.cast(a, dtype=tf.float32 )

        tx = (y_p[:,0]-a[:,0])/a[:,2]
        ty = (y_p[:,1]-a[:,1])/a[:,3]
        tw = tf.math.log( y_p[:,2]/a[:,2])
        th = tf.math.log( y_p[:,3]/a[:,3])

        gx = (y_t[:,0]-a[:,0])/a[:,2]
        gy = (y_t[:,1]-a[:,1])/a[:,3]
        gw = tf.math.log( y_t[:,2]/a[:,2])
        gh = tf.math.log( y_t[:,3]/a[:,3])

        L = K.mean(K.square(tx-gx)+\
                      K.square(ty-gy)+\
                      K.square(tw-gw)+\
                      K.square(th-gh)\
                      , axis=-1)
        return L

    return loss


def class_binary_cross_entropy_loss():
#
    def loss(y_true,y_pred):
        epsilon = 1e-10
        y_true_sum = K.sum(y_true,2)
        keep_indexes_0 = y_true[:,:,0]==1
        keep_indexes_1 = y_true[:,:,1]==1
        keep_indexes = keep_indexes_0 | keep_indexes_1
        y_t = y_true[keep_indexes]
        y_p = y_pred[keep_indexes]
        y_p += epsilon
        y_t += epsilon
        y_p_exp = K.exp(y_p)
        y_p_exp_sum = K.sum( y_p_exp, 1)
        y_p_exp_sum_repeat = K.repeat( K.expand_dims(y_p_exp_sum), 2)
        y_p_exp_sum_repeat = K.squeeze(y_p_exp_sum_repeat, 2)+epsilon
        y_p_prob = y_p_exp/y_p_exp_sum_repeat

        print( y_p)
        print( y_p_exp)
        print( y_p_exp_sum)
        print( y_p_exp_sum_repeat)
        print( y_p_prob)

        Loss = -y_t*K.log(y_p_prob)
        L = K.mean(Loss)
        print(Loss)
        print(L)
        return L

    return loss

def class_loss():
#
    def loss(y_true,y_pred):
        epsilon = 1e-10
        y_true_sum = K.sum(y_true,2)
        keep_indexes_0 = y_true[:,:,0]==1
        keep_indexes_1 = y_true[:,:,1]==1
        keep_indexes = keep_indexes_0 | keep_indexes_1
        y_t = y_true[keep_indexes]
        y_p = y_pred[keep_indexes]
        y_p += epsilon
        y_t += epsilon
        Loss = -y_t*K.log(y_p)
        L = K.mean(Loss)
        #print(Loss[0:10,:])
        #print(L)
        return L

    return loss