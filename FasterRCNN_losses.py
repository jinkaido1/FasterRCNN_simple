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
        tx = (y_pred[:,:,0]-anchors[:,0])/anchors[:,2]
        ty = (y_pred[:,:,1]-anchors[:,1])/anchors[:,3]
        tw = tf.math.log( y_pred[:,:,2]/anchors[:,2])
        th = tf.math.log( y_pred[:,:,3]/anchors[:,3])

        gx = (y_true[:,:,0]-anchors[:,0])/anchors[:,2]
        gy = (y_true[:,:,1]-anchors[:,1])/anchors[:,3]
        gw = tf.math.log( y_true[:,:,2]/anchors[:,2])
        gh = tf.math.log( y_true[:,:,3]/anchors[:,3])

        return K.mean(K.square(tx-gx)+\
                      K.square(ty-gy)+\
                      K.square(tw-gw)+\
                      K.square(th-gh)\
                      , axis=-1)

    return loss