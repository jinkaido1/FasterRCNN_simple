import numpy as np 
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
        tx = (y_pred[:,:,0]-anchors[:,:,0])/anchors[:,:,2]
        ty = (y_pred[:,:,1]-anchors[:,:,1])/anchors[:,:,3]
        tw = np.log( y_pred[:,:,2]/anchors[:,:,2])
        th = np.log( y_pred[:,:,3]/anchors[:,:,3])

        gx = (y_true[:,:,0]-anchors[:,:,0])/anchors[:,:,2]
        gy = (y_true[:,:,1]-anchors[:,:,1])/anchors[:,:,3]
        gw = np.log( y_true[:,:,2]/anchors[:,:,2])
        gh = np.log( y_true[:,:,3]/anchors[:,:,3])

        return K.mean(K.square(tx-gx)+\
                      K.square(ty-gy)+\
                      K.square(ty-gy)+\
                      K.square(ty-gy)\
                      , axis=-1)

    return loss