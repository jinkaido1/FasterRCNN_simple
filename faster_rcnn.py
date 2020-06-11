import tensorflow as tf 
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.utils.vis_utils import plot_model
from keras import layers
from keras import Model
import keras.backend as K
from FasterRCNN_losses import bounding_box_loss

# Define custom loss
def custom_loss_2(layer):

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)
   
    # Return a function
    return loss


#1. Generate simple network that
#a. Uses VGG 16 backbone, but ignores later layers
#b. Adds region proposal n/w
#2. Data generator that goes over wheat data and generates training data
#a. What format of input does nw need
#b. Simple generator
#3. Train
#4. Callbacks + Tensorboard
#5. Validation and test set separation
#6. L1 loss instead of L2
#7. Data augmentation
#8. Replace intermediate layer by conv layer
#9. Replace region proposal nw by conv layer

from generate_anchors import generate_anchors_simple

print("hello")
model = VGG16()
plot_model( model, to_file='vgg_full.png')
print( model.summary())

for layer in model.layers:
    print(layer.name)

model.layers.pop()
model.layers.pop()
model.layers.pop()
model.layers.pop()

for layer in model.layers:
    layer.trainable = False

plot_model( model, to_file='vgg_part.png')
print( model.summary())

new_dense_1 = layers.Dense(256, name='new_dense_1')(model.layers[-1].output)
new_dense_1_flat = layers.Flatten()(new_dense_1)

anchors_k = 200
alpha = 0.5
#Get img size from model instead of hard coded
anchors = generate_anchors_simple( 244, 244, anchors_k)
k = anchors.shape[0]
print( anchors.shape )

class_predictions = layers.Dense(k*2, name='class_predictions')(new_dense_1_flat)
bbox_predictions = layers.Dense(k*4, name='bbox_predictions')(new_dense_1_flat)

class_output = layers.Reshape((k,2), name='class_output')(class_predictions)
bbox_output = layers.Reshape((k,4), name='bbox_output')(bbox_predictions)

new_model = Model(inputs=[model.input], outputs=[class_output, bbox_output])
#new_model = Model(inputs=[model.input], outputs=[class_predictions, bbox_predictions])

plot_model( new_model, to_file='vgg_extend.png')
print( new_model.summary())


new_model.compile( optimizer='adam', \
    loss={
    'class_output':'binary_crossentropy', 
    'bbox_output':bounding_box_loss(anchors)},
    loss_weights={'class_output':alpha, 'bbox_output':1-alpha})
    #'class_predictions':'binary_crossentropy', 
    #'bbox_predictions':'mse'},
    #loss_weights={'class_predictions':alpha, 'bbox_predictions':1-alpha})



