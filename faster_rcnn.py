import tensorflow as tf 
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.utils.vis_utils import plot_model
from keras import layers
from keras import Model
from keras import optimizers
import keras.backend as K
from FasterRCNN_losses import bounding_box_loss, class_binary_cross_entropy_loss
from bbox_data_generator import boundingBoxImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import initializers
import datetime

# Define custom loss
def custom_loss_2(layer):

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)
   
    # Return a function
    return loss


##1. Generate simple network that
##a. Uses VGG 16 backbone, but ignores later layers
##b. Adds region proposal n/w
##2. Data generator that goes over wheat data and generates training data
##a. What format of input does nw need
##b. Simple generator
##3. Train
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

#Set layers to be trainable    
model.layers[-1].trainable = True
model.layers[-2].trainable = True

plot_model( model, to_file='vgg_part.png')
print( model.summary())

new_dense_1 = layers.Dense(256, name='new_dense_1')(model.layers[-1].output)
new_dense_1_flat = layers.Flatten()(new_dense_1)

img_num_rows = 224
img_num_cols = 224
anchors_k = 200
alpha_class = 1000
alpha_bbox = 1
#Get img size from model instead of hard coded
anchors = generate_anchors_simple( img_num_rows, img_num_cols, anchors_k )
k = anchors.shape[0]
print( anchors.shape )

class_predictions = layers.Dense(k*2, activation='relu',\
    name='class_predictions')(new_dense_1_flat)
#class_predictions = layers.Dense(k*2, name='class_predictions')(new_dense_1_flat)
bbox_predictions = layers.Dense(k*4, activation='relu', name='bbox_predictions')(new_dense_1_flat)

class_output = layers.Reshape((k,2), name='class_output')(class_predictions)
bbox_output = layers.Reshape((k,4), name='bbox_output')(bbox_predictions)

new_model = Model(inputs=[model.input], outputs=[bbox_output, class_output])
#new_model = Model(inputs=[model.input], outputs=[class_output])
#new_model = Model(inputs=[model.input], outputs=[class_predictions, bbox_predictions])

plot_model( new_model, to_file='vgg_extend.png')
print( new_model.summary())

#opt = optimizers.Adam(learning_rate=0.0001)
#opt = optimizers.Adam(learning_rate=.0001)
#opt = optimizers.SGD(learning_rate=.0001)
opt = optimizers.RMSprop(learning_rate=.00001)

bbox_loss = bounding_box_loss(anchors)
cl_loss = class_binary_cross_entropy_loss()

#Verify loss function is still ok after moving to off-center BBox definition (x,y,w,h)
new_model.compile( optimizer=opt, \
    loss={
    'bbox_output':bbox_loss,
    'class_output':cl_loss}, 
    loss_weights={'class_output':alpha_class, 'bbox_output':alpha_bbox})

#new_model.compile( optimizer=opt, \
#    loss={
#    'class_output':cl_loss}, 
#    loss_weights={'class_output':alpha_class})

#Data generator
start_train_fraction = 0
end_train_fraction = .8
start_val_fraction = .8
end_val_fraction = 1

bbox_gen_train = boundingBoxImageDataGenerator('/home/manju/code/ML/data/global-wheat-detection/train',\
    '/home/manju/code/ML/data/global-wheat-detection/train.csv',\
    anchors, img_num_rows, img_num_cols, start_fraction=start_train_fraction,
    end_fraction=end_train_fraction)
bbox_gen_val = boundingBoxImageDataGenerator('/home/manju/code/ML/data/global-wheat-detection/train',\
    '/home/manju/code/ML/data/global-wheat-detection/train.csv',\
    anchors, img_num_rows, img_num_cols, start_fraction=start_val_fraction,
    end_fraction=end_val_fraction)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_dir = "models/model-epoch-{epoch:06d}-loss-{loss:.2f}.hdf5"
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_dir,
    save_weights_only=True,
    monitor='loss',
    period = 5,
    save_best_only=True,
    verbose=1
    )

reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=250, cooldown=250, min_lr=0.00000000001,
                              verbose=1)

#Save model json before training
new_model_json = new_model.to_json()
with open("models/model.json", "w") as json_file:
    json_file.write(new_model_json)

new_model.fit_generator( generator = bbox_gen_train,\
    validation_data= bbox_gen_val,
    epochs = 10,
    steps_per_epoch=1,
#    callbacks=[tensorboard_callback, reduce_lr_callback])
    callbacks=[tensorboard_callback, reduce_lr_callback, model_checkpoint_callback] )

#new_model.fit_generator( generator = bbox_gen,\
#    epochs = 100,
#    steps_per_epoch=1)

new_model.save_weights('models/final_model_weights.hdf5')

#Why num_images_per_batch > 1 does not work?
#Remove sorting of anchor scores
#Review use of k=200. Should it come automatically from CNN feature size?
#Use tensorboard
#Use all images and use pickle after first load of all images
