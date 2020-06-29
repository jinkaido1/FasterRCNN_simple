from bbox_data_generator import boundingBoxImageDataGenerator
from generate_anchors import generate_anchors_simple
import tensorflow as tf

anchor_img_num_rows = 244
anchor_img_num_cols = 244

num_anchors = 200
anchors = generate_anchors_simple( anchor_img_num_rows, anchor_img_num_cols, num_anchors )
#starting_fraction = 0
#ending_fraction = 0.8
starting_fraction = 0.8
ending_fraction = 1


bbox_gen = boundingBoxImageDataGenerator('/home/manju/code/ML/data/global-wheat-detection/train',\
    '/home/manju/code/ML/data/global-wheat-detection/train.csv',\
    anchors, anchor_img_num_rows, anchor_img_num_cols, start_fraction=starting_fraction,
    end_fraction=ending_fraction)

import numpy as np 
from FasterRCNN_losses import bounding_box_loss, class_loss

for k in bbox_gen:
    print(k)
    print(k[1]['class_output'])
    print(k[1]['bbox_output'])
    bbox_rand = np.random.rand(k[1]['bbox_output'].shape[0],\
         k[1]['bbox_output'].shape[1],
         k[1]['bbox_output'].shape[2])
    bbox_ones = np.ones((k[1]['bbox_output'].shape[0],\
         k[1]['bbox_output'].shape[1],
         k[1]['bbox_output'].shape[2]))
    bbox_zeros = np.zeros((k[1]['bbox_output'].shape[0],\
         k[1]['bbox_output'].shape[1],
         k[1]['bbox_output'].shape[2]))
    class_rand = np.random.rand(k[1]['class_output'].shape[0],\
         k[1]['class_output'].shape[1],
         k[1]['class_output'].shape[2])
    class_zeros = np.zeros((k[1]['class_output'].shape[0],\
         k[1]['class_output'].shape[1],
         k[1]['class_output'].shape[2]))
    class_ones = np.ones((k[1]['class_output'].shape[0],\
         k[1]['class_output'].shape[1],
         k[1]['class_output'].shape[2]))
    
    db = bbox_rand-k[1]['bbox_output']
    dc = class_rand-k[1]['class_output']
    print( k[1]['bbox_output'][0,1:10,:])

    bbox_loss = bounding_box_loss(anchors)
    lb = bbox_loss( k[1]['bbox_output'], bbox_rand )
    #lb = bbox_loss( k[1]['bbox_output'], -bbox_ones )
    #lb = bbox_loss( k[1]['bbox_output'], bbox_zeros )
    print(lb)
    #bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    bce = class_loss()
    #lc = bce(k[1]['class_output'], class_rand)
    #lc = bce(k[1]['class_output'], class_zeros)
    lc = bce(k[1]['class_output'], 1*class_ones)
    s = np.sum(k[1]['class_output'], axis=1)
    print(lc)
    print(s)
    input('s')