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
from FasterRCNN_losses import bounding_box_loss, class_loss, class_binary_cross_entropy_loss

for i in range(1,10):
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
    sigmoid_rand = np.exp(abs(class_rand))
    #sigmoid_rand[0,:,0] = 1
    #sigmoid_rand[0,:,1] = 0

    sigmoid_sum = np.zeros( sigmoid_rand.shape)
    sigmoid_sum[0,:,0] = np.sum( sigmoid_rand, 2)
    sigmoid_sum[0,:,1] = np.sum( sigmoid_rand, 2)
    class_sigmoid = sigmoid_rand/sigmoid_sum

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
    #ce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    #ce = class_loss()
    ce = class_binary_cross_entropy_loss()
    #lc = ce(k[1]['class_output'], class_rand)
    lc = ce(k[1]['class_output'], abs(class_rand))
    #lc = ce(k[1]['class_output'], class_zeros)
    #lc = ce(k[1]['class_output'], 0.5*class_ones)
    #lc = ce(k[1]['class_output'], k[1]['class_output'])

    s = np.sum(k[1]['class_output'], axis=1)
    print(lc)
    print(s)
    input('s')