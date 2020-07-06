import tensorflow as tf 
import numpy as np
from keras.models import model_from_json
from bbox_data_generator import boundingBoxImageDataGenerator
from generate_anchors import generate_anchors_simple
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from PIL import Image

def to_softmax( values ):
    epsilon = 1e-10
    v_exp = np.exp(values)
    v_exp_sum = np.sum(v_exp,axis=1)+epsilon
    v_exp_sum_rep = np.zeros(v_exp.shape)
    v_exp_sum_rep[:,0] = v_exp_sum
    v_exp_sum_rep[:,1] = v_exp_sum
    prob = v_exp/v_exp_sum_rep
    return prob

json_file = open('models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("models/model-epoch-000010-loss-1672.95.hdf5")
# loaded_model.load_weights("models/final_model_weights.hdf5")
print(loaded_model.summary())

img_num_rows = 224
img_num_cols = 224
anchors_k = 200
start_val_fraction = 0.81
end_val_fraction = 1

anchors = generate_anchors_simple( img_num_rows, img_num_cols, anchors_k )

bbox_gen_val = boundingBoxImageDataGenerator('/home/manju/code/ML/data/global-wheat-detection/train',\
    '/home/manju/code/ML/data/global-wheat-detection/train.csv',\
    anchors, img_num_rows, img_num_cols, start_fraction=start_val_fraction,
    end_fraction=end_val_fraction)

max_num_test = 10
num_test = 0

for k in bbox_gen_val:
    im_in = k[0]
    im_arr = np.squeeze(im_in,0)
    prediction = loaded_model.predict(im_in)
    bbox_p = np.squeeze(prediction[0],0)
    class_p = np.squeeze(prediction[1],0)

    class_prob = to_softmax(class_p)
    
    plt.clf()
    ax = plt.subplot(1,2,1)
    ax.imshow(im_arr)

    positive_thresh = 0.99
    negative_thresh = 0.99

    for prob, bbox, anchor in zip( class_prob, bbox_p, anchors):
      ba = anchor+bbox
      print (prob, ba)
      #rect = patches.Rectangle((anchor[0], anchor[1]), anchor[2], anchor[3],\
      #     facecolor='none', edgecolor='r')
      #ax.add_patch(rect)
      if(prob[0]>negative_thresh):
        rect = patches.Rectangle((ba[0],ba[1]), ba[2], ba[3],\
             facecolor='none', edgecolor='b')
        ax.add_patch(rect)
      if(prob[1]>positive_thresh):
        rect = patches.Rectangle((ba[0],ba[1]), ba[2], ba[3],\
             facecolor='none', edgecolor='r')
        ax.add_patch(rect)
    print( bbox_p.shape, bbox_p.shape)
    plt.waitforbuttonpress()