from bbox_data_generator import boundingBoxImageDataGenerator
from generate_anchors import generate_anchors_simple

anchor_img_num_rows = 244
anchor_img_num_cols = 244

num_anchors = 200
anchors = generate_anchors_simple( anchor_img_num_rows, anchor_img_num_cols, num_anchors )

bbox_gen = boundingBoxImageDataGenerator('/home/manju/code/ML/data/global-wheat-detection/train',\
    '/home/manju/code/ML/data/global-wheat-detection/train.csv',\
    anchors, anchor_img_num_rows, anchor_img_num_cols)

for k in bbox_gen:
    print('stop')