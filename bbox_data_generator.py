import numpy as np 
import keras
import pandas as pd
from io import StringIO
from PIL import Image

class boundingBoxImageDataGenerator( keras.utils.Sequence):

    def __init__(self, img_folder, bbox_label_file_csv, batch_size = 32,\
        num_images_per_epoch = 1000, 
        shuffle=True, hard_negative=False):
        self.img_folder = img_folder
        self.bbox_label_file_csv = bbox_label_file_csv
        self.batch_size = batch_size
        self.num_images_per_batch = 1
        self.num_images_per_epoch = num_images_per_epoch
        self.shuffle = shuffle
        self.hard_negative = hard_negative
        df = pd.read_csv( self.bbox_label_file_csv)
        self.bbox_data = {}
        for i, row in df.iterrows():
          if i > 2500:
            break
          if i%1000==0:
              print(str(i) + ' of ' + str(len(df)) + ' bboxes read')
          img_id = row['image_id']
          bbox_str = df['bbox'][i]
          bbox_str = bbox_str[1:-1]
          s = StringIO( bbox_str)
          bbox = np.loadtxt(s, delimiter=',')
          if img_id in self.bbox_data:
              self.bbox_data[img_id].append(bbox)
          else:
              self.bbox_data[img_id]=[]
              self.bbox_data[img_id].append(bbox)
          self.img_ids = list(self.bbox_data.keys())

        #Get image size from input, make sure all BBs are scaled according
        #desired input image size to network

    def __len__(self):
        return int(self.num_images_per_epoch/self.num_images_per_batch)

    def __getitem__(self, index):
        #Load image pertaining to index
        #Sort all anchors. Pick the first batch_size/2 which have ROI overlap > threshold
        #From the ones that have ROI overlap < threshold, pick the first batch_size - num_positive
        if( index >= len(self.img_ids)):
            index = 0
        im = Image.open( self.img_folder+'/'+self.img_ids[index]+'.jpg')

        return im, 0
