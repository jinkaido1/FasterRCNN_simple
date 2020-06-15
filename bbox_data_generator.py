import numpy as np 
import keras
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from PIL import Image
from generate_anchors import generate_anchors_simple

class boundingBoxImageDataGenerator( keras.utils.Sequence):

    def __init__(self, img_folder, bbox_label_file_csv,\
        anchor_boxes, anchor_img_num_rows, anchor_img_num_cols,
        batch_size = 32,
        num_images_per_epoch = 1000, 
        shuffle=True, hard_negative=False):
        self.img_folder = img_folder
        self.bbox_label_file_csv = bbox_label_file_csv
        self.anchor_boxes = anchor_boxes
        self.anchor_img_num_rows = anchor_img_num_rows
        self.anchor_img_num_cols = anchor_img_num_cols
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
          #Figure out how to modify Bounding boxes to match desired anchor image size
          im = Image.open( self.img_folder+'/'+img_id+'.jpg')
          scale_y = self.anchor_img_num_rows*1.0/im.size[1]
          scale_x = self.anchor_img_num_cols*1.0/im.size[0]
          bbox[0] *= scale_x
          bbox[1] *= scale_y
          bbox[2] *= scale_x
          bbox[3] *= scale_y
          bbox = bbox.astype(int)

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
        #Load positive bounding boxes from image
        #Sort all anchors. Pick the first batch_size/2 which have ROI overlap > threshold
        #From the ones that have ROI overlap < threshold, pick the first batch_size - num_positive
        if( index >= len(self.img_ids)):
            index = 0

        

        #Load image pertaining to index
        img_id = self.img_ids[index]
        im = Image.open( self.img_folder+'/'+img_id+'.jpg')
        anchor_y = self.anchor_img_num_rows
        anchor_x = self.anchor_img_num_cols
        #Resize image to match anchor image size
        im = im.resize((anchor_x, anchor_y))

        #Create a binary image with all positive BBs filled in
        #For each anchor box, compute overlap % between anchor and positive image
        #Sort anchor boxes by decreasing score
        positive_im_arr = np.zeros((anchor_y, anchor_x))
        for bbox in self.bbox_data[img_id]:
            positive_im_arr[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = 1
        
        anchor_scores = np.zeros(self.anchor_boxes.shape[0])

        for i, a in enumerate(self.anchor_boxes, start=0):
            print(i, a)
            a_bbox = [a[0], a[1], a[2], a[3]]
            print(a_bbox)
            num_positive = np.sum(\
                positive_im_arr[a_bbox[1]:a_bbox[1]+a_bbox[3],\
                                a_bbox[0]:a_bbox[0]+a_bbox[2]])
            a_bbox_num_pixels = a_bbox[2]*a_bbox[3]
            score = num_positive/a_bbox_num_pixels
            anchor_scores[i] = score
            if score>1:
              print(num_positive, a_bbox_num_pixels, )
              input('a')

        sorted_indexes = np.argsort(-anchor_scores)
        for i in range(0, len(sorted_indexes)):
            print( i, anchor_scores[sorted_indexes[i]])

        #Positive examples 
        score_threshold = 0.5
        positives = np.empty((0,4))
        for i in range(0, int(self.batch_size/2)):
            print(i, anchor_scores[sorted_indexes[i]])
            if anchor_scores[sorted_indexes[i]]>=score_threshold:
                positives = np.append(positives, self.anchor_boxes[i])
            else:
                break
        positives = np.reshape(positives, (int(len(positives)/4), 4))
        #Negative examples
        negatives = np.empty((0,4))
        for i in range(len(sorted_indexes)-1,\
             len(sorted_indexes)-int(self.batch_size/2)-1, -1):
            if anchor_scores[sorted_indexes[i]]<score_threshold:
                negatives = np.append(negatives, self.anchor_boxes[i])
            else:
                break
        negatives = np.reshape(negatives, (int(len(negatives)/4), 4))
             
        print( positives.shape)
        print( negatives.shape)

        #TODO - Display and verify positive and negative examples
        #Format output correctly for generator
        #Verify loss function is still ok after moving to off-center BBox definition (x,y,w,h)
 
        display = True
        #display = False
        if display:
          plt.clf()
          ax = plt.subplot(2,2,1)
          ax.imshow(im)
          for bbox in self.bbox_data[img_id]:
            #print(bbox)
            rect = patches.Rectangle((bbox[0],bbox[1]), bbox[2], bbox[3],\
                 facecolor='none', edgecolor='r')
            ax.add_patch(rect)
          ax = plt.subplot(2,2,2)
          ax.imshow(positive_im_arr)
        
          plt.waitforbuttonpress()

       





        return im, 0
