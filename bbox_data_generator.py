import numpy as np 
import keras
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from PIL import Image
from generate_anchors import generate_anchors_simple
import pickle

#R is xmin ymin xmax ymax
def area(R1, R2):
    dx = min(R1[2], R2[2]) - max(R1[0], R2[0])
    dy = min(R1[3], R2[3]) - max(R1[1], R2[1])
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0

class boundingBoxImageDataGenerator( keras.utils.Sequence):

    def __init__(self, img_folder, bbox_label_file_csv,\
        anchor_boxes, anchor_img_num_rows, anchor_img_num_cols,
        batch_size = 32,
        num_images_per_epoch = 1, 
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

        #use_preloaded_data = False
        use_preloaded_data = True
        #cached_filename = '/home/manju/code/ML/data/global-wheat-detection/cached/keras_generator_bbox_imgids_2500.npy'
        #cached_filename = '/home/manju/code/ML/data/global-wheat-detection/cached/keras_generator_bbox_imgids_25000.npy'
        cached_filename = '/home/manju/code/ML/data/global-wheat-detection/cached/keras_generator_bbox_imgids_all.npy'
        #Read from raw files and save a cached file for next time
        if not use_preloaded_data:
            for i, row in df.iterrows():
                #if i > 25000:
                #  break
                if i%1000==0:
                    print(str(i) + ' of ' + str(len(df)) + ' bboxes read')
                img_id = row['image_id']
                bbox_str = df['bbox'][i]
                bbox_str = bbox_str[1:-1]
                s = StringIO( bbox_str)
                #Note bbox is x,y,w,h.
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

            overwrite_saved_data = True
            if overwrite_saved_data:
                with open(cached_filename, 'wb') as f:
                    data = {}
                    data['img_ids'] = self.img_ids
                    data['bbox_data'] = self.bbox_data
                    pickle.dump(data, f)
                    f.close()
            print( len(self.img_ids))
            print( len(self.bbox_data))
        #Load from cached saved data 
        else:
            with open(cached_filename, 'rb') as f:
                data = pickle.load(f)
                #print(data)
                #print(data.keys)
                self.bbox_data = data['bbox_data']
                self.img_ids = data['img_ids']
            print( len(self.img_ids))
            print( len(self.bbox_data))
            #input('d2')

        self.Y_reg = np.zeros((anchor_boxes.shape[0],4))
        self.Y_cls = np.zeros((anchor_boxes.shape[0],2))

        #Get image size from input, make sure all BBs are scaled according
        #desired input image size to network

    def __len__(self):
        return int(self.num_images_per_epoch/self.num_images_per_batch)

    def __getitem__(self, index):

        display = True
        #display = False

        self.Y_reg.fill(0)
        self.Y_cls.fill(0)

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
            #print(i, a)
            a_bbox = [a[0], a[1], a[2], a[3]]
            #print(a_bbox)
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
        #for i in range(0, len(sorted_indexes)):
        #    print( i, anchor_scores[sorted_indexes[i]])
        #TODO - Dont sort by randomly pick boxes that are positive > threshold
        #Positive examples 
        pos_score_threshold = 0.5
        neg_score_threshold = 0.1
        positives = np.empty((0,4))
        positive_indexes = []
        for i in range(0, int(self.batch_size/2)):
            #print(i, anchor_scores[sorted_indexes[i]])
            if anchor_scores[sorted_indexes[i]]>=pos_score_threshold:
                positives = np.append(positives, self.anchor_boxes[sorted_indexes[i]])
                positive_indexes = np.append( positive_indexes, sorted_indexes[i])
            else:
                break
        positives = np.reshape(positives, (int(len(positives)/4), 4))

        #num negatives that need to be added to positives to meet batch_size
        num_negatives = self.batch_size-positives.shape[0]
        #Negative examples
        #TODO - Remove sorting for negatives - pick at random
        negatives = np.empty((0,4))
        negative_indexes = []
        for i in range(len(sorted_indexes)-1,\
             len(sorted_indexes)-num_negatives-1, -1):
            if anchor_scores[sorted_indexes[i]]<pos_score_threshold:# and\
                #anchor_scores[sorted_indexes[i]]>neg_score_threshold:
                negatives = np.append(negatives, self.anchor_boxes[sorted_indexes[i]])
                negative_indexes = np.append( negative_indexes, sorted_indexes[i])
            else:
                break
        negatives = np.reshape(negatives, (int(len(negatives)/4), 4))
        negative_indexes = negative_indexes.astype(int)
             
        #print( positives.shape)
        #print( negatives.shape)
        #print( positive_indexes)
        #print( negative_indexes)
        
        #Format output correctly for generator
        #Find best GT BB corresponding to positives
        positives_bbox = np.empty((0,4))

        for i in range(0, positives.shape[0]):
            anchor = positives[i, :]
            max_area = 0
            max_index = -1
            for j, bbox in enumerate(self.bbox_data[img_id]):
                R1 = np.copy(bbox)
                R1[2] += R1[0]
                R1[3] += R1[1]
                R2 = np.copy(anchor)
                R2[2] += R2[0]
                R2[3] += R2[1]

                curr_area = area(R1, R2)
                if(curr_area>max_area):
                    max_area = curr_area
                    max_index = j
            self.Y_reg[max_index,:] = self.bbox_data[img_id][max_index]# - anchor
            self.Y_cls[max_index,:] = [0, 1]
            if display:
                positives_bbox = np.append( positives_bbox, self.bbox_data[img_id][max_index])

        positives_bbox = np.reshape(positives_bbox, (int(len(positives_bbox)/4), 4))

        for i in range(0, negatives.shape[0]):
            #print( negative_indexes[i])
            self.Y_cls[negative_indexes[i],:] = [1, 0]

        #print( self.Y_reg)
        #print( self.Y_cls)

        #display = True
        display = False
        if display:
          plt.clf()
          ax = plt.subplot(2,2,1)
          ax.imshow(im)
          ax.set_title('GT detections')
          for bbox in self.bbox_data[img_id]:
            #print(bbox)
            rect = patches.Rectangle((bbox[0],bbox[1]), bbox[2], bbox[3],\
                 facecolor='none', edgecolor='r')
            ax.add_patch(rect)
          ax = plt.subplot(2,2,2)
          ax.imshow(positive_im_arr)
          ax.set_title('All postives map')
          ax = plt.subplot(2,2,3)
          ax.imshow(im)
          for i in range(0, positives.shape[0]):
            anchor = positives[i, :]
            anchor_bb = positives_bbox[i,:]
            rect = patches.Rectangle((anchor[0], anchor[1]), anchor[2], anchor[3],\
                 facecolor='none', edgecolor='g')
            ax.add_patch(rect)
            rect = patches.Rectangle((anchor_bb[0], anchor_bb[1]), anchor_bb[2], anchor_bb[3],\
                 facecolor='none', edgecolor='r')
            ax.add_patch(rect)
            
          for i in range(0, negatives.shape[0]):
            anchor = negatives[i, :]
            rect = patches.Rectangle((anchor[0], anchor[1]), anchor[2], anchor[3],\
                 facecolor='none', edgecolor='b')
            ax.add_patch(rect)
          ax.set_title('Sample positive and negative anchors')
 
          plt.waitforbuttonpress()

        im_arr = keras.applications.vgg16.preprocess_input(np.array(im))

        X = np.expand_dims(im_arr, axis=0) 
        return (X, \
            {'class_output':np.expand_dims(self.Y_cls, axis=0),\
              'bbox_output':np.expand_dims(self.Y_reg, axis=0)})
