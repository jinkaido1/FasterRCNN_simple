from bbox_data_generator import boundingBoxImageDataGenerator

bbox_gen = boundingBoxImageDataGenerator('/home/manju/code/ML/data/global-wheat-detection/train',\
    '/home/manju/code/ML/data/global-wheat-detection/train.csv')

for k in bbox_gen:
    print('stop')