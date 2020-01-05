"""
Usage:
  # Make sure you have the folder structure as it is in the github repo
  
  python pcb_errors.py --input_image=path_to_image

"""

import pandas as pd
import cv2
import numpy as np
import pickle
import sys

import os
# base_folder = "/Users/aishwaryamalgonde/Aishwarya/nanonets"
input_folder = 'inputs'
# input_image_local = '/Users/aishwaryamalgonde/Aishwarya/nanonets/data/2.2.jpg'
# os.chdir(base_folder)

import tensorflow as tf
flags = tf.app.flags
flags.DEFINE_string('input_image', '','Path to the input image file of a pcb')
FLAGS = flags.FLAGS
FLAGS(sys.argv, known_only=True)

positions = pd.read_csv(os.path.join(input_folder,"avg_positions_of_components.csv"))
# positions.head()
with open(os.path.join(input_folder,'missing_threshold_mean.txt'), 'rb') as handle:
    missing_threshold = pickle.loads(handle.read())
# missing_threshold

missing_components_list = ['3_b', '4_b', '33_e', '20_f', '25_f', '24_h', '28_i', '29_i', '30_i', 
                           '45_i', '47_n', '48_n', '11_q', '13_s', '42_z', '53_ac', '55_l', '56_ae', '57_af']
rotated_components_list = ['6_6', '20_f', '25_f', '28_i', '30_i', '36_i', '45_i', '44_m', 
                            '46_n', '51_n', '47_n', '48_n', '60_o', '13_s', '27_x', '42_z']

common_components_list = []
onlyr_components_list = []
for c in rotated_components_list:
    if c in missing_components_list:
        common_components_list.append(c)
    else:
        onlyr_components_list.append(c)
# common_components_list

onlym_components_list = [x for x in missing_components_list if x not in common_components_list]
# onlym_components_list

test_img = cv2.imread(FLAGS.input_image)
ref_img = cv2.imread(os.path.join(os.getcwd(),input_folder,'ref_img.jpg'))

def load_model(model_path):    
    from keras import backend as K
    # This line must be executed before loading Keras model.
    K.set_learning_phase(0)

    from keras.models import load_model
    loaded_model = load_model(model_path)
    return loaded_model

def predict_model(model_path, test_crop):
    model = load_model(model_path)
    
    img_width = 224
    img_height = 224
    x = cv2.resize(test_crop, (img_width, img_height))
#     x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255.
    
    classes = model.predict(x)
    return classes[0][0]

def main(_):
    error = 0
    ## check for missing components and if they are present then check for rotation
    for c in common_components_list:
        threshold = missing_threshold[c]
        position = positions.loc[positions['component'] == c]
        
        xmin_avg = int(position['xmin_avg'].mean())
        ymin_avg = int(position['ymin_avg'].mean())
        xmax_avg = int(position['xmax_avg'].mean())
        ymax_avg = int(position['ymax_avg'].mean())
        
        ref_crop = ref_img[ymin_avg:ymax_avg, xmin_avg:xmax_avg, :]
        test_crop = test_img[ymin_avg:ymax_avg, xmin_avg:xmax_avg, :]
        
        ref_mean = np.mean(ref_crop, axis = 2)
        test_mean = np.mean(test_crop, axis = 2)
        output = abs(np.mean(ref_mean-test_mean))
        
        if (output > threshold):
            error += 1
            print('Component', c, 'Missing')
        else:
            model_path = os.path.join(input_folder, 'rotation_detection_models', c, 'vgg16', 'vgg16_best_ld')
            rotation = predict_model(model_path, test_crop)
            if rotation>=0.09:
                error += 1
                print('Component', c.split('_')[0], 'Rotated')

    ## check for missing components only
    for c in onlym_components_list:
        threshold = missing_threshold[c]
        position = positions.loc[positions['component'] == c]
        
        xmin_avg = int(position['xmin_avg'].mean())
        ymin_avg = int(position['ymin_avg'].mean())
        xmax_avg = int(position['xmax_avg'].mean())
        ymax_avg = int(position['ymax_avg'].mean())
        
        ref_crop = ref_img[ymin_avg:ymax_avg, xmin_avg:xmax_avg, :]
        test_crop = test_img[ymin_avg:ymax_avg, xmin_avg:xmax_avg, :]
    
        ref_mean = np.mean(ref_crop, axis = 2)
        test_mean = np.mean(test_crop, axis = 2)
        output = abs(np.mean(ref_mean-test_mean))

        if (output > threshold):
            error += 1
            print('Component', c.split('_')[0], 'Missing')

    ## check for rotation of components olny
    for c in onlyr_components_list:
        
        position = positions.loc[positions['component'] == c]
        xmin_avg = int(position['xmin_avg'].mean())
        ymin_avg = int(position['ymin_avg'].mean())
        xmax_avg = int(position['xmax_avg'].mean())
        ymax_avg = int(position['ymax_avg'].mean())
        
        test_crop = test_img[ymin_avg:ymax_avg, xmin_avg:xmax_avg, :]
        
        model_path = os.path.join(input_folder, 'rotation_detection_models', c, 'vgg16', 'vgg16_best_ld')
        rotation = predict_model(model_path, test_crop)
        
        if rotation>0.09:
            error += 1
            print('Component', c.split('_')[0], 'Rotated')

    if error == 0:
        print('No Errors')

if __name__ == '__main__':
    tf.app.run()

