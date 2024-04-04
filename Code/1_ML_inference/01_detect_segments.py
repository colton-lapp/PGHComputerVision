# 01_detect_segments.py
# Collects each object instance detected in the street segment images of a
# given neighborhood, including its bounding box (bbox) size and confidence.
#
# Usage: Run the following command in terminal (modified to your neighborhood of choice)
#   python 01_detect_segments.py -w path_to_weights.pt -s 1080
#   -d Data/ProcessedData/SFStreetView/segment_dictionary_MissionDistrictBlock.json
#   -o Outputs/Detection/Res_640/MissionDistrictBlock_2011-02-01_3/
#   -i Data/ProcessedData/SFStreetView/Res_640/MissionDistrictBlock_2011-02-01_3/
#
# Data inputs:
#   - Segment dictionary for the selected location (from 01_generate_street_segments.py)
#   - Directory containing the location's images (from 02_collect_street_segment_images.py)
#
# Outputs:
#   - CSV file including one row per detected object instance, saved to the
#     selected output_path

import argparse
import glob
import json
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm

import sys
sys.path.append('/Users/coltonlapp/Dropbox/My Mac (Coltons-MacBook-Pro.local)/Desktop/SCHOOL/Year2_Spring/IndepStudyProject/yoloso-urbanchange-6f00b5e')
from DataScripts.read_files import load_segment_dict
from DataScripts.urbanchange_utils import AppendLogger

from ultralytics import YOLO
yolo_model = 'v8'

run_from_command_line=False

if ~run_from_command_line:
    # hard code argparser input
    args = {
        'weights': 'path_to_weights.pt',
        'size': 640,
        'segment_dictionary': '/Users/coltonlapp/Dropbox/My Mac (Coltons-MacBook-Pro.local)/Desktop/SCHOOL/Year2_Spring/IndepStudyProject/yoloso-urbanchange-6f00b5e/Data/ProcessedData/SFStreetView/segment_dictionary_PittsburghDowntown.json',
        'output_path': '/Users/coltonlapp/Dropbox/My Mac (Coltons-MacBook-Pro.local)/Desktop/SCHOOL/Year2_Spring/IndepStudyProject/yoloso-urbanchange-6f00b5e/Data/Output/YOLO_output/',
        'input_images': '/Users/coltonlapp/Dropbox/My Mac (Coltons-MacBook-Pro.local)/Desktop/SCHOOL/Year2_Spring/IndepStudyProject/yoloso-urbanchange-6f00b5e/Data/ProcessedData/SFStreetView/Res_640/PittsburghDowntown_2024-03-20/'
    }


# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights', required=False,
                    help='Path to model weights')
parser.add_argument('-s', '--size', required=True, default=640,
                    help='Image size', type=int)
parser.add_argument('-d', '--segment_dictionary', required=True,
                    help='Path to segment JSON dictionary')
parser.add_argument('-o', '--output_path', required=True,
                    help='Output path for segment vectors')
parser.add_argument('-i', '--input_images', required=True,
                    help='Path to input images for inference')




def get_objects(img_result, model_names_list, image_path, seg_id, torch_device):
    """
    Returns a dictionary including the bbox size, confidence and class of each
    object instance detected in an image.
    :param img_result: (tensor) of size (number of objects detected, 6),
    where the columns represent: x1, y1, x2, x2, confidence, class
    :param model_names_list: (list) of classes being predicted (in the order
    they are being encoded)
    :param seg_id: (str)
    :param image_path: (str)
    :param torch_device: one of ['cuda', 'cpu']
    :return: (dict) including the count for each of the classes in model_names_list
    """
    object_dict = {}
    num_objects = img_result.shape[0]

    if torch_device == 'cuda':
        img_result = img_result.cpu()
    img_result = img_result.numpy()

    # Add image and segment ID
    object_dict['segment_id'] = [seg_id] * num_objects
    image_id = image_path.split(os.path.sep)[-1]. \
        split('img_{}_'.format(seg_id))[-1].split('.png')[0]
    object_dict['img_id'] = [image_id] * num_objects

    # Get objects
    object_dict['confidence'] = img_result[:, 4].tolist()
    object_dict['bbox_size'] = np.multiply(
        img_result[:, 2] - img_result[:, 0], img_result[:, 3] - img_result[:, 1]).tolist()

    # get coordinates for bbox
    object_dict['x1'] = img_result[:, 0].tolist()
    object_dict['y1'] = img_result[:, 1].tolist()
    object_dict['x2'] = img_result[:, 2].tolist()
    object_dict['y2'] = img_result[:, 3].tolist()

    for vv in object_dict['y2']:
        if type(vv) == str:
            print(vv)

    # Get object classes
    obj_classes = img_result[:, 5].tolist()
    object_dict['class'] = [model_names_list[int(c_id)] for c_id in obj_classes]

    return object_dict


if __name__ == '__main__':

    # Capture command line arguments
    if run_from_command_line:
        args = vars(parser.parse_args())
    else:
        args = args

    model_weights = args['weights']
    image_size = args['size']
    segment_dictionary_file = args['segment_dictionary']
    output_path = args['output_path']
    input_images = args['input_images']

    # Load segment dictionary
    segment_dictionary = load_segment_dict(segment_dictionary_file)

    # Check images directory
    num_images = len(glob.glob(os.path.join(input_images, '*.png')))
    if num_images == 0:
        raise Exception('[ERROR] No images found in images directory.')

    ## Load model with custom weights
    #print('[INFO] Loading YOLOv5 model with custom weights.')
    #model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_weights)

    # Load model with standard weights
    if yolo_model == 'v5':
        print('[INFO] Loading YOLOv5 model with standard weights.')
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)   

    if yolo_model == 'v8':
        print('[INFO] Loading YOLOv8m model with standard weights.')
        model = YOLO("yolov8m.pt")
    model_names = model.names

    # Set up intermediate txt file
    logger_path = os.path.join(output_path, 'detections_temp.txt')
    logger = AppendLogger(logger_path)

    # Identify device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Verify image neighborhood matches segment dictionary neighborhood
    segment_neighborhood = \
        segment_dictionary_file.split(os.path.sep)[-1].split('.')[0].split('_')[-1]
    image_neighborhood = input_images.split(os.path.sep)[-1].split('_')[0]
    image_neighborhood = segment_neighborhood  # added by Colton
    
    if segment_neighborhood != image_neighborhood:
        raise Exception('[ERROR] Image neighborhood should match segment neighborhood.')

    # Check past progress
    if not os.path.exists(logger_path):
        if not os.path.exists(output_path):
            print('[INFO] Creating output directories: {}'.format(output_path))
            os.makedirs(output_path)
        key_start = 0
    else:
        with open(logger_path, 'r') as file:
            processed_segments = file.readlines()
        processed_segments = [
            segment.split('|')[0] for segment in processed_segments]
        key_start = len(set(processed_segments)) - 1

    # Inference on each segment and image
    print('[INFO] Generating {} segment vectors for {}'.format(
        len(segment_dictionary) - key_start, segment_neighborhood))
    for key in tqdm(range(key_start, len(segment_dictionary))):
        segment = segment_dictionary[str(key)]

        # Hash segment ID
        segment_id = json.loads(segment['segment_id'])
        segment_id = '{}-{}'.format(segment_id[0], segment_id[1])

        # Get segment images
        images = glob.glob(
            os.path.join(input_images, 'img_{}_*.png'.format(segment_id)))
        image_paths = images.copy()

        # Skip the segment if it has no associated images (identify missing
        # by adding a row of Nones). A segment will usually have no associated
        # images if it had an unavailable heading for it's first node or if
        # it had no associated coordinates.
        if len(image_paths) == 0:
            logger.write('{}|{}|{}|{}|{}|{}|{}|{}|{}|{}'.format(
                segment_id, None, None, None, None, None, None, None, None, None))
            

        results_list, image_paths_list = [], []
        if len(images) != 0: # colton added to deal with deletions of images
            try:
                if yolo_model == 'v5':
                    results = model(images, size=image_size)
                if yolo_model == 'v8':
                    results = model(images, task='detect', mode='predict', imgsz=image_size, )
                results_list.append(results)
                image_paths_list.append(image_paths)
            except RuntimeError:
                # Handle out of memory errors
                image_path_groups = np.array_split(image_paths, 4)
                for image_path_group in image_path_groups:
                    image_path_group = image_path_group.tolist()
                    images_group = image_path_group.copy()
                    results_group = model(images_group, size=image_size)
                    results_list.append(results_group)
                    image_paths_list.append(image_path_group)

            # Add to segment vector DataFrame
            # Loop over each image in the segment
            segment_result_counter = 0
            for results_batch, image_paths_batch in zip(results_list, image_paths_list):
                # Loop over each image
                for i in range(len(results_batch)):
                    if yolo_model == 'v5':
                        # Get objects and image ID
                        img_objects = get_objects(
                            results_batch.xyxy[i], model_names, image_paths_batch[i],
                            segment_id, device)
                    if yolo_model == 'v8':
                        # Get objects and image ID
                        img_objects = get_objects(
                            results_batch[i].boxes.data, model.names, image_paths_batch[i],
                            segment_id, device)

                    # Loop over each object instance in the image and write to logger
                    for j in range(len(img_objects['segment_id'])):
                        logger.write('{}|{}|{}|{}|{}|{}|{}|{}|{}|{}'.format(
                            img_objects['segment_id'][j],  # Segment ID
                            img_objects['img_id'][j],  # Image ID
                            j,  # Object instance ID
                            round(img_objects['confidence'][j], 4),  # Confidence
                            round(img_objects['bbox_size'][j], 2),  # Bounding box size
                            round(img_objects['x1'][j], 4),
                            round(img_objects['y1'][j], 4),
                            round(img_objects['x2'][j], 4),
                            round(img_objects['y2'][j], 4),
                            # Class, but replace spaces with underscores
                            img_objects['class'][j].replace(' ', '_')
                        ))
                        segment_result_counter += 1

            if segment_result_counter == 0:
                logger.write('{}|{}|{}|{}|{}|{}|{}|{}|{}|{}'.format(
                    segment_id, None, None, None, None, None,None, None, None, None))

    # Check number of processed object vectors and save to DataFrame
    with open(logger_path, 'r') as file:
        processed_object_vectors = file.readlines()
    processed_segments = [
        segment.split('|')[0] for segment in processed_object_vectors]
    number_of_processed_segments = len(set(processed_segments))

    # Export to pd.DataFrame if all segments have been processed
    if number_of_processed_segments == len(segment_dictionary):
        df_columns = {'segment_id': [], 'img_id': [], 'object_id': [],
                      'confidence': [], 'bbox_size': [], 'x1' : [], 'y1':[], 'x2':[], 'y2':[], 'class': []}
        object_vectors = pd.DataFrame(df_columns)

        # Loop over the object instances
        for object_instance in tqdm(processed_object_vectors):

            if 'None' not in object_instance:

                # unpack object_instance
                segment_id, img_id, object_id, confidence, \
                    bbox_size, x1, y1, x2, y2 , class_id= object_instance.split('|')


                # Old code
                #segment_id, img_id, object_id, confidence, bbox_size, class_id = \
                    #object_instance.rstrip().split(' ')

                object_instance_dict = {'segment_id': segment_id,
                                        'img_id': img_id,
                                        'object_id': object_id,
                                        'confidence': confidence,
                                        'bbox_size': bbox_size,
                                        'x1': x1,
                                        'y1': y1,
                                        'x2': x2,
                                        'y2': y2,
                                        'class': class_id.replace('\n', '') }
                
                # concat to object_vectors
                object_vectors = pd.concat(
                    [object_vectors, pd.DataFrame(object_instance_dict, index=[0])],
                    ignore_index=True)



        # Export to CSV
        object_vectors.to_csv(
            os.path.join(output_path, 'detections.csv'), index=False)
    else:
        raise Exception('[ERROR] Incomplete street segment temporary file.')
