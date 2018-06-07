# export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
# Imports
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from PIL import Image
from object_detection.utils import ops as utils_ops

import cv2
import json
import xml.etree.ElementTree as ET

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# Object detection imports
# Here are the imports from the object detection module.

from utils import label_map_util
from utils import visualization_utils as vis_util

#Read it from file...
tree = ET.parse("origin.xml")
root = tree.getroot()

rm = root.find("rotationMatrixInv")
counter = 1
rotationMatrixInv = np.empty((0,3))
row = np.empty((0,3))
for data in rm.findall("data"):
    if (counter % 3) == 0:
        row = np.append(row,[float(data.text) ])
        row = np.reshape(row, (-1, 3))
        rotationMatrixInv = np.append( rotationMatrixInv, row, axis = 0 )
        row = np.empty((0,3))
    else:
        row = np.append(row,[ float(data.text) ])
    counter+=1

# cameraMatrix_inv = 
cmi = root.find("cameraMatrixInv")
counter = 1
cameraMatrixInv = np.empty((0,3))
row = np.empty((0,3))
for data in cmi.findall("data"):
    if (counter % 3) == 0:
        row = np.append(row,[float(data.text) ])
        row = np.reshape(row, (-1, 3))
        cameraMatrixInv = np.append( cameraMatrixInv, row, axis = 0 )
        row = np.empty((0,3))
    else:
        row = np.append(row,[ float(data.text) ])
    counter+=1

# scalar
sc = root.find("scalar")
scalar = float(sc.find("data").text )

# tvec
tv = root.find("tVec")
tVec = np.empty((0,1))
for data in tv.findall("data"):
        tVec = np.append(tVec,[ float(data.text) ] )
tVec = np.reshape(tVec, (-1, tVec.size))
tVec = np.swapaxes(tVec,0,1)

def px_to_mm(x,y):
  print("here")
  px = np.array([[x],[y],[1]])
  mm = np.dot(rotationMatrixInv,(np.dot( cameraMatrixInv, scalar*px ) - tVec))
  return mm[0][0],mm[1][0]

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)

      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def detect_object_from_images(detection_graph,category_index, min_score_thresh):
  print("ready to read images")
  while True:
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = cv2.imread("io/input.jpg")

    if image_np is None:
      continue
    print("Reading image")

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    #Takes a while next line.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    #print(output_dict)
    #/object_detection/utils/visualization_utils.py
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=min_score_thresh
        )

    #Create json file
    data = []
    im_height,im_width,_ = image_np.shape
    for detection in range(0,output_dict['num_detections']):
      if float(output_dict['detection_scores'][detection]) < min_score_thresh:
        break
      coordinates = output_dict['detection_boxes'][detection].tolist()
      (left, right, bottom, top) = (coordinates[1] * im_width, coordinates[3] * im_width, coordinates[2] * im_height, coordinates[0] * im_height)
      xmin, ymax = px_to_mm(left,top)
      xmax, ymin = px_to_mm(right,bottom)

      data.append({
        'bndbox': {
                    'xmin': xmin,
                    'xmax': xmax,
                    'ymin': ymin,
                    'ymax': ymax,
                  },
        'name': category_index[output_dict['detection_classes'][detection]]['name'],
        'id': int(output_dict['detection_classes'][detection]),
        'score': float(output_dict['detection_scores'][detection]) #here
      })
    with open('io/output.json', 'w') as outfile: 
      json.dump(data, outfile)

    #delete input file
    os.remove("io/input.jpg")

    #create output file
    cv2.imwrite('io/output.jpg',image_np)
    print("done")

def prepare_model(PATH_TO_MODEL,PATH_TO_LABELS, min_score_thresh):

  # Load a (frozen) Tensorflow model into memory.
  print("loading...")
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  # Loading label map
  with open( PATH_TO_LABELS, 'r' ) as f:
    data_pbtxt = f.read()

  NUM_CLASSES = data_pbtxt.count('item')

  # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)
  detect_object_from_images(detection_graph,category_index, min_score_thresh)

def main():
  PATH_TO_MODEL = sys.argv[1]
  PATH_TO_LABELS = sys.argv[2]

  min_score_thresh = float( sys.argv[3] ) / 100
  if(min_score_thresh > 1.0 or min_score_thresh < 0.0):
    min_score_thresh = .75
  print(min_score_thresh)
  prepare_model(PATH_TO_MODEL,PATH_TO_LABELS, min_score_thresh)

if __name__ == "__main__":
    main()
