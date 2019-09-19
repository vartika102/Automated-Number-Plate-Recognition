#import time
import cv2
#import mss
import numpy as np
import os
#import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
#from distutils.version import StrictVersion
#from collections import defaultdict
#from io import StringIO
#import matplotlib.pyplot as plt


# title of our window
title = "CAR"
# set start time to current time
#start_time = time.time()
# displays the frame rate every 2 second
#display_time = 2
# Set primarry FPS to 0
#fps = 0
# Load mss library as sct
#sct = mss.mss()
# Set monitor size to capture to MSS
monitor = {"top": 40, "left": 0, "width": 800, "height": 640}



# ## Env setup
#from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util


# # Model preparation 
PATH_TO_FROZEN_GRAPH = 'D:/KPIT ANPR/test/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'D:/KPIT ANPR/test/labelmap.pbtxt'
NUM_CLASSES = 1


# ## Load a (frozen) Tensorflow model into memory.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# # Detection
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    #while True:
      # Get raw pixels from the screen, save it to a Numpy array
      image_np = cv2.imread('./CAR/! (4).jpg')
      #cv2.imshow('image_np',image_np)
      image_np = cv2.resize(image_np,(800,640))
      # To get real color we do this:
      image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Visualization of the results of a detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=3)
      
      total_final=cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
      # Show image with detection
      cv2.imshow(title,total_final )
      
    
    
              
      
      ymin = int((boxes[0][0][0]*640))
      xmin = int((boxes[0][0][1]*800))
      ymax = int((boxes[0][0][2]*640))
      xmax = int((boxes[0][0][3]*800))
      #print(scores[0])

      Result = np.array(total_final[ymin:ymax,xmin:xmax])
      #in_range = cv2.inRange(total_final,(0,127,0),(0,255,127))
      cv2.imshow('in_range',total_final[ymin:ymax,xmin:xmax])
      
      
      # Bellow we calculate our FPS
      #fps+=1
      #TIME = time.time() - start_time
      #if (TIME) >= display_time :
        #print("FPS: ", fps / (TIME))
       # fps = 0
        #start_time = time.time()
      # Press "q" to quit
      #print(num_detections)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
       # break