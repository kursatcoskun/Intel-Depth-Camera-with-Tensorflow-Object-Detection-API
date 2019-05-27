import pyrealsense2 as rs
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from utils import label_map_util
from utils import visualization_utils as vis_util
from utils.detection_module import detectionClass
import requests
import datetime
x=0
y=0
detectioncs = detectionClass()
url_constant = "http://localhost:50104/"

def postMethod(url,jsonData):
    r = requests.post(url_constant + url, json=jsonData)
    print(url_constant+url)
    return r.status_code


# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 3

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


pc = rs.pointcloud()
points = rs.points()
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# This is the minimal recommended resolution for D435
config.enable_stream(rs.stream.depth,  848, 480, rs.format.z16, 90)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

# Start streaming
# Getting the depth sensor's depth scale
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Create an align object
align_to = rs.stream.depth

align = rs.align(align_to)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("\n\t depth_scale: " + str(depth_scale))
# Getting the depth sensor's depth scale
pipeline.stop()

# Start streaming
pipeline.start(config)
try:
    while True:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        # Create an align object
        align_to = rs.stream.depth
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        colorImage=np.array(color_frame.get_data())
        #depthImage=np.array(depth_frame.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))
        # Show images


        frame_expanded = np.expand_dims(colorImage, axis=0)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            colorImage,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=10,
            min_score_thresh=0.80)

        aligned_frames = align.process(frames)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
        color_to_depth_extrin = color_frame.profile.get_extrinsics_to(depth_frame.profile)
        print("\n Depth intrinsics: " + str(depth_intrin))
        print("\n Color intrinsics: " + str(color_intrin))
        print("\n Depth to color extrinsics: " + str(depth_to_color_extrin))

        # Tell pointcloud object to map to this color frame
        pc.map_to(color_frame)

        # Generate the pointcloud and texture mappings
        points = pc.calculate(depth_frame)

        # Save pointcloud as ply
        #print("Saving as ply...")
        #points.export_to_ply("C:/Users/Student/Downloads/point_cloud.ply", color_frame)
        #print("Done")

        # Validate that both frames are valid
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        width= 848
        height = 490
        for i, box in enumerate(np.squeeze(boxes)):
            if (np.squeeze(scores)[i] > 0.5):
                print("ymin={}, xmin={}, ymax={}, xmax{}".format(box[0] * height, box[1] * width, box[2] * height,
                                                                 box[3] * width))
                (left, right, top, bottom) = (box[1] * width, box[3] * width, box[0] * height, box[2] * height)
                x = int(((top - bottom) / 2) + bottom)
                y = int(((right - left) / 2) + left)
                depth_pixel = [x, y]  # Random pixel
                depth_value = depth_image[x][y] * depth_scale

        #print("X: "+str(x)+" Y: "+ str(y))
        #depth_pixel = [200,200]  # Random pixel
        #depth_value = depth_image[200][200] * depth_scale
                print("\n\t depth_pixel@" + str(depth_pixel) + " value: " + str(depth_value) + " meter")
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_value)


                for index, value in enumerate(classes[0]):
                    if (scores[0, index] > 0.8):
                        print(int((category_index.get(value).get('id'))))
                        detectionJson = {"detectedObjectId": int((category_index.get(value).get('id'))), "detectionTime": datetime.datetime.now().isoformat(), "deviceId": "2",
                                 "xCoordinate": depth_point[0], "yCoordinate": depth_point[1],
                                 "zCoordinate": depth_point[2]}
                        print(postMethod("/api/Detection/AddDetection", detectionJson))
        #if  for index, value in enumerate(classes[0]) if scores[0, index] > 0.8][0]:


                print("\n\t 3D depth_point: " + str(depth_point))
                print(classes)

                color_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)
                print("\n\t 3D color_point: " + str(color_point))
                print(("Scores"))
                print([int((category_index.get(value).get('id'))) for index, value in enumerate(classes[0]) if scores[0, index] > 0.8])

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', colorImage)
        cv2.imshow('Depth Cam',depth_colormap)
        cv2.waitKey(1)
finally:

    # Stop streaming
    pipeline.stop()


