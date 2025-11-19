#!/usr/bin/env python3
import rospy
from duckietown_msgs.msg import WheelsCmdStamped
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

MODEL_PATH = "packages/object_detection/yolov8n.pt"

class TrafficLightController:
    def __call__(self, *args, **kwds):
        rospy.loginfo("Initializing traffic light controller")
        self.model = YOLO(MODEL_PATH)
        self.bridge = CvBridge() # Converts ROS image to OpenCV image

        # Publisher to wheels
        self.pub = rospy.Publisher("/wheels_driver_node/wheels_cmd", 
                                   WheelsCmdStamped, 
                                   queue_size=1)
        
        # Subscriber to camera
        rospy.Subscriber("/camera_node/image/compressed", 
                                    CompressedImage, 
                                    self.process_image, 
                                    queue_size=1)
        
        self.state = "DRIVE"
        self.last_detected_color = None
        rospy.loginfo("Traffic light controller initialized")

    def callback(self, msg):
        frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

        # Get detection results and parse
        results = self.model(frame, imgsz=640, conf=0.4)
        traffic_lights = self.parse_detections(results)

        if traffic_lights:
            color = traffic_lights[0]
            self.last_detected_color = color
            self.update_state(color)
        else:
            # If nothing, continue driving
            self.update_state("green")

        self.apply_wheel_command()

    def parse_detections(self, results):
        # Go through detections and return detected traffic light colors
        colors_detected = []

        for r in results:
            for b in r.boxes:
                cls = int(b.cls[0])

                if cls == 0:
                    colors_detected.append("red")
                elif cls == 1:
                    colors_detected.append("green")
                elif cls == 2:
                    colors_detected.append("yellow")

        return colors_detected
    
    def update_state(self, color):
        if color == "red":
            self.state = "STOP"
        elif color == "green":
            self.state = "DRIVE"
        # Add yellow in the future?

    def apply_wheel_command(self):
        cmd = WheelsCmdStamped()
        cmd.header.stamp = rospy.Time.now()

        if self.state == "STOP":
            cmd.vel_left = 0.0
            cmd.vel_right = 0.0

        elif self.state == "DRIVE":
            cmd.vel_left = 0.30
            cmd.vel_right = 0.30

        self.pub_wheels.publish(cmd)
