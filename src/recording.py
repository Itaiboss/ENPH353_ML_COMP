#! /usr/bin/env python3
import rospy          
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import cv2
import numpy as np


#initialize publishers
rospy.init_node('imitation_node')
move_pub = rospy.Publisher('/R1/cmd_vel', Twist, 
  queue_size=10)    
br = CvBridge()
rospy.sleep(1)

class Imitate:
    def __init__(self):
        self.cur_state = None
        self.prev_state = None

    def update_state(self, new_state):
        self.prev_state = self.cur_state
        self.cur_state = new_state

    def get_current_state(self):
        return self.cur_state

    def get_previous_state(self):
        return self.prev_state

    def camera_callback(self, data):
        try:
            cv_image = br.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        img_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        bi = cv2.bilateralFilter(cv_image, 5, 75, 75)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        uh = 130
        us = 255
        uv = 255
        lh = 110
        ls = 50
        lv = 50
        lower_hsv = np.array([lh,ls,lv])
        upper_hsv = np.array([uh,us,uv])
        lower_white = np.array([0,0,90])
        upper_white = np.array([10,10,110])

        # Threshold the HSV image to get only blue colors
        mask_blue = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask_white = cv2.bitwise_not(mask_blue)
        # threshold = 180
        # _, binary = cv2.threshold(img_gray,threshold,255,cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.fillPoly(mask_blue, cnts, (255,255,255))
        # filtered_cnts = [contour for contour in cnts if cv2.contourArea(contour) > 1000]
        mask_clue = cv2.bitwise_and(mask_blue,mask_white)
        filtered_cnts = [contour for contour in cnts if cv2.contourArea(contour) > 1000]
        if filtered_cnts:
            dst = cv2.cornerHarris(mask_clue,2,3,0.04)
            ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
            dst = np.uint8(dst)
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            corners = cv2.cornerSubPix(mask_clue,np.float32(centroids),(5,5),(-1,-1),criteria)
            
            #Now draw them
            src =corners[1:]
            #print(src)
            

            # Rearrange the corners based on the assigned indicesght
            centroid = corners[0]
            def sort_key(point):
                angle = np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
                return (angle + 2 * np.pi) % (2 * np.pi)
            # Sort the source points based on their relative positions to match the destination points format
            sorted_src = sorted(src, key=sort_key)
            sorted_src = np.array(sorted_src)

            # Reorder 'src' points to match the 'dest' format
            src = np.array([sorted_src[2], sorted_src[3], sorted_src[1], sorted_src[0]], dtype=np.float32)
            print(src)

            width = 600
            height= 400
            dest = np.float32([[0, 0],
                        [width, 0],
                        [0, height],
                        [width , height]])

            M = cv2.getPerspectiveTransform(src,dest)
            clue = cv2.warpPerspective(cv_image,M,(width, height),flags=cv2.INTER_LINEAR)

            gray_clue = cv2.cvtColor(clue, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray_clue = clahe.apply(gray_clue)
            #gray_clue = cv2.equalizeHist(gray_clue)
            # perform threshold
            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpen = cv2.filter2D(gray_clue, -1, sharpen_kernel)

            # # remove noise / close gaps
            # kernel =  np.ones((5,5),np.uint8)
            # noise = cv2.morphologyEx(sharpen, cv2.MORPH_CLOSE, kernel)

            # # dilate result to make characters more solid
            # kernel2 =  np.ones((3,3),np.uint8)
            # dilate = cv2.dilate(noise,kernel2,iterations = 1)

            retr, mask2 = cv2.threshold(gray_clue, 100, 255, cv2.THRESH_BINARY_INV)
            #45 equihist

            

            #invert to get black text on white background
            result = cv2.bitwise_not(mask2)
            
            for corner in src:
                x, y = int(corner[0]), int(corner[1])
                cv2.circle(cv_image,(x,y), 2, (0,255,0), -1)  # -1 signifies filled circle
        
        
        cv2.imshow("blue", mask_blue)
        cv2.imshow("white", mask_white)
        cv2.imshow("image", mask_clue)
        cv2.imshow("done", result)

        cv2.waitKey(1) 
        # save sate and image


    def move_callback(self, data):
        linear_x = data.linear.x
        angular_z = data.angular.z
        self.cur_state = f"{linear_x},{angular_z}"

trial = Imitate()

def controller():
    rospy.Subscriber('/R1/pi_camera/image_raw',Image,trial.camera_callback)
    rospy.Subscriber('/clock',Clock)
    rospy.Subscriber('/R1/cmd_vel',Twist,trial.move_callback)
    rospy.sleep(1)

while not rospy.is_shutdown():
    controller()
    rospy.sleep(1)
    rospy.spin()    



#!/usr/bin/env python3


