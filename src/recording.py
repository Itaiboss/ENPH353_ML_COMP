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

import tensorflow as tf


interpreter = tf.lite.Interpreter(model_path='quantized_model3.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

def check_if_space(letter):
    # Count non-white pixels
    non_white_pixels = np.sum(letter < 255)

    # Calculate percentage of non-white pixels
    total_pixels = np.prod(letter.shape)
    non_white_percentage = (non_white_pixels / total_pixels)

    if non_white_percentage < 0.1:
        return True
    else:
        return False

# Go from char to one hot
def convert_to_one_hot(Y):
    Y = np.array([ord(char) - ord('A') if char.isalpha() else (int(char)+26) for char in Y])
    Y = np.eye(36)[Y.reshape(-1)]
    return Y

# Go from one hot / probabilities to a character
def convert_from_one_hot(one_hot_labels):
    # Convert the one-hot encoded labels back to their original representation
    # Assuming the labels were one-hot encoded using the provided convert_to_one_hot function
    decoded_labels = np.argmax(one_hot_labels, axis=1)

    # Convert the numerical representation back to the original characters or numbers
    decoded_labels = [chr(index + ord('A')) if index < 26 else str(index - 26) for index in decoded_labels]

    return decoded_labels
  
def clue_detect(clue_board):

    # detect the key
    key_let1 = clue_board[40:115, 250:295]
    key_let2 = clue_board[40:115, 295:340]
    key_let3 = clue_board[40:115, 340:385]
    key_let4 = clue_board[40:115, 385:430]
    key_let5 = clue_board[40:115, 430:475]
    key_let6 = clue_board[40:115, 475:520]

    letters = [key_let1, key_let2, key_let3, key_let4, key_let5, key_let6]

    key_array = []

    for letter in letters:
        #letter = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
        #print(letter.shape)
        #cv2_imshow(letter)

        # check if white space
        if (check_if_space(letter)):
            key_array.append(' ')

        else:

            # regular version

            #input_letter = np.expand_dims(letter, axis=-1)
            #input_letter = np.expand_dims(input_letter, axis=0)
            #pred_letter = convert_from_one_hot(conv_model.predict(input_letter))

            # quantized version

            input_letter = np.expand_dims(letter, axis=0)
            input_letter = np.expand_dims(input_letter, axis=-1)
            input_letter = input_letter.astype(np.float32)

            #
            interpreter.set_tensor(input_details["index"], input_letter)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details["index"])

            pred_letter = convert_from_one_hot(output)[0]
            key_array.append(pred_letter)

    key = ''.join(key_array)
    print(f"Key = {key}")

    clue_let1 = clue_board[260:335, 30:75]
    clue_let2 = clue_board[260:335, 75:120]
    clue_let3 = clue_board[260:335, 120:165]
    clue_let4 = clue_board[260:335, 165:210]
    clue_let5 = clue_board[260:335, 210:255]
    clue_let6 = clue_board[260:335, 255:300]
    clue_let7 = clue_board[260:335, 300:345]
    clue_let8 = clue_board[260:335, 345:390]
    clue_let9 = clue_board[260:335, 390:435]
    clue_let10 = clue_board[260:335, 435:480]
    clue_let11 = clue_board[260:335, 480:525]
    clue_let12 = clue_board[260:335, 525:570]

    letters = [clue_let1, clue_let2, clue_let3, clue_let4, clue_let5, clue_let6,
                clue_let7, clue_let8, clue_let9, clue_let10, clue_let11, clue_let12]

    value_array = []

    for letter in letters:
        #letter = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
        #cv2_imshow(letter)

        # check if white space
        if (check_if_space(letter)):
            value_array.append(' ')
        else:
            # regular version

            #input_letter = np.expand_dims(letter, axis=-1)
            #input_letter = np.expand_dims(input_letter, axis=0)
            #pred_letter = convert_from_one_hot(conv_model.predict(input_letter))

            # quantized version

            input_letter = np.expand_dims(letter, axis=0)
            input_letter = np.expand_dims(input_letter, axis=-1)
            input_letter = input_letter.astype(np.float32)

            #
            interpreter.set_tensor(input_details["index"], input_letter)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details["index"])

            pred_letter = convert_from_one_hot(output)[0]
            value_array.append(pred_letter)

    value = ''.join(value_array)
    print(f"Value = {value}")

    return key, value

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
        im_cut = cv_image[360:720,0:1280]
        im_grey = cv2.cvtColor(im_cut, cv2.COLOR_BGR2GRAY)
        hsv_cut = cv2.cvtColor(im_cut, cv2.COLOR_BGR2HSV)
        #print(hsv_cut[230][600])
        lower_road_hsv = np.array([0,0,79])
        upper_road_hsv= np.array([6,6,90])
        mask_road = cv2.inRange(hsv_cut, lower_road_hsv, upper_road_hsv)

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
        filtered_cnts = [contour for contour in cnts if cv2.contourArea(contour) > 1000]
        mask_clue = cv2.bitwise_and(mask_blue,mask_white)
        num_white_pixels = cv2.countNonZero(mask_blue)
        cnts, _ = cv2.findContours(mask_clue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_cnts = [contour for contour in cnts if cv2.contourArea(contour) > 1000]
        #print(num_white_pixels)
        if num_white_pixels > 3000 and filtered_cnts:
            for c in filtered_cnts:
                epsilon = 0.08 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)

                #cv2.drawContours(cv_image, [approx], -1, (0, 255, 0), 2)
            # print(approx)
        #     dst = cv2.cornerHarris(mask_clue,2,3,0.04)
        #     ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
        #     dst = np.uint8(dst)
        #     ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        #     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        #     corners = cv2.cornerSubPix(mask_clue,np.float32(centroids),(5,5),(-1,-1),criteria)
            
        #     #Now draw them
        #     src =corners[1:]
        #     #print(src)
            

        #     # Rearrange the corners based on the assigned indicesght
        #     centroid = corners[0]
        #     def sort_key(point):
        #         angle = np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
        #         return (angle + 2 * np.pi) % (2 * np.pi)
        #     # Sort the source points based on their relative positions to match the destination points format
        #     sorted_src = sorted(src, key=sort_key)
        #     sorted_src = np.array(sorted_src)

        #     # Reorder 'src' points to match the 'dest' format
            # approx
            src = np.array([approx[0], approx[3], approx[1], approx[2]], dtype=np.float32)
        #     print(src)

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
            # #45 equihist

            

            # #invert to get black text on white background
            result = cv2.bitwise_not(mask2)

            cv2.imshow("clue board", result)
            cv2.waitKey(1)

            clue_detect(result)

            
            # for corner in src:
            #     x, y = int(corner[0]), int(corner[1])
            #     cv2.circle(cv_image,(x,y), 2, (0,255,0), -1)  # -1 signifies filled circle
        
        cv2.imshow("gray", mask_road)
        #cv2.imshow("cut", im_cut)
        cv2.imshow("hsv", hsv_cut)
        # cv2.imshow("blue", mask_blue)
        # cv2.imshow("out", result)
        # cv2.imshow("white", mask_white)
        # cv2.imshow("image", mask_clue)
        # cv2.imshow("done", result)

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

