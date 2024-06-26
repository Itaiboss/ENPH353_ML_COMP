#! /usr/bin/env python3

## @package controller
#  This script is used to control the robot in the ROS environment.

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

from collections import Counter

# imports for data collection
import os
import csv
import re

## @var csv_file_path
#  The path to the CSV file for data collection.
csv_file_path = '/home/fizzer/ros_ws/src/2023_competition/enph353/enph353_gazebo/scripts/plates.csv'

## @var output_path
#  The path where new plates will be stored.
output_path = '/home/fizzer/new_plates/'

# Load the trained neural networks

## @var interpreter_road
#  The TensorFlow Lite interpreter for the road model.
interpreter_road = tf.lite.Interpreter(model_path='quantized_model_road3.tflite')
interpreter_road.allocate_tensors()
input_details_road = interpreter_road.get_input_details()[0]
output_details_road = interpreter_road.get_output_details()[0]
print("Loaded Road")

## @var interpreter_grass
#  The TensorFlow Lite interpreter for the grass model.
interpreter_grass = tf.lite.Interpreter(model_path='quantized_model_grass9.tflite')
interpreter_grass.allocate_tensors()
input_details_grass = interpreter_grass.get_input_details()[0]
output_details_grass = interpreter_grass.get_output_details()[0]
print("Loaded Grass")

## @var interpreter_mountain
#  The TensorFlow Lite interpreter for the mountain model.
interpreter_mountain = tf.lite.Interpreter(model_path='quantized_model_mountain7.tflite')
interpreter_mountain.allocate_tensors()
input_details_mountain = interpreter_mountain.get_input_details()[0]
output_details_mountain = interpreter_mountain.get_output_details()[0]
print("Loaded Mountain")

## @var interpreter_yoda
#  The TensorFlow Lite interpreter for the Yoda model.
interpreter_yoda = tf.lite.Interpreter(model_path='quantized_model_yoda5.tflite')
interpreter_yoda.allocate_tensors()
input_details_yoda = interpreter_yoda.get_input_details()[0]
output_details_yoda = interpreter_yoda.get_output_details()[0]
print("Loaded Yoda")

## @var interpreter_clues
#  The TensorFlow Lite interpreter for the clues model.
interpreter_clues = tf.lite.Interpreter(model_path='quantized_model_clues3.tflite')
interpreter_clues.allocate_tensors()
input_details_clues = interpreter_clues.get_input_details()[0]
output_details_clues = interpreter_clues.get_output_details()[0]
print("Loaded Clue Interpreter")

#path = os.path.join(output_path)
#print(path)
# os.mkdir(path, mode = 0o777)
#os.chdir(path)

## @fn run_model
#  @brief This function runs the model on the input image.
#  @param img The input image.
#  @param interpreter The TensorFlow Lite interpreter.
#  @param input_details The input details of the interpreter.
#  @param output_details The output details of the interpreter.
#  @param steer The steering value.
#  @return The output of the model.
def run_model(img, interpreter, input_details, output_details, steer):
    img_aug = img.reshape((1, 90, 160, 1)).astype(input_details["dtype"])
    interpreter.set_tensor(input_details_road["index"], img_aug)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    #print(output)
    output = denormalize_value(output, -steer, steer)
    return output

## @fn denormalize_value
#  @brief This function denormalizes a value from the range [-1, 1] to [min_val, max_val].
#  @param val The value to denormalize.
#  @param min_val The minimum value of the new range.
#  @param max_val The maximum value of the new range.
#  @return The denormalized value.
def denormalize_value(normalized_value, min_val, max_val):
    return (normalized_value * (max_val - min_val)) + min_val

# Denormalize a value from the range [-1, 1] to [min_val, max_val]
## @fn denormalize_value
#  @brief This function denormalizes a value from the range [-1, 1] to [min_val, max_val].
#  @param val The value to denormalize.
#  @param min_val The minimum value of the new range.
#  @param max_val The maximum value of the new range.
#  @return The denormalized value.
possible_keys = {"SIZE  " : 1, "VICTIM" : 2, "CRIME " : 3, "TIME  " : 4, "PLACE " : 5, "MOTIVE" : 6, "WEAPON" : 7, "BANDIT" : 8}

## @fn check_if_space
#  @brief This function checks if a given letter is a space.
#  @param letter The letter to check.
#  @return True if the letter is a space, False otherwise.
def check_if_space(letter):
    # Count non-white pixels
    non_white_pixels = np.sum(letter < 255)

    # Calculate percentage of non-white pixels
    total_pixels = np.prod(letter.shape)
    non_white_percentage = (non_white_pixels / total_pixels)

    if non_white_percentage < 0.13:
        return True
    else:
        return False

## @fn convert_to_one_hot
#  @brief This function converts a given label to one-hot encoding.
#  @param Y The label to convert.
#  @return The one-hot encoded label.
def convert_to_one_hot(Y):
    Y = np.array([ord(char) - ord('A') if char.isalpha() else (int(char)+26) for char in Y])
    Y = np.eye(36)[Y.reshape(-1)]
    return Y

## @fn convert_from_one_hot
#  @brief This function converts a given one-hot encoded label back to its original representation.
#  @param one_hot_labels The one-hot encoded labels to convert.
#  @return The original representation of the labels.
def convert_from_one_hot(one_hot_labels):
    # Convert the one-hot encoded labels back to their original representation
    # Assuming the labels were one-hot encoded using the provided convert_to_one_hot function
    decoded_labels = np.argmax(one_hot_labels, axis=1)

    # Convert the numerical representation back to the original characters or numbers
    decoded_labels = [chr(index + ord('A')) if index < 26 else str(index - 26) for index in decoded_labels]

    return decoded_labels

## @fn clue_detect
#  @brief This function detects the key and value of a clue.
#  @param clue_board The clue board to detect from.
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
            interpreter_clues.set_tensor(input_details_clues["index"], input_letter)
            interpreter_clues.invoke()
            output = interpreter_clues.get_tensor(output_details_clues["index"])
            pred_letter = convert_from_one_hot(output)[0]
            if pred_letter == 'Q':
                print(output)
                print(output.shape)
                if output[0][14] > 1e-4:
                    pred_letter = 'O'
                    print("SWAPPED")
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
            
            # quantized version

            input_letter = np.expand_dims(letter, axis=0)
            input_letter = np.expand_dims(input_letter, axis=-1)
            input_letter = input_letter.astype(np.float32)

            interpreter_clues.set_tensor(input_details_clues["index"], input_letter)
            interpreter_clues.invoke()
            output = interpreter_clues.get_tensor(output_details_clues["index"])

            pred_letter = convert_from_one_hot(output)[0]
            if pred_letter == 'Q':
                print(output)
                print(output.shape)
                if output[0][14] > 1e-4:
                    pred_letter = 'O'
                    print("SWAPPED")
            value_array.append(pred_letter)

    value = ''.join(value_array)
    print(f"Value = {value}")

    return key, value


# Clue message structure 
# team ID: max 8 characters (no spaces)
# team password: max 8 characters (no spaces)
# clue location: int (-1 to 8); -1 and 0 are special cases - see below
# clue prediction: n characters (no spaces, all capitals)

## @fn pub_clue
#  @brief This function publishes a clue message.
#  @param id The team ID.
#  @param password The team password.
#  @param location The clue location.
#  @param prediction The clue prediction.
def pub_clue(id,password,location,prediciton):
    formatted_string = f"{id},{password},{location},{prediciton}"
    score_pub.publish(formatted_string)

## @fn pub_cmd_vel
#  @brief This function publishes a velocity command.
#  @param x The linear velocity.
#  @param z The angular velocity.
def pub_cmd_vel(x,z):
    move = Twist()
    move.angular.z = z
    move.linear.y = 0
    move.linear.x = x
    move_pub.publish(move)

## @class StateMachine
#  @brief This class represents a state machine.
class StateMachine:
    ## @fn __init__
    #  @brief This function initializes the state machine.
    def __init__(self):
        self.current_state = "ROAD"
        print(f'Starting state:{self.current_state}')
        self.drive_input = None
        self.pink_cooldown = False
        self.ped_xing = False
        self.state_data1 = None
        self.state_data2 = None
        self.cv_image = None
        self.yoda_wait = False
        self.aligned = False
        self.tunnel = False
        self.frame_counter = 0 
        self.clue_count = 0
        self.clue_cooldown = False
        self.mountain_boost = False
        self.bridge_boost = False
        self.started = False
        self.cur_clue = []
        self.clues=[]
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

        # Clue Detect Variables
        self.good_values = []

    # State Functions
        
    ## @fn road_state
    #  @brief This function defines the behavior of the robot in the road state.
    def road_state(self):
        if not self.started:
            pub_clue(id,password,0,"NA")
            rospy.Timer(rospy.Duration(120), self.end_run, oneshot=True)
            self.started = True
        else:
            z = run_model(self.drive_input,interpreter_road,input_details_road,output_details_road,2.8)#2.8
            pub_cmd_vel(.7,z)

    ## @fn grass_state
    #  @brief This function defines the behavior of the robot in the grass state.
    def grass_state(self):
        # if self.bridge_boost:
        #     z = run_model(self.drive_input,interpreter_grass,input_details_grass,output_details_grass,3)
        #     pub_cmd_vel(1,z)
        # else:
        z = run_model(self.drive_input,interpreter_grass,input_details_grass,output_details_grass,4) #4
        print(z)
        pub_cmd_vel(.65,z)

    ## @fn yoda_drive_state
    #  @brief This function defines the behavior of the robot in the Yoda drive state.
    def yoda_drive_state(self):
        z = run_model(self.drive_input,interpreter_yoda,input_details_yoda,output_details_yoda,3)
        if self.yoda_wait:
            pub_cmd_vel(1,z)
        elif self.tunnel:
            pub_cmd_vel(.5,z)
        else:
            pub_cmd_vel(1,z)

    ## @fn yoda_wait_state
    #  @brief This function defines the behavior of the robot in the Yoda wait state.
    def yoda_wait_state(self):
        pub_cmd_vel(0,0)
        filtered_cnts=[]
        if not self.aligned:
            self.align()
        else:
            if self.yoda_wait:
                # Check if yoda is in the frame
                fg_mask = self.bg_subtractor.apply(self.state_data1)
                # Apply additional morphological operations to clean the mask (optional)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Iterate through contours and filter based on size (area)
                filtered_cnts = [contour for contour in contours if cv2.contourArea(contour) > 300]
                if cv2.countNonZero(self.state_data1)>8 or filtered_cnts:
                    self.frame_counter = 0
                else:
                    self.frame_counter +=1 
                    print(self.frame_counter)
            else:
                self.pink_cooldown = True
                if (cv2.countNonZero(self.state_data1) > 900):
                    self.frame_counter = 0
                else:
                    self.frame_counter +=1 
                    print(self.frame_counter)
            # If Yoda is not in the frame, start the pink cooldown
            if self.frame_counter > 11:
                self.frame_counter = 0
                holder = self.current_state
                if not self.yoda_wait:
                    self.current_state = "YODA_DRIVE"
                    rospy.Timer(rospy.Duration(5), self.reset_pink_cooldown, oneshot=True)
                    rospy.Timer(rospy.Duration(4.35), self.set_yoda_wait, oneshot=True)
                else:
                    self.current_state = "YODA_DRIVE"
                    self.yoda_wait = 0
                    self.tunnel = True
                print(f'{holder} -------> {self.current_state}')
        
    
    ## @fn mountain_state
    #  @brief This function defines the behavior of the robot in the mountain state.
    def mountain_state(self):
        # if not self.aligned:
        #     self.align()
        # else:
        if self.mountain_boost:
            z = run_model(self.drive_input,interpreter_mountain,input_details_mountain,output_details_mountain,2.9)
            pub_cmd_vel(.65,z)
        else:
            z = run_model(self.drive_input,interpreter_mountain,input_details_mountain,output_details_mountain,2.7)
            pub_cmd_vel(.5,z) # mountain not sped up works consistently 

    ## @fn pedestrian_state
    #  @brief This function defines the behavior of the robot in the pedestrian state.
    def pedestrian_state(self):
        # Check if pedestrian is in the frame
        fg_mask = self.bg_subtractor.apply(self.state_data1)
        # Apply additional morphological operations to clean the mask (optional)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Iterate through contours and filter based on size (area)
        filtered_cnts = [contour for contour in contours if cv2.contourArea(contour) > 30]
        if filtered_cnts:
            self.frame_counter = 0
        else:
            self.frame_counter +=1 
            print(self.frame_counter)
        if self.frame_counter > 10:
            self.frame_counter = 0
            holder = self.current_state
            self.current_state = "ROAD"
            pub_cmd_vel(.1,0)
            print(f'{holder} -------> {self.current_state}')
             
    ## @fn vehicle_state
    #  @brief This function defines the behavior of the robot in the vehicle state.
    def vehicle_state(self):
        # Check if vehicle is in the frame
        fg_mask = self.bg_subtractor.apply(self.state_data1)
        # Apply additional morphological operations to clean the mask (optional)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Iterate through contours and filter based on size (area)
        filtered_cnts = [contour for contour in contours if cv2.contourArea(contour) > 100 ]
        if filtered_cnts:
            big_contour = max(filtered_cnts, key=cv2.contourArea)
            M = cv2.moments(big_contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            #print(f'x:{cx} y:{cy}')
            if (cx<600):
                self.frame_counter = 0
                print("STOP")
            else: 
                self.frame_counter +=1 
                print("GOO")
        if self.frame_counter > 10:
            self.frame_counter = 0
            holder = self.current_state
            self.current_state = "ROAD"
            print(f'{holder} -------> {self.current_state}')

    ## @fn publish_clues
    #  @brief This function publishes the clues detected by the robot.
    def publish_clues(self):
        print("Publishing Clues")
        # Iterate over the list of clues
        for i, clue_set in enumerate(self.clues):
            # Reset the list of good values
            self.good_values = []
            
            # Reverse the order of the clues in the set and take the first 5
            clue_set2 = clue_set[::-1]
            clue_set3 = clue_set2[:5]
            for clue in clue_set3:
                # Detect the key and value of the clue
                key, value = clue_detect(clue)
                if key in possible_keys:
                    self.good_values.append(value)
                    submit_key = key

            # If there are any good values 
            if self.good_values:
                # Use Counter to count occurrences of each element in the list
                counts = Counter(self.good_values)

                # Find the most common element and its count - make it so its the most recent most common
                submit_value, count = counts.most_common(1)[0]

                print(f"Submitting {submit_key} = {submit_value}")
                pub_clue(id,password,possible_keys[submit_key],submit_value)
            else:
                print("No good values found")
                # Publish most recent value regardless (if any)
                clue = clue_set2[0]
                key, value = clue_detect(clue)
                print(f"Submitting clue#{i+1} = {value}")
                pub_clue(id,password,i+1,value)

        
        # The following commented out code saves clues with correct names to files
        # if not os.path.exists(output_path):
        #    os.makedirs(output_path)

        # with open(csv_file_path, 'r', newline='') as file:
        #     reader = csv.reader(file)
        #     rows = list(reader)
        #     #print(rows.shape)
            
        #     for i, clue_set in enumerate(self.clues):
                
        #         row = []
        #         if 0 <= i <= len(rows):
        #             row = rows[i]
        #         else:
        #             print(f"Row number {i} is out of range.")
                
        #         print(row)
        #         #pattern = r'([^,]+),(.+)'
        #         #match = re.match(pattern, row)

        #         key = row[0]
        #         value = row[1]

        #         for j, clue in enumerate(clue_set):

        #             filename = f'plate{j}_{key}_{value}.png'
        #             filename = filename.replace(' ', '_')
        #             file_output = output_path + filename
        #             # Write the image using the specified filename
        #             print(f"DOWNLOADING FILE: {filename}")
        #             #cv2.imshow(clue)
        #             cv2.imwrite(filename, clue)

    ## @fn end_run
    #  @brief This function is called when the run is ending.
    def end_run(self, event):
        print("ENDING RUN")
        self.publish_clues()
        pub_clue(id,password,-1,"NA")

    ## @fn event_occurred
    #  @brief This function is called when an event occurs.
    def event_occurred(self, event, data):
        # Callback function to handle the event
        if event == "PINK" and not self.pink_cooldown:
            print(event)
            self.pink_cooldown = True
            rospy.Timer(rospy.Duration(2), self.reset_pink_cooldown, oneshot=True)
            holder = self.current_state
            if holder == "ROAD": 
                self.current_state = "GRASS"
                self.grass_state()
            elif holder == "GRASS":
                #pub_cmd_vel(0,0)
                #self.publish_clues()
                self.current_state = "YODA_WAIT"
                pub_cmd_vel(0,0)
                self.publish_clues()
            elif holder == "YODA_DRIVE":
                #self.align() 
                self.current_state = "MOUNTAIN"
                self.aligned = False
                self.mountain_state()
                rospy.Timer(rospy.Duration(4), self.mountain_boost_on, oneshot=True)
            print(f'{holder} -------> {self.current_state}')
        
        if event == "RED" and not self.ped_xing:
            pub_cmd_vel(0,0)
            print(event)
            self.ped_xing = True
            holder = self.current_state
            if holder == "ROAD": 
                self.current_state = "PEDESTRIAN"
            print(f'{holder} -------> {self.current_state}')
        
        if event == "YODA_STOP" and self.yoda_wait is True:
            pub_cmd_vel(0,0)
            print(event)
            holder = self.current_state
            if holder == "YODA_DRIVE": 
                self.current_state = "YODA_WAIT"
                
            print(f'{holder} -------> {self.current_state}')

        ## @brief If the event is "CLUE", the clue is added to the current clue list.
        if event == "CLUE":
            if self.clue_count == 4:
                self.bridge_boost = False
            if not self.clue_cooldown:
                self.clue_cooldown = True
                rospy.Timer(rospy.Duration(1), self.update_clue_count, oneshot=True)
            self.cur_clue.append(data)
            print("CLUE ADDED")
            
    ## @fn reset_pink_cooldown
    #  @brief This function resets the pink cooldown.
    def reset_pink_cooldown(self, event):
        self.pink_cooldown = False

    ## @fn update_clue_count
    #  @brief This function updates the clue count and moves to the next clueboard.
    def update_clue_count(self, event):
        self.clue_cooldown = False
        self.clues.append(self.cur_clue)
        print(len(self.cur_clue))
        self.cur_clue = []
        #Code for vehicle detection if necessary
        # if self.clue_count == 2:
        #     pub_cmd_vel(0,0)
        #     holder = self.current_state
        #     if holder == "ROAD": 
        #         self.current_state = "VEHICLE"
        #     print(f'{holder} -------> {self.current_state}')
        if self.clue_count == 4:
            rospy.Timer(rospy.Duration(1.2), self.bridge_boost_on, oneshot=True)
        if self.clue_count == 7:
            #pub_cmd_vel(0,0)
            holder = self.current_state
            if holder == "MOUNTAIN": 
                #self.current_state = "FINISH"
                self.publish_clues()
                pub_clue(id,password,-1,"NA")
            print(f'{holder} -------> {self.current_state}')
        self.clue_count +=1
        print(f'Moving to CLUE #{self.clue_count+1}')

        # self.publish_clues()

        
    ## @fn set_yoda_wait
    #  @brief This function sets Yoda to wait.
    def set_yoda_wait(self, event):
        self.yoda_wait = True
        print("YODA 1 passed")
    
    ## @fn bridge_boost_on
    #  @brief This function turns on the bridge boost.
    def bridge_boost_on(self, event):
        print("bridge boost")
        self.bridge_boost = True

    ## @fn mountain_boost_on
    #  @brief This function turns on the mountain boost.
    def mountain_boost_on(self, event):
        print("mountain boost")
        self.mountain_boost = True

    ## @fn align
    #  @brief This function aligns the robot.
    def align(self):
        contours, _ = cv2.findContours(self.state_data2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        # Get orientation of the line using its bounding rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        if angle>45:
            angle = angle-90
        # Rotate the robot to align with the line (example, adjust as needed)
        # Your robot control logic here to adjust orientation based on 'angle'
        # For simulation purposes, let's print the angle
        print("Angle to straighten:", angle)
        if (state_machine.current_state == "YODA_WAIT"):
            thresh1 = .3
            thresh2 = -.3
        else:
            thresh1 = -.3
            thresh2 = -.6
        if angle > thresh1:
            pub_cmd_vel(-.07,-.5)
            print("RIGHT")
            self.frame_counter+=1
        elif angle < thresh2:
            pub_cmd_vel(-.07,.5)
            print("LEFT")
            self.frame_counter+=1
        else:
            if self.frame_counter > 2:
                print("ALIGNED")
                self.aligned = True
                self.frame_counter=0
            self.frame_counter+=1
        
            

# Initialize publishers
rospy.init_node('controller_node')
move_pub = rospy.Publisher('/R1/cmd_vel', Twist, 
  queue_size=10)    
br = CvBridge()
score_pub = rospy.Publisher('/score_tracker', String, 
  queue_size=10)
rospy.sleep(1)
id = "winux"
password = "password"

state_machine = StateMachine()


## @fn camera_callback
#  @brief Callback function to handle the image.
def camera_callback(data):
    lower_pink= np.array([140,80,105])
    upper_pink= np.array([153,255,255])
    lower_red= np.array([0,42,102])
    upper_red= np.array([9,255,255])
    lower_blue = np.array([110,50,50])
    upper_blue= np.array([130,255,255])
    lower_ped = np.array([98,20,22])
    upper_ped = np.array([110,255,116])
    lower_blue = np.array([110,50,50])
    upper_blue= np.array([130,255,255])
    lower_vehicle = np.array([0,0,93])
    upper_vehicle = np.array([10,10,249])
    lower_yoda1 = np.array([165,10,14])
    upper_yoda1 = np.array([186,170,170])
    lower_yoda2 = np.array([40,88,40])
    upper_yoda2 = np.array([50,100,50])
    try:
        cv_image = br.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)

    # process image for NN
    im_grey = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    dim = (160, 90)
    resized = cv2.resize(im_grey, dim, interpolation = cv2.INTER_AREA)
    img = resized / 255.0 
    state_machine.drive_input = img

    # image masking for line and clue detection 
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_white = cv2.bitwise_not(mask_blue)
    mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_white = cv2.bitwise_not(mask_blue)
    mask_ped = cv2.inRange(hsv, lower_ped, upper_ped)
    mask_vehicle = cv2.inRange(hsv, lower_vehicle, upper_vehicle)
    mask_yoda1 = cv2.inRange(hsv, lower_yoda1, upper_yoda1)
    mask_yoda2 = cv2.inRange(cv_image, lower_yoda2, upper_yoda2)

    # Clue Detection
    cnts, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.fillPoly(mask_blue, cnts, (255,255,255))
    filtered_cnts = [contour for contour in cnts if cv2.contourArea(contour) > 1000]
    mask_clue = cv2.bitwise_and(mask_blue,mask_white)
    cnts, _ = cv2.findContours(mask_clue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_cnts = [contour for contour in cnts if cv2.contourArea(contour) > 1000]
    
    if cv2.countNonZero(mask_clue) > 10000 and filtered_cnts:
            print(cv2.countNonZero(mask_clue))
            approx = []
            for c in filtered_cnts:
                epsilon = 0.08 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
            #     centroid = corners[0]
            #     def sort_key(point):
            #         angle = np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
            #         return (angle + 2 * np.pi) % (2 * np.pi)
            #     # Sort the source points based on their relative positions to match the destination points format
            #     sorted_src = sorted(src, key=sort_key)
            #     sorted_src = np.array(sorted_src)

            #     # Reorder 'src' points to match the 'dest' format
                # approx
            if len(approx) == 4:
                
                # Sort Corners
                centroid = np.mean(approx, axis=0)[0]

                #get x cord of centroid 
                x = centroid[0]
                #get y cord of centroid
                y = centroid[1]

                def sort_key(point):
                    angle = np.arctan2(point[0][1] - centroid[1], point[0][0] - centroid[0])
                    return (angle + 2 * np.pi) % (2 * np.pi)
                # Sort the source points based on their relative positions to match the destination points format
                sorted_approx = sorted(approx, key=sort_key)
                sorted_approx = np.array(sorted_approx)

            #     # Reorder 'src' points to match the 'dest' format
                # approx
                #print(sorted_approx)
                src = np.array([sorted_approx[2], sorted_approx[3], sorted_approx[1], sorted_approx[0]], dtype=np.float32)


                width = 600
                height= 400
                dest = np.float32([[0, 0],
                            [width, 0],
                            [0, height],
                            [width , height]])

                M = cv2.getPerspectiveTransform(src,dest)
                clue = cv2.warpPerspective(cv_image,M,(width, height),flags=cv2.INTER_LINEAR)
                #cv2.imshow("cam", cv_image)
                #cv2.imshow("clue", clue)

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
                #cv2.imshow("result", result)
                #cv2.waitKey(1)
                state_machine.event_occurred("CLUE",result)
                #print("Update result")

    if cv2.countNonZero(mask_pink) > 15000:
            state_machine.event_occurred("PINK",None)

    if cv2.countNonZero(mask_red) > 30000:
            state_machine.event_occurred("RED",None)

    if (cv2.countNonZero(mask_yoda2) > 10 and state_machine.current_state == "YODA_DRIVE"):
            state_machine.event_occurred("YODA_STOP",None)

    state_machine.cv_image = cv_image
      # resize image
    

    if state_machine.current_state == "ROAD":
        state_machine.state_data1 = mask_clue
        state_machine.road_state()
    elif state_machine.current_state == "GRASS":
        state_machine.grass_state()
        state_machine.state_data1 = mask_pink
    elif state_machine.current_state == "PEDESTRIAN":
        state_machine.state_data1 = mask_ped
        state_machine.pedestrian_state()
    elif state_machine.current_state == "VEHICLE":
        state_machine.state_data1 = mask_vehicle
        state_machine.vehicle_state()
    elif state_machine.current_state == "YODA_WAIT":
        if state_machine.yoda_wait == False:
            state_machine.state_data1 = mask_yoda1
            state_machine.state_data2 = mask_pink
        elif state_machine.yoda_wait == True:
            state_machine.state_data1 = mask_yoda2
        state_machine.yoda_wait_state()
    elif state_machine.current_state == "YODA_DRIVE":
        state_machine.yoda_drive_state()
    elif state_machine.current_state == "MOUNTAIN":
        state_machine.state_data2 = mask_pink
        state_machine.mountain_state()
            
    #pub_clue(id,password,-1,"NA")
## @fn subscribe()
# @brief The function subscribe() subscribes to the ROS topics '/R1/pi_camera/image_raw' and '/clock'.
def subscribe():
    rospy.Subscriber('/R1/pi_camera/image_raw',Image,camera_callback)
    rospy.Subscriber('/clock',Clock)
    rospy.sleep(1)

subscribe()

## @brief The main loop of the program. It keeps running until ROS is shutdown.
rate = rospy.Rate(10)
while not rospy.is_shutdown():
    rospy.spin()    
