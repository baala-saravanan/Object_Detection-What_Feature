import time
import os
from tflite_runtime.interpreter import Interpreter
import re
import cv2
import subprocess
import vlc
import sys
from pydub import AudioSegment
import numpy as np

sys.path.insert(0, '/home/rock/Desktop/Hearsight/')
from play_audio import GTTSA

play_audio = GTTSA()

img_path = "/home/rock/Desktop/Hearsight/English/what/01.jpg"

class SSD:
    def __init__(self):
        self.labels = self.load_labels("/home/rock/Desktop/Hearsight/English/what/model_data/labels.txt")
        self.interpreter = Interpreter("/home/rock/Desktop/Hearsight/English/what/model_data/object_detetion_advantech.tflite")
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.interpreter.allocate_tensors()
        _, self.img_inp_shape, _, _ = self.input_details[0]['shape']
        self.threshold = 0.5

    def load_labels(self, path):
        # Loads the labels file. Supports files with or without index numbers.
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            labels = {}
            for row_number, content in enumerate(lines):
                pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
                if len(pair) == 2 and pair[0].strip().isdigit():
                    labels[int(pair[0])] = pair[1].strip()
                else:
                    labels[row_number] = pair[0].strip()
            return labels

    def get_image(self):
        cap = cv2.VideoCapture(1)  # Change the camera index as needed
        if not cap.isOpened():
#            play_audio.play_machine_audio("camera_is_not_working_so_switch_off_the_HearSight_device_for_some_time_and_then_start_it_again.mp3")
#            play_audio.play_machine_audio("check_your_connection_and_proceed.mp3")
            play_audio.play_machine_audio("hold_on_connection_in_progress_initiating_shortly.mp3")
            play_audio.play_machine_audio("Thank You.mp3")
            subprocess.run(["reboot"])
            return None
        cap.release()
        cv2.destroyAllWindows()

        if os.path.exists(img_path):
            os.remove(img_path)
        
        # Capture the image using OpenCV (cv2)
        cap = cv2.VideoCapture(1)
        ret, frame = cap.read()
        if not ret:
            play_audio.play_machine_audio("image_capture_failed_so_retake_it_again.mp3")
            return None
        cv2.imwrite(img_path, frame)
        cap.release()
        cv2.destroyAllWindows()

#        play_audio.play_machine_audio('Image_captured.mp3')
#        play_audio.play_machine_audio('Processing.mp3')

        frame = cv2.imread(img_path)
        self.orig_width, self.orig_height, _ = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.img_inp_shape, self.img_inp_shape))
        frame = frame.astype(np.uint8)  # Ensure the image data type is UINT8
        frame = np.expand_dims(frame, axis=0)  # Add a batch dimension
        return frame

    def count_and_say(self, lst):
        count_dict = {}
        output_list = []
        for item in lst:
            if item in count_dict:
                count_dict[item] += 1
            else:
                count_dict[item] = 1
        for item, count in count_dict.items():
            item = item.replace("_", " ")
            if count >= 1:
                output_list.append(f"{count} {item}")
                play_audio.play_machine_audio(f"number_{count}.mp3")
                play_audio.play_machine_audio(f"{item}.mp3")
            else:
                output_list.append(item)
                play_audio.play_machine_audio(f"{item}.mp3")
        return output_list

    def detect(self):
        img = self.get_image()
        if img is None:
            return  # Exit early if image capture failed

        start_time = time.time()

        # Modify this line to pass the image as a NumPy array
        self.interpreter.set_tensor(self.input_details[0]['index'], img)

        self.interpreter.invoke()
        rects = self.interpreter.get_tensor(self.output_details[0]['index'])
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])

        img = cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR)  # Convert the image back for display
        obj_list = []
        for index, score in enumerate(scores[0]):
            if score > self.threshold:
                obj = self.labels[classes[0][index]]
                print(obj)
                obj_list.append(obj)
        result = self.count_and_say(obj_list)
        for item in result:
            print(item)
        print("Objects: ", obj_list)
        new_img = cv2.resize(img, (self.orig_height, self.orig_width))

        elapsed_time = time.time() - start_time
        print(elapsed_time)
        if elapsed_time < 1:
#            play_audio.play_machine_audio("unknown.mp3")
            play_audio.play_machine_audio("undefined.mp3")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        os.remove(img_path)
