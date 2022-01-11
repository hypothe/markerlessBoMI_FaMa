# General imports
from tkinter.constants import DISABLED, INSERT
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
# For multithreading
from threading import Thread, Lock
import queue
# For OpenCV
import cv2
# For GUI
import tkinter as tk
from tkinter import Label, Button, BooleanVar, Checkbutton, Text, Entry, Radiobutton, IntVar
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

# For pygame
import pygame
# For reaching task
from scripts.reaching import Reaching
import scripts.reaching as reaching
import scripts.cv_utils as cv_utils
from scripts.stopwatch import StopWatch
from scripts.filter_butter_online import FilterButter3
import scripts.reaching_functions as reaching_functions
import scripts.tk_utils as tk_utils
import scripts.mediapipe_utils as mediapipe_utils
# For controlling computer cursor
import pyautogui
# For Mediapipe
import mediapipe as mp
# For training pca/autoencoder
from scripts.compute_bomi_map import Autoencoder, PrincipalComponentAnalysis, compute_vaf, load_bomi_map, train_ae, train_vae, train_pca, save_bomi_map

# Custom packages
import ctypes
import math
import copy

pyautogui.PAUSE = 0.01  # set fps of cursor to 100Hz ish when mouse_enabled is True


class SharedDetImage:
    def __init__(self):
        self.image_id = 0
        self.image = None
        self.result = None
        self.result_face = None
        self.lock = Lock()

class JointMapper(tk.Frame):
    """
    class that defines the main tkinter window --> graphic with buttons etc..
    """

    def __init__(self, win, n_map_components, *args, **kwargs):

        self.video_camera_device = 0 # -> 0 for the camera

        self.current_image_data = SharedDetImage()
        # from single instance to queue
        self.body = queue.Queue()

        tk.Frame.__init__(self, win, *args, **kwargs)
        self.parent = win

        self.app = CustomizationApplication(self)

        self.calibPath = os.path.dirname(os.path.abspath(__file__)) + "/../calib/"
        self.drPath = ''
        self.num_joints = 0
        self.joints = np.zeros((5, 1))
        self.dr_mode = 'ae'
        self.font_size = 18
        self.n_map_component = n_map_components

        self.btn_num_joints = Button(win, text="Select Joints", command=self.select_joints, activeforeground='blue', activebackground='#4682b4', bg='#abcdef')
        self.btn_num_joints.config(font=("Times", self.font_size))
        self.btn_num_joints.grid(row=0, column=0, columnspan=2, padx=20, pady=30, sticky='nesw')

        # set checkboxes for selecting joints
        self.check_nose = BooleanVar()
        self.check1 = Checkbutton(win, text="Nose", variable=self.check_nose, activeforeground='blue', activebackground='#4682b4', bg='#abcdef')
        self.check1.config(font=("Times", self.font_size))
        self.check1.grid(row=0, column=2, padx=(0, 40), pady=30, sticky='w')

        self.check_eyes = BooleanVar()
        self.check2 = Checkbutton(win, text="Eyes", variable=self.check_eyes, activeforeground='blue', activebackground='#4682b4', bg='#abcdef')
        self.check2.config(font=("Times", self.font_size))
        self.check2.grid(row=0, column=3, padx=(0, 40), pady=30, sticky='w')

        self.check_shoulders = BooleanVar()
        self.check3 = Checkbutton(win, text="Shoulders", variable=self.check_shoulders, activeforeground='blue', activebackground='#4682b4', bg='#abcdef')
        self.check3.config(font=("Times", self.font_size))
        self.check3.grid(row=0, column=4, padx=(0, 30), pady=30, sticky='w')

        self.check_forefinger = BooleanVar()
        self.check4 = Checkbutton(win, text="Right Forefinger",
                                  variable=self.check_forefinger, activeforeground='blue', activebackground='#4682b4', bg='#abcdef')
        self.check4.config(font=("Times", self.font_size))
        self.check4.grid(row=0, column=5, padx=(0, 20), pady=30, sticky='w')

        self.check_fingers = BooleanVar()
        self.check5 = Checkbutton(win, text="Fingers", variable=self.check_fingers, activeforeground='blue', activebackground='#4682b4', bg='#abcdef')
        self.check5.config(font=("Times", self.font_size))
        self.check5.grid(row=0, column=6, padx=(0, 20), pady=30, sticky='nesw')

        self.btn_calib = Button(win, text="Calibration", command=self.calibration, activeforeground='blue', activebackground='#4682b4', bg='#abcdef')
        self.btn_calib["state"] = "disabled"
        self.btn_calib.config(font=("Times", self.font_size))
        self.btn_calib.grid(row=1, column=0, columnspan=2, padx=20, pady=(20, 30), sticky='nesw')
        self.calib_duration = 10000 #30000

        # Calibration time remaining
        self.lbl_calib = Label(win, text='Calibration time: ', activebackground='#4682b4', bg='#abcdef')
        self.lbl_calib.config(font=("Times", self.font_size))
        self.lbl_calib.grid(row=1, column=2, columnspan=2, pady=(20, 30), sticky='w')

        # BoMI map button and checkboxes
        self.btn_map = Button(win, text="Calculate BoMI Map", command=self.train_map, activeforeground='blue', activebackground='#4682b4', bg='#abcdef')
        self.btn_map["state"] = "disabled"
        self.btn_map.config(font=("Times", self.font_size))
        self.btn_map.grid(row=2, column=0, columnspan=2, padx=20, pady=(20, 30), sticky='nesw')

        self.check_alg = IntVar()

        # self.check_pca = BooleanVar()
        self.check_pca1 = Radiobutton(win, text="PCA", variable=self.check_alg, value=0, activeforeground='blue', activebackground='#4682b4', bg='#abcdef')
        self.check_pca1.config(font=("Times", self.font_size))
        self.check_pca1.grid(row=2, column=2, padx=(0, 20), pady=(20, 30), sticky='w')

        # self.check_ae = BooleanVar()
        self.check_ae1 = Radiobutton(win, text="AE", variable=self.check_alg, value=1, activeforeground='blue', activebackground='#4682b4', bg='#abcdef')
        self.check_ae1.config(font=("Times", self.font_size))
        self.check_ae1.grid(row=2, column=3, padx=(0, 20), pady=(20, 30), sticky='w')

        # self.check_vae = BooleanVar()
        self.check_vae1 = Radiobutton(win, text="VAE", variable=self.check_alg, value=2, activeforeground='blue', activebackground='#4682b4', bg='#abcdef')
        self.check_vae1.config(font=("Times", self.font_size))
        self.check_vae1.grid(row=2, column=4, pady=(20, 30), sticky='w')

        self.btn_custom = Button(win, text="Customization", command=self.customization, activeforeground='blue', activebackground='#4682b4', bg='#abcdef')
        self.btn_custom["state"] = "disabled"
        self.btn_custom.config(font=("Times", self.font_size))
        self.btn_custom.grid(row=3, column=0, columnspan=2, padx=20, pady=(20, 30), sticky='nesw')

        self.btn_start = Button(win, text="Practice", command=self.start, activeforeground='blue', activebackground='#4682b4', bg='#abcdef')
        self.btn_start["state"] = "disabled"
        self.btn_start.config(font=("Times", self.font_size))
        self.btn_start.grid(row=4, column=0, columnspan=2, padx=20, pady=(20, 30), sticky='nesw')

        # set label for number of target remaining
        self.lbl_tgt = Label(win, text='Remaining targets: ', activebackground='#4682b4', bg='#abcdef')
        self.lbl_tgt.config(font=("Times", self.font_size))
        self.lbl_tgt.grid(row=4, column=2, pady=(20, 30), columnspan=2, sticky='w')


        # Camera video input
        self.btn_cam = Button(win, text='Ext. Video Source', command=self.selectVideoFile,activeforeground='blue', activebackground='#4682b4', bg='#abcdef')
        self.btn_cam.config(font=("Times", self.font_size))
        self.btn_cam.grid(row=5, column=0, columnspan=2, padx=20, pady=(20, 30), sticky='nesw')
        self.ent_cam = Entry(win, width=30)
        self.ent_cam.config(font=("Times", self.font_size))
        self.ent_cam.grid(row=5, column=2, pady=(20, 30), columnspan=3, sticky='w')
        self.btn_camClear = Button(win, text='Camera Video Source', command=self.clearVideoSource, bg='red', activebackground='grey')
        self.btn_camClear.config(font=("Times", self.font_size))
        self.btn_camClear.grid(row=5, column=5, columnspan=2, padx=20, pady=(20, 30), sticky='nesw')

        #############################################################

        self.btn_close = Button(win, text="Close", command=win.destroy, bg="red", activebackground='grey')
        self.btn_close.config(font=("Times", self.font_size))
        self.btn_close.grid(row=8, column=0, columnspan=2, padx=20, pady=(20, 30), sticky='nesw')

    def selectVideoFile(self):
        filetypes = (
            ('all files', '*.*'),
            ('mp4', '*.mp4'),
            ('mkv', '*.mkv'),
            ('avi', '*.avi')
        )
        filename = fd.askopenfilename(
            title = 'Select a video',
            initialdir='.',
            filetypes=filetypes
        )

        self.video_camera_device = filename
        out_txt = filename
        if not self.video_camera_device:
            self.video_camera_device = 0
            out_txt = "/dev/video0"

        showinfo(
            title = 'Selected File',
            message = str(out_txt)
        )
        # clear any previous text and add this one
        self.ent_cam.delete(0, 'end')
        self.ent_cam.insert(INSERT, out_txt)

    def clearVideoSource(self):
        self.video_camera_device = 0
        self.ent_cam.delete(0, 'end')
        self.ent_cam.insert(INSERT, '/dev/video0')


    # Count number of joints selected
    def select_joints(self):
        nose_enabled = self.check_nose.get()
        eyes_enabled = self.check_eyes.get()
        shoulders_enabled = self.check_shoulders.get()
        forefinger_enabled = self.check_forefinger.get()
        fingers_enabled = self.check_fingers.get()
        if nose_enabled:
            self.num_joints += 2
            self.joints[0, 0] = 1
        if eyes_enabled:
            self.num_joints += 4
            self.joints[1, 0] = 1
        if shoulders_enabled:
            self.num_joints += 4
            self.joints[2, 0] = 1
        if forefinger_enabled:
            self.num_joints += 2
            self.joints[3, 0] = 1
        if fingers_enabled:
            self.num_joints += 10
            self.joints[4, 0] = 1
        if np.sum(self.joints, axis=0) != 0:
            self.btn_calib["state"] = "normal"
            self.btn_map["state"] = "normal"
            self.btn_custom["state"] = "normal"
            self.btn_start["state"] = "normal"
            print('Joints correctly selected.')
        else:
            self.w = tk_utils.popupWindow(self.master, "No Joint selected.")

    def calibration(self):
        # start calibration dance - collect webcam data
        self.w = tk_utils.popupWindow(self.master, "You will now start calibration. \n You have the possibility to control the virtual mouse. \n If you are interested, after the calibration, select the checkbox 'Mouse Control'.")
        self.master.wait_window(self.w.top)
        if self.w.status:
            # This variable helps to check which joint to display
            self.check_summary = [self.check_nose.get(), self.check_eyes.get(), self.check_shoulders.get(),
                                    self.check_forefinger.get(), self.check_fingers.get()]
            self.compute_calibration(self.calibPath, self.calib_duration, self.lbl_calib, self.num_joints, self.joints,
                                self.check_summary, self.video_camera_device)
            self.btn_map["state"] = "normal"

    def train_map(self):
        # check whether calibration file exists first
        if os.path.isfile(self.calibPath + "Calib.txt"):
            self.w = tk_utils.popupWindow(self.master, "You will now train BoMI map")
            self.master.wait_window(self.w.top)
            if self.w.status:
                print(self.check_alg.get())

                if self.check_alg.get() == 0:
                    self.drPath = self.calibPath + 'PCA/'
                    train_cu = train_pca(self.calibPath, self.drPath, self.n_map_component)
                    self.dr_mode = 'pca'

                elif self.check_alg.get() == 1:
                    self.drPath = self.calibPath + 'AE/'
                    train_cu = train_ae(self.calibPath, self.drPath, self.n_map_component)
                    self.dr_mode = 'ae'

                elif self.check_alg.get() == 2:
                    self.drPath = self.calibPath + 'VAE/'
                    train_cu = train_vae(self.calibPath, self.drPath, self.n_map_component)
                    self.dr_mode = 'vae'

                # rotate, scale and offset the original features
                # implementation-dependant (depends on the workspace of
                # each variable, eg. screen space vs. joint space)
                self.map_to_workspace(self.drPath, train_cu)

                self.btn_custom["state"] = "normal"
        else:
            self.w = tk_utils.popupWindow(self.master, "Perform calibration first.")
            self.master.wait_window(self.w.top)
            self.btn_map["state"] = "disabled"

    def map_to_workspace(self, drPath, train_cu):
        self.w = tk_utils.popupWindow(self.master, "This function is not implemented here.")

    def customization(self):
        # check whether PCA/AE parameters have been saved
        if os.path.isfile(self.drPath + "weights1.txt"):
            # open customization window
            self.newWindow = tk.Toplevel(self.master)
            self.newWindow.geometry("1000x500")
            self.newWindow.title("Customization")
            self.app.generate_window(self.newWindow, drPath=self.drPath, num_joints=self.num_joints,
                                                joints=self.joints, dr_mode=self.dr_mode, video_camera_device=self.video_camera_device)
        else:
            self.w = tk_utils.popupWindow(self.master, "Compute BoMI map first.")
            self.master.wait_window(self.w.top)
            self.btn_custom["state"] = "disabled"

    def start(self):
        # implementation-specific action
        self.w = tk_utils.popupWindow(self.master, "This function is not implemented here.")


    def compute_calibration(self, drPath, calib_duration, lbl_calib, num_joints, joints, active_joints, video_device=0):
        """
        function called to collect calibration data from webcam
        :param drPath: path to save calibration file
        :param calib_duration: duration of calibration as read by the textbox in the main window
        :param lbl_calib: label in the main window that shows calibration time remaining
        :return:
        """
        # Create object of openCV and Reaching (needed for terminating mediapipe thread)
        # try using an external video source, if present
        print("Using video device {}".format(video_device))

        cap = cv_utils.VideoCaptureOpt(video_device)

        r = Reaching()

        # The clock will be used to control how fast the screen updates. Stopwatch to count calibration time elapsed
        clock = pygame.time.Clock()
        timer_calib = StopWatch()

        # initialize MediaPipe Pose
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                        smooth_landmarks=False)

        # initialize lock for avoiding race conditions in threads
        lock = Lock()

        # class member accessed by main and mediapipe threads that contains the current vector of body landmarks
        
        self.body = queue.Queue()
        #body_calib = []  # initialize local variable (list of body landmarks during calibration)

        # start thread for OpenCV. current frame will be appended in a queue in a separate thread
        q_frame = queue.Queue()
        opencv_thread = Thread(target=cv_utils.get_data_from_camera, args=(cap, q_frame, r, calib_duration))
        opencv_thread.start()
        print("openCV thread started in calibration.")

        # initialize thread for mediapipe operations
        mediapipe_thread = Thread(target=mediapipe_utils.mediapipe_forwardpass,
                                args=(self.current_image_data, self.body, holistic, mp_holistic, lock, q_frame, r, num_joints, joints, cap.get(cv2.CAP_PROP_FPS), None))
        mediapipe_thread.start()

        body_write_thread = Thread(target=save_bomi_map, args=(self.body, drPath, r))
        print("mediapipe thread started in calibration.")

        # start the timer for calibration
        #timer_calib.start()

        print("main thread: Starting calibration...")

        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        wind_name = "Group 12 cam"
        cv2.namedWindow(wind_name)

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        body_write_thread.start()

        while not r.is_terminated:

            # safe access to the current image and results, since they can
            # be modified by the mediapipe_forwardpass thread
            with self.current_image_data.lock:
                if self.current_image_data.image_id == 1:
                    timer_calib.start()
                frame = copy.deepcopy(self.current_image_data.image)
                results = copy.deepcopy(self.current_image_data.result)


            if frame is None or results is None:
                continue
            # Draw landmark annotation on the image.
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if active_joints[1] == True:
                mp_drawing.draw_landmarks(
                    frame,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
            if (active_joints[0] == True) or (active_joints[2] == True):
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                        .get_default_pose_landmarks_style())
            if (active_joints[3] == True) or (active_joints[4] == True):
                mp_drawing.draw_landmarks(
                    frame,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                        .get_default_pose_landmarks_style())
                mp_drawing.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                        .get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            screen_width, screen_height = pyautogui.size()

            window_width = math.ceil(screen_width / 2)
            window_height = math.ceil(screen_height / 2)

            cv2.imshow(wind_name, frame)
            cv2.moveWindow(wind_name, int(window_width / 2), int(window_height / 4))

            if cv2.waitKey(1) == 27:
                break  # esc to quit

            # update time elapsed label
            time_remaining = int((calib_duration - timer_calib.elapsed_time) / 1000)
            lbl_calib.configure(text='Calibration time: ' + str(time_remaining))
            lbl_calib.update()

            # --- Limit to 50 frames per second
            clock.tick(50)

        opencv_thread.join()
        mediapipe_thread.join()
        body_write_thread.join()

        cv2.destroyAllWindows()
        # Stop the game engine and release the capture
        holistic.close()
        print("pose estimation object released in calibration.")
        cap.release()
        print("openCV object released in calibration.")

        # print calibration file
        #body_calib = np.array(body_calib)
        #if not os.path.exists(drPath):
        #    os.makedirs(drPath)
        #np.savetxt(drPath + "Calib.txt", body_calib)

        print('Calibration finished. You can now train BoMI forward map.')
    
    

class CustomizationApplication(tk.Frame):
    """
    class that defines the customization tkinter window
    """

    def __init__(self, mainTk):
        self.mainTk = mainTk
        self.current_image_data = SharedDetImage()
        self.body = queue.Queue()

    def generate_window(self, parent, drPath, num_joints, joints, dr_mode, video_camera_device):
        self.w = tk_utils.popupWindow(self.master, "This function is not implemented here.")


