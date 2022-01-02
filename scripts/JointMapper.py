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
import scripts.tk_utilities as tk_utils
import scripts.mediapipe_utils as mediapipe_utils
# For controlling computer cursor
import pyautogui
# For Mediapipe
import mediapipe as mp
# For training pca/autoencoder
from scripts.compute_bomi_map import Autoencoder, PrincipalComponentAnalysis, compute_vaf, load_bomi_map, train_ae, train_pca

# Custom packages
import ctypes
import math
import copy

pyautogui.PAUSE = 0.01  # set fps of cursor to 100Hz ish when mouse_enabled is True


class SharedDetImage:
    def __init__(self):
        self.image = None
        self.result = None
        self.lock = Lock()

class BodyWrap:
    def __init__(self):
        self.body = None

class JointMapper(tk.Frame):
    """
    class that defines the main tkinter window --> graphic with buttons etc..
    """

    def __init__(self, win, n_map_components, *args, **kwargs):

        self.video_camera_device = 0 # -> 0 for the camera
        #self.current_image = None
        #self.results = None

        self.current_image_data = SharedDetImage()
        self.body_wrap = BodyWrap()

        tk.Frame.__init__(self, win, *args, **kwargs)
        self.parent = win
        self.calibPath = os.path.dirname(os.path.abspath(__file__)) + "/../calib/"
        self.drPath = ''
        self.num_joints = 0
        self.joints = np.zeros((5, 1))
        self.dr_mode = 'ae'
        self.font_size = 18
        self.n_map_component = n_map_components

        self.btn_num_joints = Button(win, text="Select Joints", command=self.select_joints)
        self.btn_num_joints.config(font=("Arial", self.font_size))
        self.btn_num_joints.grid(row=0, column=0, columnspan=2, padx=20, pady=30, sticky='nesw')

        # set checkboxes for selecting joints
        self.check_nose = BooleanVar()
        self.check1 = Checkbutton(win, text="Nose", variable=self.check_nose)
        self.check1.config(font=("Arial", self.font_size))
        self.check1.grid(row=0, column=2, padx=(0, 40), pady=30, sticky='w')

        self.check_eyes = BooleanVar()
        self.check2 = Checkbutton(win, text="Eyes", variable=self.check_eyes)
        self.check2.config(font=("Arial", self.font_size))
        self.check2.grid(row=0, column=3, padx=(0, 40), pady=30, sticky='w')

        self.check_shoulders = BooleanVar()
        self.check3 = Checkbutton(win, text="Shoulders", variable=self.check_shoulders)
        self.check3.config(font=("Arial", self.font_size))
        self.check3.grid(row=0, column=4, padx=(0, 30), pady=30, sticky='w')

        self.check_forefinger = BooleanVar()
        self.check4 = Checkbutton(win, text="Right Forefinger",
                                  variable=self.check_forefinger)
        self.check4.config(font=("Arial", self.font_size))
        self.check4.grid(row=0, column=5, padx=(0, 20), pady=30, sticky='w')

        self.check_fingers = BooleanVar()
        self.check5 = Checkbutton(win, text="Fingers", variable=self.check_fingers)
        self.check5.config(font=("Arial", self.font_size))
        self.check5.grid(row=0, column=6, padx=(0, 20), pady=30, sticky='nesw')

        self.btn_calib = Button(win, text="Calibration", command=self.calibration)
        self.btn_calib["state"] = "disabled"
        self.btn_calib.config(font=("Arial", self.font_size))
        self.btn_calib.grid(row=1, column=0, columnspan=2, padx=20, pady=(20, 30), sticky='nesw')
        self.calib_duration = 10000 #30000

        # Calibration time remaining
        self.lbl_calib = Label(win, text='Calibration time: ')
        self.lbl_calib.config(font=("Arial", self.font_size))
        self.lbl_calib.grid(row=1, column=2, columnspan=2, pady=(20, 30), sticky='w')

        # BoMI map button and checkboxes
        self.btn_map = Button(win, text="Calculate BoMI Map", command=self.train_map)
        self.btn_map["state"] = "disabled"
        self.btn_map.config(font=("Arial", self.font_size))
        self.btn_map.grid(row=2, column=0, columnspan=2, padx=20, pady=(20, 30), sticky='nesw')

        self.check_alg = IntVar()

        # self.check_pca = BooleanVar()
        self.check_pca1 = Radiobutton(win, text="PCA", variable=self.check_alg, value=0)
        self.check_pca1.config(font=("Arial", self.font_size))
        self.check_pca1.grid(row=2, column=2, padx=(0, 20), pady=(20, 30), sticky='w')

        # self.check_ae = BooleanVar()
        self.check_ae1 = Radiobutton(win, text="AE", variable=self.check_alg, value=1)
        self.check_ae1.config(font=("Arial", self.font_size))
        self.check_ae1.grid(row=2, column=3, padx=(0, 20), pady=(20, 30), sticky='w')

        # self.check_vae = BooleanVar()
        self.check_vae1 = Radiobutton(win, text="Variational AE", variable=self.check_alg, value=2)
        self.check_vae1.config(font=("Arial", self.font_size))
        self.check_vae1.grid(row=2, column=4, pady=(20, 30), sticky='w')

        self.btn_custom = Button(win, text="Customization", command=self.customization)
        self.btn_custom["state"] = "disabled"
        self.btn_custom.config(font=("Arial", self.font_size))
        self.btn_custom.grid(row=3, column=0, columnspan=2, padx=20, pady=(20, 30), sticky='nesw')

        self.btn_start = Button(win, text="Practice", command=self.start)
        self.btn_start["state"] = "disabled"
        self.btn_start.config(font=("Arial", self.font_size))
        self.btn_start.grid(row=4, column=0, columnspan=2, padx=20, pady=(20, 30), sticky='nesw')

        # set label for number of target remaining
        self.lbl_tgt = Label(win, text='Remaining targets: ')
        self.lbl_tgt.config(font=("Arial", self.font_size))
        self.lbl_tgt.grid(row=4, column=2, pady=(20, 30), columnspan=2, sticky='w')


        # Camera video input
        self.btn_cam = Button(win, text='Ext. Video Source', command=self.selectVideoFile)
        self.btn_cam.config(font=("Arial", self.font_size))
        self.btn_cam.grid(row=5, column=0, columnspan=2, padx=20, pady=(20, 30), sticky='nesw')
        self.ent_cam = Entry(win, width=30)
        self.ent_cam.config(font=("Arial", self.font_size))
        self.ent_cam.grid(row=5, column=2, pady=(20, 30), columnspan=3, sticky='w')
        self.btn_camClear = Button(win, text='Camera Video Source', command=self.clearVideoSource, bg='red')
        self.btn_camClear.config(font=("Arial", self.font_size))
        self.btn_camClear.grid(row=5, column=5, columnspan=2, padx=20, pady=(20, 30), sticky='nesw')

        #############################################################

        self.btn_close = Button(win, text="Close", command=win.destroy, bg="red")
        self.btn_close.config(font=("Arial", self.font_size))
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

    def calibration(self):
        # start calibration dance - collect webcam data
        self.w = tk_utils.popupWindow(self.master, "You will now start calibration.")
        self.master.wait_window(self.w.top)
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
                self.drPath = self.calibPath + 'AE/'
                train_cu = train_ae(self.calibPath, self.drPath, self.n_map_component)
                self.dr_mode = 'ae'

            # rotate, scale and offset the original features
            # implementation-dependant (depends on the workspace of
            # each variable, eg. screen space vs. joint space)
            # DEBUG
            ##self.map_to_workspace(self.calibPath, train_cu)

            self.btn_custom["state"] = "normal"
        else:
            self.w = tk_utils.popupWindow(self.master, "Perform calibration first.")
            self.master.wait_window(self.w.top)
            self.btn_map["state"] = "disabled"

    def map_to_workspace(self, drPath, train_cu):
        pass

    def customization(self):
        # check whether PCA/AE parameters have been saved
        if os.path.isfile(self.drPath + "weights1.txt"):
            # open customization window
            self.newWindow = tk.Toplevel(self.master)
            self.newWindow.geometry("1000x500")
            self.newWindow.title("Customization")
            self.app = CustomizationApplication(self.newWindow, self, drPath=self.drPath, num_joints=self.num_joints,
                                                joints=self.joints, dr_mode=self.dr_mode, video_camera_device=self.video_camera_device)
        else:
            self.w = tk_utils.popupWindow(self.master, "Compute BoMI map first.")
            self.master.wait_window(self.w.top)
            self.btn_custom["state"] = "disabled"

    def start(self):
        # implementation-specific action
        pass


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
        lockImageResults = Lock()

        # global variable accessed by main and mediapipe threads that contains the current vector of body landmarks
        
        self.body_wrap.body = np.zeros((num_joints,))  # initialize global variable
        body_calib = []  # initialize local variable (list of body landmarks during calibration)

        # start thread for OpenCV. current frame will be appended in a queue in a separate thread
        q_frame = queue.Queue()
        cal = 1  # if cal==1 (meaning during calibration) the opencv thread will display the image
        opencv_thread = Thread(target=cv_utils.get_data_from_camera, args=(cap, q_frame, r, cal, cap.get(cv2.CAP_PROP_FPS)))
        opencv_thread.start()
        print("openCV thread started in calibration.")

        # initialize thread for mediapipe operations
        mediapipe_thread = Thread(target=mediapipe_utils.mediapipe_forwardpass,
                                args=(self.current_image_data, self.body_wrap, holistic, mp_holistic, lock, q_frame, r, num_joints, joints))
        mediapipe_thread.start()
        print("mediapipe thread started in calibration.")

        # start the timer for calibration
        timer_calib.start()

        print("main thread: Starting calibration...")

        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        wind_name = "Group 12 cam"
        cv2.namedWindow(wind_name)

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles


        while not r.is_terminated:

            # safe access to the current image and results, since they can
            # be modified by the mediapipe_forwardpass thread
            with self.current_image_data.lock:
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
            cv2.imshow(wind_name, frame)

            if cv2.waitKey(1) == 27:
                break  # esc to quit

            if timer_calib.elapsed_time > calib_duration:
                r.is_terminated = True

            # get current value of body
            body_calib.append(np.copy(self.body_wrap.body))

            # update time elapsed label
            time_remaining = int((calib_duration - timer_calib.elapsed_time) / 1000)
            lbl_calib.configure(text='Calibration time: ' + str(time_remaining))
            lbl_calib.update()

            # --- Limit to 50 frames per second
            clock.tick(50)

        cv2.destroyAllWindows()
        # Stop the game engine and release the capture
        holistic.close()
        print("pose estimation object released in calibration.")
        cap.release()
        print("openCV object released in calibration.")

        # print calibration file
        body_calib = np.array(body_calib)
        if not os.path.exists(drPath):
            os.makedirs(drPath)
        np.savetxt(drPath + "Calib.txt", body_calib)

        print('Calibration finished. You can now train BoMI forward map.')
    
    

class CustomizationApplication(tk.Frame):
    """
    class that defines the customization tkinter window
    """

    def __init__(self, parent, mainTk, drPath, num_joints, joints, dr_mode, video_camera_device):

        # TODO: take this from GUI
        #self.video_camera_device = "../biorob/bomi_play.mkv" # -> 0 for the camera

        self.video_camera_device = video_camera_device
        self.current_image_data = SharedDetImage()

        self.body_wrap = BodyWrap()

        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.mainTk = mainTk
        self.drPath = drPath
        self.num_joints = num_joints
        self.joints = joints
        self.dr_mode = dr_mode
        self.font_size = 18

        self.lbl_rot = Label(parent, text='Rotation ')
        self.lbl_rot.config(font=("Arial", self.font_size))
        self.lbl_rot.grid(column=0, row=0, padx=(300, 0), pady=(40, 20), sticky='w')
        self.txt_rot = Text(parent, width=10, height=1)
        self.txt_rot.config(font=("Arial", self.font_size))
        self.txt_rot.grid(column=1, row=0, pady=(40, 20))
        self.txt_rot.insert("1.0", '0')

        self.lbl_gx = Label(parent, text='Gain x ')
        self.lbl_gx.config(font=("Arial", self.font_size))
        self.lbl_gx.grid(column=0, row=1, padx=(300, 0), pady=(40, 20), sticky='w')
        self.txt_gx = Text(parent, width=10, height=1)
        self.txt_gx.config(font=("Arial", self.font_size))
        self.txt_gx.grid(column=1, row=1, pady=(40, 20))
        self.txt_gx.insert("1.0", '1')

        self.lbl_gy = Label(parent, text='Gain y ')
        self.lbl_gy.config(font=("Arial", self.font_size))
        self.lbl_gy.grid(column=0, row=2, padx=(300, 0), pady=(40, 20), sticky='w')
        self.txt_gy = Text(parent, width=10, height=1)
        self.txt_gy.config(font=("Arial", self.font_size))
        self.txt_gy.grid(column=1, row=2, pady=(40, 20))
        self.txt_gy.insert("1.0", '1')

        self.lbl_ox = Label(parent, text='Offset x ')
        self.lbl_ox.config(font=("Arial", self.font_size))
        self.lbl_ox.grid(column=0, row=3, padx=(300, 0), pady=(40, 20), sticky='w')
        self.txt_ox = Text(parent, width=10, height=1)
        self.txt_ox.config(font=("Arial", self.font_size))
        self.txt_ox.grid(column=1, row=3, pady=(40, 20))
        self.txt_ox.insert("1.0", '0')

        self.lbl_oy = Label(parent, text='Offset y ')
        self.lbl_oy.config(font=("Arial", self.font_size))
        self.lbl_oy.grid(column=0, row=4, padx=(300, 0), pady=(40, 20), sticky='w')
        self.txt_oy = Text(parent, width=10, height=1)
        self.txt_oy.config(font=("Arial", self.font_size))
        self.txt_oy.grid(column=1, row=4, pady=(40, 20))
        self.txt_oy.insert("1.0", '0')

        self.btn_save = Button(parent, text="Save parameters", command=self.save_parameters)
        self.btn_save.config(font=("Arial", self.font_size))
        self.btn_save.grid(column=2, row=1, sticky='nesw', padx=(80, 0), pady=(40, 20))

        self.btn_start = Button(parent, text="Start", command=self.customization)
        self.btn_start.config(font=("Arial", self.font_size))
        self.btn_start.grid(column=2, row=2, sticky='nesw', padx=(80, 0), pady=(40, 20))

        self.btn_close = Button(parent, text="Close", command=parent.destroy, bg='red')
        self.btn_close.config(font=("Arial", self.font_size))
        self.btn_close.grid(column=2, row=3, sticky='nesw', padx=(80, 0), pady=(40, 20))



    # functions to retrieve values of textbox programmatically
    def retrieve_txt_rot(self):
        return self.txt_rot.get("1.0", "end-1c")

    def retrieve_txt_gx(self):
        return self.txt_gx.get("1.0", "end-1c")

    def retrieve_txt_gy(self):
        return self.txt_gy.get("1.0", "end-1c")

    def retrieve_txt_ox(self):
        return self.txt_ox.get("1.0", "end-1c")

    def retrieve_txt_oy(self):
        return self.txt_oy.get("1.0", "end-1c")

    def customization(self):
        self.initialize_customization(self.dr_mode, self.drPath, self.num_joints, self.joints, self.video_camera_device)

    def save_parameters(self):
        self.save_custom_parameters(self.drPath)
        self.parent.destroy()
        self.mainTk.btn_start["state"] = "normal"

    def initialize_customization(self, dr_mode, drPath, num_joints, joints, video_device):
        """
        initialize objects needed for online cursor control. Start all the customization threads as well
        :param self: CustomizationApplication tkinter Frame. needed to retrieve textbox values programmatically
        :param drPath: path to load the BoMI forward map
        :return:
        """

        # Create object of openCV, Reaching class and filter_butter3
        cap = cv_utils.VideoCaptureOpt(video_device)

        r = Reaching()
        reaching_functions.initialize_targets(r)
        
        filter_curs = FilterButter3("lowpass_4")
        # load BoMI forward map parameters for converting body landmarks into cursor coordinates
        map = load_bomi_map(dr_mode, drPath)

        # initialize MediaPipe Pose
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                        smooth_landmarks=False)

        # load scaling values saved after training AE for covering entire monitor workspace
        rot = pd.read_csv(drPath + 'rotation_dr.txt', sep=' ', header=None).values
        scale = pd.read_csv(drPath + 'scale_dr.txt', sep=' ', header=None).values
        scale = np.reshape(scale, (scale.shape[0],))
        off = pd.read_csv(drPath + 'offset_dr.txt', sep=' ', header=None).values
        off = np.reshape(off, (off.shape[0],))

        print("SCALE: {}".format(scale))

        # initialize lock for avoiding race conditions in threads
        lock = Lock()

        # global variable accessed by main and mediapipe threads that contains the current vector of body landmarks
        
        self.body_wrap.body = np.zeros((num_joints,))  # initialize global variable

        # start thread for OpenCV. current frame will be appended in a queue in a separate thread
        q_frame = queue.Queue()
        cal = 0
        opencv_thread = Thread(target=cv_utils.get_data_from_camera, args=(cap, q_frame, r, cal))
        opencv_thread.start()
        print("openCV thread started in customization.")

        # initialize thread for mediapipe operations
        mediapipe_thread = Thread(target=mediapipe_utils.mediapipe_forwardpass,
                                args=(self.current_image_data, self.body_wrap, holistic, mp_holistic, lock, q_frame, r, num_joints, joints))
        mediapipe_thread.start()
        print("mediapipe thread started in customization.")

        # Define some colors
        BLACK = (0, 0, 0)
        GREEN = (0, 255, 0)
        CURSOR = (0.19 * 255, 0.65 * 255, 0.4 * 255)

        pygame.init()

        # The clock will be used to control how fast the screen updates
        clock = pygame.time.Clock()

        # Open a new window
        size = (r.width, r.height)
        screen = pygame.display.set_mode(size)
        # screen = pygame.display.toggle_fullscreen()

        # -------- Main Program Loop -----------
        while not r.is_terminated:
            # --- Main event loop
            for event in pygame.event.get():  # User did something
                if event.type == pygame.QUIT:  # If user clicked close
                    r.is_terminated = True  # Flstart_reac
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:  # Pressing the space Key will click the mouse
                        pyautogui.click(r.crs_x, r.crs_y)

            if not r.is_paused:
                # Copy old cursor position
                r.old_crs_x = r.crs_x
                r.old_crs_y = r.crs_y

                # get current value of body
                r.body = np.copy(self.body_wrap.body)

                # apply BoMI forward map to body vector to obtain cursor position
                r.crs_x, r.crs_y = reaching_functions.update_cursor_position_custom(r.body, map, rot, scale, off)

                # Apply extra customization according to textbox values (try/except allows to catch invalid inputs)
                try:
                    rot_custom = int(self.retrieve_txt_rot())
                except:
                    rot_custom = 0
                try:
                    gx_custom = float(self.retrieve_txt_gx())
                except:
                    gx_custom = 1.0
                try:
                    gy_custom = float(self.retrieve_txt_gy())
                except:
                    gy_custom = 1.0
                try:
                    ox_custom = int(self.retrieve_txt_ox())
                except:
                    ox_custom = 0
                try:
                    oy_custom = int(self.retrieve_txt_oy())
                except:
                    oy_custom = 0

                
                # normalize before transformation
                r.crs_x -= r.width/2.0
                r.crs_y -= r.height/2.0

                # Applying rotation
                # edit: screen space is left-handed! remember that in rotation!
                r.crs_x, r.crs_y = reaching_functions.rotate_xy_LH([r.crs_x, r.crs_y], rot_custom)

                # Applying scale
                r.crs_x = r.crs_x * gx_custom
                r.crs_y = r.crs_y * gy_custom
                # Applying offset
                r.crs_x = r.crs_x + ox_custom + r.width/2.0
                r.crs_y = r.crs_y + oy_custom + r.height/2.0

                # Limit cursor workspace
                if r.crs_x >= r.width:
                    r.crs_x = r.width
                if r.crs_x <= 0:
                    r.crs_x = 0
                if r.crs_y >= r.height:
                    r.crs_y = 0
                if r.crs_y <= 0:
                    r.crs_y = r.height

                # Filter the cursor
                r.crs_x, r.crs_y = reaching_functions.filter_cursor(r, filter_curs)

                # Set target position to update the GUI
                reaching_functions.set_target_reaching_customization(r)

                # First, clear the screen to black. In between screen.fill and pygame.display.flip() all the draw
                screen.fill(BLACK)

                # draw cursor
                pygame.draw.circle(screen, CURSOR, (int(r.crs_x), int(r.crs_y)), r.crs_radius)

                # draw each test target
                for i in range(8):
                    tgt_x = r.tgt_x_list[r.list_tgt[i]]
                    tgt_y = r.tgt_y_list[r.list_tgt[i]]
                    pygame.draw.circle(screen, GREEN, (int(tgt_x), int(tgt_y)), r.tgt_radius, 2)

                # --- update the screen with what we've drawn.
                pygame.display.flip()

                # --- Limit to 50 frames per second
                clock.tick(50)

        # Once we have exited the main program loop, stop the game engine and release the capture
        pygame.quit()
        print("game engine object released in customization.")
        holistic.close()
        print("pose estimation object released terminated in customization.")
        cap.release()
        cv2.destroyAllWindows()
        print("openCV object released in customization.")

    def save_custom_parameters(self, drPath):
        """
        function to save customization values
        :param self: CustomizationApplication tkinter Frame. needed to retrieve textbox values programmatically
        :param drPath: path where to load the BoMI forward map
        :return:
        """
        # retrieve values stored in the textbox
        rot = int(self.retrieve_txt_rot())
        gx_custom = float(self.retrieve_txt_gx())
        gy_custom = float(self.retrieve_txt_gy())
        scale = [gx_custom, gy_custom]
        ox_custom = int(self.retrieve_txt_ox())
        oy_custom = int(self.retrieve_txt_oy())
        off = [ox_custom, oy_custom]

        # save customization values
        with open(drPath + "rotation_custom.txt", 'w') as f:
            print(rot, file=f)
        np.savetxt(drPath + "scale_custom.txt", scale)
        np.savetxt(drPath + "offset_custom.txt", off)

        print('Customization values have been saved. You can continue with practice.')


