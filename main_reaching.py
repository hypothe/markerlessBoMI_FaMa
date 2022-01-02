import os
import queue
from threading import Thread, Lock
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
import scripts.reaching_functions as reaching_functions
import scripts.mediapipe_utils as mediapipe_utils
from scripts.stopwatch import StopWatch
from scripts.filter_butter_online import FilterButter3
import scripts.compute_bomi_map as compute_bomi_map
import scripts.cv_utils as cv_utils
from scripts.JointMapper import JointMapper
import scripts.tk_utilities as tk_utils
from scripts.reaching import Reaching, write_practice_files
import tkinter as tk
import pyautogui
import pygame
import mouse


class BoMIReaching(JointMapper):
    
    def __init__(self, win, n_map_components, *args, **kwargs):
        JointMapper.__init__(self, win, n_map_components, *args, **kwargs)

        # mouse control
        
        self.check_mouse = tk.BooleanVar()
        self.check_m1 = tk.Checkbutton(win, text="Mouse Control", variable=self.check_mouse)
        self.check_m1.config(font=("Arial", self.font_size))
        self.check_m1.grid(row=6, column=1, pady=(20, 30), sticky='w')

    def map_to_workspace(self, drPath, train_cu):
        r = Reaching()

        rot = 0
        train_cu = reaching_functions.rotate_xy_RH(train_cu, rot)
        # Applying scale
        scale = [r.width / np.ptp(train_cu[:, 0]), r.height / np.ptp(train_cu[:, 1])]
        train_cu = train_cu * scale
        # Applying offset
        off = [r.width / 2 - np.mean(train_cu[:, 0]), r.height / 2 - np.mean(train_cu[:, 1])]
        train_cu = train_cu + off

        # save PCA scaling values
        print("CALIBPATH {}".format(drPath))
        with open(drPath + "rotation_dr.txt", 'w') as f:
            print(rot, file=f)
        np.savetxt(drPath + "scale_dr.txt", scale)
        np.savetxt(drPath + "offset_dr.txt", off)
        
        print("Screen-space affine tranformation done")

    def start(self):
        # check whether customization parameters have been saved
        if os.path.isfile(self.drPath + "offset_custom.txt"):
            # open pygame and start reaching task
            self.w = tk_utils.popupWindow(self.master, "You will now start practice.")
            self.master.wait_window(self.w.top)
            self.start_reaching(self.drPath, self.lbl_tgt, self.num_joints, self.joints, self.dr_mode,
                            self.check_mouse.get(), self.video_camera_device)
        else:
            self.w = tk_utils.popupWindow(self.master, "Perform customization first.")
            self.master.wait_window(self.w.top)
            self.btn_start["state"] = "disabled"
            

    def start_reaching(self, drPath, lbl_tgt, num_joints, joints, dr_mode, mouse_bool, video_device=0):
        """
        function to perform online cursor control - practice
        :param drPath: path where to load the BoMI forward map and customization values
        :param check_mouse: tkinter Boolean value that triggers mouse control instead of reaching task
        :param lbl_tgt: label in the main window that shows number of targets remaining
        :return:
        """
        pygame.init()
        if mouse_bool == True:
            print("Mouse control active")
        else:
            print("Game active")

        ############################################################

        # Define some colors
        BLACK = (0, 0, 0)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        YELLOW = (255, 255, 0)
        CURSOR = (0.19 * 255, 0.65 * 255, 0.4 * 255)

        # Create object of openCV, Reaching class and filter_butter3
        cap = cv_utils.VideoCaptureOpt(video_device)

        r = Reaching()
        filter_curs = FilterButter3("lowpass_4")

        # Open a new window
        size = (r.width, r.height)
        if mouse_bool == False:
            screen = pygame.display.set_mode(size)

        else:
            print("Control the mouse")
        # screen = pygame.display.toggle_fullscreen()

        # The clock will be used to control how fast the screen updates
        clock = pygame.time.Clock()

        # Initialize stopwatch for counting time elapsed in the different states of the reaching
        timer_enter_tgt = StopWatch()
        timer_start_trial = StopWatch()
        timer_practice = StopWatch()

        # initialize targets and the reaching log file header
        reaching_functions.initialize_targets(r)
        reaching_functions.write_header(r)

        # load BoMI forward map parameters for converting body landmarks into cursor coordinates
        map = compute_bomi_map.load_bomi_map(dr_mode, drPath)

        # initialize MediaPipe Pose
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                        smooth_landmarks=False)

        # load scaling values for covering entire monitor workspace
        rot_dr = pd.read_csv(drPath + 'rotation_dr.txt', sep=' ', header=None).values
        scale_dr = pd.read_csv(drPath + 'scale_dr.txt', sep=' ', header=None).values
        scale_dr = np.reshape(scale_dr, (scale_dr.shape[0],))
        off_dr = pd.read_csv(drPath + 'offset_dr.txt', sep=' ', header=None).values
        off_dr = np.reshape(off_dr, (off_dr.shape[0],))
        rot_custom = pd.read_csv(drPath + 'rotation_custom.txt', sep=' ', header=None).values
        scale_custom = pd.read_csv(drPath + 'scale_custom.txt', sep=' ', header=None).values
        scale_custom = np.reshape(scale_custom, (scale_custom.shape[0],))
        off_custom = pd.read_csv(drPath + 'offset_custom.txt', sep=' ', header=None).values
        off_custom = np.reshape(off_custom, (off_custom.shape[0],))

        # initialize lock for avoiding race conditions in threads
        lock = Lock()

        # global variable accessed by main and mediapipe threads that contains the current vector of body landmarks
        self.body_wrap.body = np.zeros((num_joints,))  # initialize global variable

        # start thread for OpenCV. current frame will be appended in a queue in a separate thread
        q_frame = queue.Queue()
        cal = 0
        opencv_thread = Thread(target=cv_utils.get_data_from_camera, args=(cap, q_frame, r, cal))
        opencv_thread.start()
        print("openCV thread started in practice.")

        # initialize thread for mediapipe operations
        mediapipe_thread = Thread(target=mediapipe_utils.mediapipe_forwardpass,
                                args=(self.current_image_data, self.body_wrap, holistic, mp_holistic, lock, q_frame, r, num_joints, joints))
        mediapipe_thread.start()
        print("mediapipe thread started in practice.")

        # initialize thread for writing reaching log file
        wfile_thread = Thread(target=write_practice_files, args=(r, timer_practice))
        timer_practice.start()  # start the timer for PracticeLog
        wfile_thread.start()
        print("writing reaching log file thread started in practice.")

        print("cursor control thread is about to start...")

        # -------- Main Program Loop -----------
        while not r.is_terminated:
            # --- Main event loop
            for event in pygame.event.get():  # User did something
                if event.type == pygame.QUIT:  # If user clicked close
                    r.is_terminated = True  # Flag that we are done so we exit this loop
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_x:  # Pressing the x Key will quit the game
                        r.is_terminated = True
                    if event.key == pygame.K_p:  # Pressing the p Key will pause/resume the game
                        reaching_functions.pause_acquisition(r, timer_practice)
                    if event.key == pygame.K_SPACE:  # Pressing the space Key will click the mouse
                        pyautogui.click(r.crs_x, r.crs_y)

            if not r.is_paused:
                # Copy old cursor position
                r.old_crs_x = r.crs_x
                r.old_crs_y = r.crs_y

                # get current value of body
                r.body = np.copy(self.body_wrap.body)

                # apply BoMI forward map to body vector to obtain cursor position.
                r.crs_x, r.crs_y = reaching_functions.update_cursor_position \
                    (r.body, map, rot_dr, scale_dr, off_dr, rot_custom, scale_custom, off_custom, r.width, r.height)
                # Check if the crs is bouncing against any of the 4 walls:

                if mouse_bool == False:

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

                # if mouse checkbox was enabled do not draw the reaching GUI,
                # only change coordinates of the computer cursor
                if mouse_bool == True:

                    # pyautogui.move(r.crs_x, r.crs_y, pyautogui.FAILSAFE)
                    mouse.move(r.crs_x, r.crs_y, absolute=True, duration=1/50)

                else:

                    # Set target position to update the GUI
                    reaching_functions.set_target_reaching(r)
                    # First, clear the screen to black. In between screen.fill and pygame.display.flip() all the draw
                    screen.fill(BLACK)
                    # Do not show the cursor in the blind trials when the cursor is outside the home target
                    if not r.is_blind:
                        # draw cursor
                        pygame.draw.circle(screen, CURSOR, (int(r.crs_x), int(r.crs_y)), r.crs_radius)

                    # draw target. green if blind, state 0 or 1. yellow if notBlind and state 2
                    if r.state == 0:  # green
                        pygame.draw.circle(screen, GREEN, (int(r.tgt_x), int(r.tgt_y)), r.tgt_radius, 2)
                    elif r.state == 1:
                        pygame.draw.circle(screen, GREEN, (int(r.tgt_x), int(r.tgt_y)), r.tgt_radius, 2)
                    elif r.state == 2:  # yellow
                        if r.is_blind:  # green again if blind trial
                            pygame.draw.circle(screen, GREEN, (int(r.tgt_x), int(r.tgt_y)), r.tgt_radius, 2)
                        else:  # yellow if not blind
                            pygame.draw.circle(screen, YELLOW, (int(r.tgt_x), int(r.tgt_y)), r.tgt_radius, 2)

                    # Display scores:
                    font = pygame.font.Font(None, 80)
                    text = font.render(str(r.score), True, RED)
                    screen.blit(text, (1250, 10))

                    # --- update the screen with what we've drawn.
                    pygame.display.flip()

                    # After showing the cursor, check whether cursor is in the target
                    reaching_functions.check_target_reaching(r, timer_enter_tgt)
                    # Then check if cursor stayed in the target for enough time
                    reaching_functions.check_time_reaching(r, timer_enter_tgt, timer_start_trial, timer_practice)

                    # update label with number of targets remaining
                    tgt_remaining = 248 - r.trial + 1
                    lbl_tgt.configure(text='Remaining targets: ' + str(tgt_remaining))
                    lbl_tgt.update()

                # --- Limit to 50 frames per second
                clock.tick(50)

        # Once we have exited the main program loop, stop the game engine and release the capture
        pygame.quit()
        print("game engine object released in practice.")
        # pose.close()
        holistic.close()
        print("pose estimation object released in practice.")
        cap.release()
        cv2.destroyAllWindows()
        print("openCV object released in practice.")

# MAIN
if __name__ == "__main__":
    # initialize tkinter window
    win = tk_utils.win_init("BoMi Settings")

    obj = BoMIReaching(win=win, n_map_components=2)

    # initiate Tkinter mainloop
    win.mainloop()
