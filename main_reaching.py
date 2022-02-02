import os
import queue
from threading import Thread, Lock
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
import scripts.blinkdetector_utils as bd_utils
import scripts.reaching_functions as reaching_functions
import scripts.mediapipe_utils as mediapipe_utils
from scripts.stopwatch import StopWatch
from scripts.filter_butter_online import FilterButter3
import scripts.compute_bomi_map as compute_bomi_map
import scripts.cv_utils as cv_utils
from scripts.JointMapper import JointMapper, CustomizationApplication
from scripts.KeyBoard_Top import KeyBoard_Top
import scripts.tk_utils as tk_utils
from scripts.tk_utils import BLACK, RED, GREEN, YELLOW, CURSOR
from scripts.reaching import Reaching, write_practice_files
from scripts.decorators import outer_control_loop
import tkinter as tk
from tkinter import Label, Text, Button
from tkinter import messagebox
import pyautogui
import pygame
import time
import math
import copy


class BoMIReaching(JointMapper):
    
    def __init__(self, win, nmap_components, *args, **kwargs):
        JointMapper.__init__(self, win, nmap_components, *args, **kwargs)
        self.app = CustomizationApplicationReaching(self)

        # set label for number of target remaining
        self.lbl_tgt = Label(win, text='', activebackground='#4682b4', bg='#abcdef')
        self.lbl_tgt.config(font=("Times", self.font_size))
        self.lbl_tgt.grid(row=4, column=2, pady=(20, 30), columnspan=2, sticky='w')


        # mouse control
        
        self.check_mouse = tk.BooleanVar()
        self.check_m1 = tk.Checkbutton(win, text="Mouse Control", variable= self.check_mouse, command=self.mouse_check, activeforeground='blue', activebackground='#4682b4', bg='#abcdef')
        self.check_m1.config(font=("Times", self.font_size))
        self.check_m1.grid(row=6, column=1, pady=(20, 30), sticky='w')

        # keyboard

        self.check_kb = tk.BooleanVar()
        self.check_kb1 = tk.Checkbutton(win, text="External Key", variable=self.check_kb, activeforeground='blue', activebackground='#4682b4', bg='#abcdef')
        self.check_kb1["state"]="disabled"
        self.check_kb1.config(font=("Times", self.font_size))
        self.check_kb1.grid(row=6, column=2, pady=(20, 30), sticky='w')

        self.refresh_rate = 30 # frames per second at max
        self.interframe_delay = 1/self.refresh_rate 

    def mouse_check(self):
        mouse_enabled = self.check_mouse.get()
        if mouse_enabled:
            self.check_kb1["state"]= "normal"

        print('Keyboard available.')
    
    def map_to_workspace(self, drPath, train_cu):
        r = Reaching()

        # save weights and biases
        if not os.path.exists(drPath):
            os.makedirs(drPath)

        rot = 0
        train_cu = reaching_functions.rotate_xy_RH(train_cu, rot)
        # Applying scale
        scale = [r.width / np.ptp(train_cu[:, 0]), r.height / np.ptp(train_cu[:, 1])]
        train_cu = train_cu * scale
        # Applying offset
        off = [r.width / 2 - np.mean(train_cu[:, 0]), r.height / 2 - np.mean(train_cu[:, 1])]
        train_cu = train_cu + off

        # save PCA scaling values
        with open(drPath + "rotation_dr.txt", 'w') as f:
            print(rot, file=f)
        np.savetxt(drPath + "scale_dr.txt", scale)
        np.savetxt(drPath + "offset_dr.txt", off)
        
        print("Screen-space affine tranformation done")

    def map_workspace2screenspace_offsetScale(self, param):
        """
        Remap the learned mapping from the boundaries of the Reaching workspace to those of the
        entire screen.
        To do that reverse the workspace mapping first.
        """

        # Note that offset and scale should suffice. Do not touch the "custom" transformation.

        r = Reaching()
        screen_width, screen_height = pyautogui.size()

        param[0] *= screen_width / r.width
        param[1] *= screen_height / r.height
        return param

    def start(self):
        # check whether customization parameters have been saved
        if os.path.isfile(self.drPath + "offset_custom.txt"):
            # open pygame and start reaching task
            self.w = tk_utils.popupWindow(self.master, "You will now start practice.")
            self.master.wait_window(self.w.top)
            if self.w.status:
                if self.check_mouse.get() == False:
                    self.start_reaching()
                else:
                    self.start_mouse_control(keyboard_bool=self.check_kb.get())
        else:
            self.w = tk_utils.popupWindow(self.master, "Perform customization first.")
            self.master.wait_window(self.w.top)
            self.btn_start["state"] = "disabled"

    @outer_control_loop()  
    def start_mouse_control(self, r=None, map=None, filter_curs=None, rot=0, scale=1, off=0, keyboard_bool=False):
        """
        function to perform online cursor control
        :param drPath: path where to load the BoMI forward map and customization values
        :param mouse_bool: tkinter Boolean value that triggers mouse control instead of reaching task
        :param keyboard_bool: tkinter Boolean value that activates the digital keyboard
        :return:
        """
        print("Mouse control active")

        # Create eye-counters objects
        left_eye = bd_utils.Eye(300, 1000) #ms
        right_eye = bd_utils.Eye(300, 1000) #ms

        # load scaling values for covering entire monitor workspace
        scale = self.map_workspace2screenspace_offsetScale(scale)
        off = self.map_workspace2screenspace_offsetScale(off)

        rot_custom, scale_custom, off_custom = compute_bomi_map.read_transform(self.drPath, "custom")

        # Keyboard Thread
        if keyboard_bool:
            # initialize keyboard operations
            kb_app = KeyBoard_Top(self.master)
            print("Keyboard interface started in calibration.")

        print("cursor control thread is about to start...")

        win_name = "Cursor Control"

        pyautogui.FAILSAFE = False
        is_mouse_down = False

        screen_width, screen_height = pyautogui.size()
        window_width = math.ceil(screen_width / 2)

        # -------- Main Program Loop -----------
        while not r.is_terminated:
            # --- Main event loop
            start_time = 0
            end_time = 0

            if not r.is_paused:
                
                start_time = time.time()
                # Copy old cursor position
                r.old_crs_x = r.crs_x
                r.old_crs_y = r.crs_y

                # get current value of body
                try:
                    # if the queue is empty just keep the previous value
                    r.body = self.body.get_nowait()
                except queue.Empty:
                    pass

                # apply BoMI forward map to body vector to obtain cursor position.
                r.crs_x, r.crs_y = reaching_functions.update_cursor_position \
                    (r.body, map, rot, scale, off, rot_custom, scale_custom, off_custom, screen_width, screen_height, dr_mode=self.dr_mode)
                # Check if the crs is bouncing against any of the 4 walls:

                # Filter the cursor
                r.crs_x, r.crs_y = reaching_functions.filter_cursor(r, filter_curs)

                # if mouse checkbox was enabled do not draw the reaching GUI,
                # only change coordinates of the computer cursor
                pyautogui.moveTo(r.crs_x, r.crs_y)

                with self.current_image_data.lock:
                    results_face = copy.deepcopy(self.current_image_data.result)
                    frame = copy.deepcopy(self.current_image_data.image)

                if results_face and results_face.face_landmarks:
                    
                    mesh_coords = bd_utils.face_landmarks_detection(frame, results_face.face_landmarks, False)
                    right_ratio, left_ratio = bd_utils.blink_ratio(frame, mesh_coords)

                    #print("#DEBUG: blink ratio L{} R{}".format(left_ratio, right_ratio))
                    
                    frame.flags.writeable = True

                    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_RGB2BGR)
                    cv2.imshow(win_name, frame)
                    cv2.moveWindow(win_name,  int(window_width + window_width / 4), 0)

                    assert isinstance(right_ratio, object)

                    left_eye.set_ratio(left_ratio)
                    right_eye.set_ratio(right_ratio)

                    if left_eye.is_blink_detected(long_blink=True) and right_eye.is_blink_detected(long_blink=True):
                        r.is_terminated = True
                        #print("#DEBUG: Both eyes closed")
                    elif left_eye.is_blink_detected():
                        #mouse.click('left')
                        pyautogui.mouseDown(button='left')
                        pyautogui.mouseUp(button='right')
                        is_mouse_down = True
                        #print("#DEBUG: Left eye closed")
                    elif right_eye.is_blink_detected():
                        pyautogui.mouseDown(button='right')
                        pyautogui.mouseUp(button='left')
                        is_mouse_down = True
                        #mouse.click('right')
                        #print("#DEBUG: Right eye closed")
                    elif is_mouse_down == True:
                        pyautogui.mouseUp(button='left')
                        pyautogui.mouseUp(button='right')
                        is_mouse_down = False

                    # update keyboard
                    self.master.update()
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit

                end_time = time.time()

                time.sleep(max(0, self.interframe_delay - (end_time - start_time)))

        # Close the keyboard
        if keyboard_bool:
            kb_app.cleanup()


    @outer_control_loop()
    def start_reaching(self, r=None, map=None, filter_curs=None, rot=0, scale=1, off=0):
        """
        function to perform online cursor control - practice
        :param drPath: path where to load the BoMI forward map and customization values
        :param check_mouse: tkinter Boolean value that triggers mouse control instead of reaching task
        :param lbl_tgt: label in the main window that shows number of targets remaining
        :return:
        """
        pygame.init()
        print("Game active")

        ############################################################

        # Open a new window
        size = (r.width, r.height)
        screen = pygame.display.set_mode(size)

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

        rot_custom, scale_custom, off_custom = compute_bomi_map.read_transform(self.drPath, "custom")

        # initialize thread for writing reaching log file
        wfile_thread = Thread(target=write_practice_files, args=(r, timer_practice))
        timer_practice.start()  # start the timer for PracticeLog
        wfile_thread.start()
        print("writing reaching log file thread started in practice.")

        print("reaching task thread is about to start...")

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
                try:
                    r.body = self.body.get_nowait()
                except queue.Empty:
                    pass

                # apply BoMI forward map to body vector to obtain cursor position.
                r.crs_x, r.crs_y = reaching_functions.update_cursor_position \
                    (r.body, map, rot, scale, off, rot_custom, scale_custom, off_custom, r.width, r.height, dr_mode=self.dr_mode)
                # Check if the crs is bouncing against any of the 4 walls:
                
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
                self.lbl_tgt.configure(text='Remaining targets: ' + str(tgt_remaining))
                self.lbl_tgt.update()

                # --- Limit to 50 frames per second
                clock.tick(self.refresh_rate)

        wfile_thread.join()
        # Once we have exited the main program loop, stop the game engine and release the capture
        pygame.quit()
        print("game engine object released in practice.")

class CustomizationApplicationReaching(CustomizationApplication):
    def __init__(self, mainTk):
        CustomizationApplication.__init__(self, mainTk)

    def generate_window(self, parent, drPath, num_joints, joints, dr_mode, video_camera_device, nmap_component):
        tk.Frame.__init__(self, parent)
        self.video_camera_device = video_camera_device
        self.parent = parent
        self.drPath = drPath
        self.num_joints = num_joints
        self.joints = joints
        self.dr_mode = dr_mode
        self.font_size = 18
        self.nmap_component = nmap_component

        self.lbl_rot = Label(parent, text='Rotation ')
        self.lbl_rot.config(font=("Times", self.font_size))
        self.lbl_rot.grid(column=0, row=0, padx=(300, 0), pady=(40, 20), sticky='w')
        self.txt_rot = Text(parent, width=10, height=1)
        self.txt_rot.config(font=("Times", self.font_size))
        self.txt_rot.grid(column=1, row=0, pady=(40, 20))
        self.txt_rot.insert("1.0", '0')

        self.lbl_gx = Label(parent, text='Gain x ')
        self.lbl_gx.config(font=("Times", self.font_size))
        self.lbl_gx.grid(column=0, row=1, padx=(300, 0), pady=(40, 20), sticky='w')
        self.txt_gx = Text(parent, width=10, height=1)
        self.txt_gx.config(font=("Times", self.font_size))
        self.txt_gx.grid(column=1, row=1, pady=(40, 20))
        self.txt_gx.insert("1.0", '1')

        self.lbl_gy = Label(parent, text='Gain y ')
        self.lbl_gy.config(font=("Times", self.font_size))
        self.lbl_gy.grid(column=0, row=2, padx=(300, 0), pady=(40, 20), sticky='w')
        self.txt_gy = Text(parent, width=10, height=1)
        self.txt_gy.config(font=("Times", self.font_size))
        self.txt_gy.grid(column=1, row=2, pady=(40, 20))
        self.txt_gy.insert("1.0", '1')

        self.lbl_ox = Label(parent, text='Offset x ')
        self.lbl_ox.config(font=("Times", self.font_size))
        self.lbl_ox.grid(column=0, row=3, padx=(300, 0), pady=(40, 20), sticky='w')
        self.txt_ox = Text(parent, width=10, height=1)
        self.txt_ox.config(font=("Times", self.font_size))
        self.txt_ox.grid(column=1, row=3, pady=(40, 20))
        self.txt_ox.insert("1.0", '0')

        self.lbl_oy = Label(parent, text='Offset y ')
        self.lbl_oy.config(font=("Times", self.font_size))
        self.lbl_oy.grid(column=0, row=4, padx=(300, 0), pady=(40, 20), sticky='w')
        self.txt_oy = Text(parent, width=10, height=1)
        self.txt_oy.config(font=("Times", self.font_size))
        self.txt_oy.grid(column=1, row=4, pady=(40, 20))
        self.txt_oy.insert("1.0", '0')

        self.btn_save = Button(parent, text="Save parameters", command=self.save_parameters)
        self.btn_save.config(font=("Times", self.font_size))
        self.btn_save.grid(column=2, row=1, sticky='nesw', padx=(80, 0), pady=(40, 20))

        self.btn_start = Button(parent, text="Start", command=self.customization)
        self.btn_start.config(font=("Times", self.font_size))
        self.btn_start.grid(column=2, row=2, sticky='nesw', padx=(80, 0), pady=(40, 20))

        self.btn_close = Button(parent, text="Close", command=parent.destroy, bg='red')
        self.btn_close.config(font=("Times", self.font_size))
        self.btn_close.grid(column=2, row=3, sticky='nesw', padx=(80, 0), pady=(40, 20))

        cap = cv_utils.VideoCaptureOpt(video_camera_device)

        self.refresh_rate = math.ceil(cap.get(cv2.CAP_PROP_FPS)) # frames per second at max
        cap.release()
        if self.refresh_rate <= 0:
            self.refresh_rate = 30
        self.interframe_delay = 1/self.refresh_rate


    # function to close the main window
    def on_closing(self):

        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.destroy()

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
        self.initialize_customization()

    def save_parameters(self):
        self.save_custom_parameters(self.drPath)
        self.parent.destroy()
        self.mainTk.btn_start["state"] = "normal"

    @outer_control_loop()
    def initialize_customization(self, r=None, map=None, filter_curs=None, rot=0, scale=1, off=0, drPath=""):
        """
        initialize objects needed for online cursor control. Start all the customization threads as well
        :param self: CustomizationApplication tkinter Frame. needed to retrieve textbox values programmatically
        :param drPath: path to load the BoMI forward map
        :return:
        """
        reaching_functions.initialize_targets(r)

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
                try:
                    r.body = self.body.get_nowait()
                except queue.Empty:
                    pass

                # apply BoMI forward map to body vector to obtain cursor position
                r.crs_x, r.crs_y = reaching_functions.update_cursor_position_custom(r.body, map, rot, scale, off, dr_mode=self.dr_mode)

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
                clock.tick(self.refresh_rate)

       
        # Once we have exited the main program loop, stop the game engine and release the capture
        pygame.quit()
        print("game engine object released in customization.")
       

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

# MAIN
if __name__ == "__main__":
    # initialize tkinter window
    win = tk_utils.win_init("BoMi Settings")

    obj = BoMIReaching(win=win, nmap_components=2)

    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            win.destroy()

    win.protocol("WM_DELETE_WINDOW", on_closing)

    # initiate Tkinter mainloop
    win.mainloop()
