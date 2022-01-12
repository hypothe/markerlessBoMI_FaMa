import os
import queue
from threading import Thread, Lock
from tkinter.constants import END, INSERT
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
import tkinter as tk
from tkinter import Label, Text, Button
from tkinter import messagebox
import pyautogui
import pygame
import time
import math
import copy


class BoMIMechanism(JointMapper):
	def __init__(self, win, n_map_components, *args, **kwargs):
		JointMapper.__init__(self, win, n_map_components, *args, **kwargs)
		
		# DEBUG
		video_name = "/root/videos/calib_bomi.mp4"
		self.video_camera_device = video_name
		self.ent_cam.insert(INSERT, video_name)
		# end - DEBUG

		self.app = CustomizationApplicationMechanism(self)

	def map_to_workspace(self, drPath, train_cu):
		# retrieve transformation from generated values, to normalize 
		# the forward pass later
		# Here the joints are assumed all rotational, full range [-pi, pi]
		rot = 0
		scale = []
		off = []

		if not os.path.exists(drPath):
			os.makedirs(drPath)

		# (x - mi)/(ma - mi)
		# [-pi, pi]
		for i in range(np.size(train_cu, axis=1)):
			scale.append(2.0*math.pi/np.ptp(train_cu[:, i]))
			off.append(-math.pi - 2.0*math.pi*np.min(train_cu[:, i])/np.ptp(train_cu[:, i]))

		# save PCA scaling values
		with open(drPath + "rotation_dr.txt", 'w') as f:
				print(rot, file=f)
		np.savetxt(drPath + "scale_dr.txt", scale)
		np.savetxt(drPath + "offset_dr.txt", off)
		
		print("Joint-space affine tranformation done")


class CustomizationApplicationMechanism(CustomizationApplication):
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

		# TODO: ask for the number of joints, for now stuck with 3

		self.lbl_g = []
		self.txt_g = []
		self.lbl_o = []
		self.txt_o = []

		for i in range(self.nmap_component):
			self.lbl_g.append(Label(parent, text='Gain {} '.format(i)))
			self.lbl_g[-1].config(font=("Arial", self.font_size))
			self.lbl_g[-1].grid(column=i, row=0, padx=(40, 20), pady=(40, 20), sticky='w')
			self.txt_g.append(Text(parent, width=10, height=1))
			self.txt_g[-1].config(font=("Arial", self.font_size))
			self.txt_g[-1].grid(column=i, row=1, pady=(40, 20))
			self.txt_g[-1].insert("1.0", '1')

			self.lbl_o.append(Label(parent, text='Offset {} '.format(i)))
			self.lbl_o[-1].config(font=("Arial", self.font_size))
			self.lbl_o[-1].grid(column=i, row=2, padx=(40, 20), pady=(40, 20), sticky='w')
			self.txt_o.append(Text(parent, width=10, height=1))
			self.txt_o[-1].config(font=("Arial", self.font_size))
			self.txt_o[-1].grid(column=i, row=3, pady=(40, 20))
			self.txt_o[-1].insert("1.0", '0')


		self.btn_save = Button(parent, text="Save parameters", command=self.save_parameters)
		self.btn_save.config(font=("Arial", self.font_size))
		self.btn_save.grid(column=5, row=1, sticky='nesw', padx=(80, 0), pady=(40, 20))

		self.btn_start = Button(parent, text="Start", command=self.customization)
		self.btn_start.config(font=("Arial", self.font_size))
		self.btn_start.grid(column=5, row=2, sticky='nesw', padx=(80, 0), pady=(40, 20))

		self.btn_close = Button(parent, text="Close", command=parent.destroy, bg='red')
		self.btn_close.config(font=("Arial", self.font_size))
		self.btn_close.grid(column=5, row=3, sticky='nesw', padx=(80, 0), pady=(40, 20))


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

	def retrieve_txt_g(self):
		g = []
		for i in range(len(self.txt_g)):
			try:
				g.append(float(self.txt_g[i].get("1.0", "end-1c")))
			except ValueError:
				g.append(1)
		return g

	def retrieve_txt_o(self):
		o = []
		for i in range(len(self.txt_o)):
			try:
				o.append(float(self.txt_o[i].get("1.0", "end-1c")))
			except ValueError:
				o.append(0)
		return o

	def customization(self):
		self.initialize_customization(self.dr_mode, self.drPath, self.num_joints, self.joints, self.video_camera_device)

	# ---- # Testing Interface # ---- #
	def initialize_customization(self, dr_mode, drPath, num_joints, joints, video_device):
		"""
		Allow the user to test out and customize the BoMI mapping.
		:param self: CustomizationApplicationMechanism tkinter Frame. needed to retrieve textbox values programmatically
		:param dr_mode:
		:param drPath: path to load the BoMI forward map
		:param num_joints: 
		:param joints:
		:param video device:
		:return:
		"""
		
		# Create object of openCV, Reaching class and filter_butter3
		cap = cv_utils.VideoCaptureOpt(video_device)

		r = Reaching()
		map = compute_bomi_map.load_bomi_map(dr_mode, drPath)
		
		filter_curs = FilterButter3("lowpass_4", nc=self.nmap_component)
		
		# initialize MediaPipe Pose
		mp_holistic = mp.solutions.holistic
		holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,
																		smooth_landmarks=False)


		rot, scale, off = compute_bomi_map.read_transform(drPath, "dr")
		
		# initialize lock for avoiding race conditions in threads
		lock = Lock()

		self.body = queue.Queue(maxsize=1)

		# start thread for OpenCV. current frame will be appended in a queue in a separate thread
		q_frame = queue.Queue()
		opencv_thread = Thread(target=cv_utils.get_data_from_camera, args=(cap, q_frame, r, None))
		opencv_thread.start()
		print("openCV thread started in customization.")

		# initialize thread for mediapipe operations
		mediapipe_thread = Thread(target=mediapipe_utils.mediapipe_forwardpass,
														args=(self.current_image_data, self.body, holistic, mp_holistic, lock, q_frame, r, num_joints, joints, cap.get(cv2.CAP_PROP_FPS), None))
		mediapipe_thread.start()
		print("mediapipe thread started in customization.")

		# TODO: show n sliders, one for each joint, (simulate them with balls and lines if needed)
		# making them go up or down depending on the joint moved by the user.

		pygame.init()

		# The clock will be used to control how fast the screen updates
		clock = pygame.time.Clock()

		# Open a new window
		size = (r.width, r.height)
		screen = pygame.display.set_mode(size)
		joints_display_displacement = None #[r.height]*self.nmap_component
		slider_length = 300 # px

		while not r.is_terminated:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					r.is_terminated = True
			
			if r.is_paused: # wait and skip
				clock.tick(self.refresh_rate)
				break
		
			 # get current value of body
			try:
				r.body = self.body.get_nowait()
			except queue.Empty:
				pass

			# -- Mapping --
			scale_custom = [s*slider_length/(2.0*math.pi) for s in self.retrieve_txt_g()]
			off_custom   = [s + r.height/2.0 for s in self.retrieve_txt_o()]
			joint_values = reaching_functions.get_mapped_values(r.body, map, \
																													rot, scale, off, \
																													0, scale_custom, off_custom,\
																													joints_display_displacement)

			# -- Saturation --
			joint_values = [reaching_functions.saturate(j, r.height/2 - slider_length/2, r.height/2 + slider_length/2) for j in joint_values]

			# -- Filtering --
			for i in range(self.nmap_component):
				filter_curs.update_cursor(joint_values[i], i)

			joint_values = [filter_curs.filtered_value[i] for i in range(self.nmap_component)]

			# -- Display --
			screen.fill(BLACK)
			
			for i in range(self.nmap_component):
				pygame.draw.line(screen, bd_utils.WHITE, (int(r.width * (i+1)/(self.nmap_component+1)), r.height/2-slider_length/2 ),\
					(int(r.width * (i+1)/(self.nmap_component+1)), r.height/2+slider_length/2 ))
				
				pygame.draw.circle(screen, CURSOR, (int(r.width * (i+1)/(self.nmap_component+1)), \
																						int(joint_values[i])), r.crs_radius)

			pygame.display.flip()

			clock.tick(self.refresh_rate)

		opencv_thread.join()
		mediapipe_thread.join()
		# Once we have exited the main program loop, stop the game engine and release the capture
		pygame.quit()
		print("game engine object released in customization.")
		holistic.close()
		print("pose estimation object released terminated in customization.")
		cap.release()
		cv2.destroyAllWindows()
		print("openCV object released in customization.")


	# ---- # Param Saving # ---- #
	def save_parameters(self):
		self.save_custom_parameters(self.drPath)
		self.parent.destroy()
		self.mainTk.btn_start["state"] = "normal"

	def save_custom_parameters(self, drPath):
		"""
		function to save customization values
		:param self: CustomizationApplication tkinter Frame. needed to retrieve textbox values programmatically
		:param drPath: path where to load the BoMI forward map
		:return:
		"""
		# retrieve values stored in the textbox
		scale = self.retrieve_txt_g()
		off   = self.retrieve_txt_o()

		# save customization values
		with open(drPath + "rotation_custom.txt", 'w') as f:
				print('', file=f)
		np.savetxt(drPath + "scale_custom.txt", scale)
		np.savetxt(drPath + "offset_custom.txt", off)

		print('Customization values have been saved. You can continue with practice.')

	

# MAIN
if __name__ == "__main__":
		# initialize tkinter window
		win = tk_utils.win_init("BoMi Settings")

		

		obj = BoMIMechanism(win=win, n_map_components=3)

		# initiate Tkinter mainloop
		win.mainloop()