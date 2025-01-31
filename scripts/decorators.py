import functools
from unittest import skip
from scripts.reaching import Reaching
import scripts.compute_bomi_map as compute_bomi_map
import scripts.cv_utils as cv_utils
import scripts.mediapipe_utils as mediapipe_utils
from scripts.filter_butter_online import FilterButter3
#from scripts.JointMapper import JointMapper, CustomizationApplication
import mediapipe as mp
from threading import Thread, Lock
import queue
import pygame
import cv2

def outer_control_loop(qbody_maxsize=1, is_calib = False):
	"""
	Wrapper of decorator with parameters.
	It instanciates the opencv and mediapipe threads
	"""
	def outer_wrapper(func):
		@functools.wraps(func)
		def wrapper(bomi, *args, **kwargs):
			# TODO: improve readability. Like, by a lot.

			#assert isinstance(bomi, (JointMapper, CustomizationApplication))

			video_device = bomi.video_camera_device
			dr_mode = bomi.dr_mode
			drPath = bomi.drPath
			num_joints = bomi.num_joints
			joints = bomi.joints

			# Create object of openCV, Reaching class and filter_butter3
			cap = cv_utils.VideoCaptureOpt(video_device)
			
			if not cap.isOpened():
				raise IOError("Cannot open webcam")

			bomi.refresh_rate = cap.get(cv2.CAP_PROP_FPS)
			if bomi.refresh_rate <= 0:
				bomi.refresh_rate = 30
			bomi.interframe_delay = 1/bomi.refresh_rate 

			r = Reaching()

			map = None
			rot, scale, off = 0, 1, 0
			if not is_calib:
				map = compute_bomi_map.load_bomi_map(dr_mode, drPath)
				rot, scale, off = compute_bomi_map.read_transform(drPath, "dr")

			filter_curs = FilterButter3("lowpass_4", nc=bomi.nmap_component)
			
			# initialize MediaPipe Pose
			mp_holistic = mp.solutions.holistic
			holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,
																			smooth_landmarks=False)


			
			# initialize lock for avoiding race conditions in threads
			lock = Lock()

			bomi.body = queue.Queue(maxsize=qbody_maxsize)

			# start thread for OpenCV. current frame will be appended in a queue in a separate thread
			q_frame = queue.Queue()

			cv_timeout = is_calib and bomi.calib_duration or None
			opencv_thread = Thread(target=cv_utils.get_data_from_camera, args=(cap, q_frame, r, cv_timeout))
			opencv_thread.start()
			print("openCV thread started.")

			# initialize thread for mediapipe operations
			mediapipe_thread = Thread(target=mediapipe_utils.mediapipe_forwardpass,
															args=(bomi.current_image_data, bomi.body, holistic, mp_holistic, lock, q_frame, r, num_joints, joints, cap.get(cv2.CAP_PROP_FPS), None))
			mediapipe_thread.start()
			print("mediapipe thread started.")

			# ---- #
			func(bomi, r, map, filter_curs, rot, scale, off, *args, **kwargs)
			# ---- #

			opencv_thread.join()
			mediapipe_thread.join()
			
			holistic.close()
			print("pose estimation object released terminated.")
			cap.release()
			cv2.destroyAllWindows()
			print("openCV object released.")

		return wrapper
	return outer_wrapper