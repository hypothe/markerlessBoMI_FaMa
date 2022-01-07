import math
import mediapipe as mp
from scripts.tk_utils import GREEN
import cv2
import queue
import numpy as np
import time


def mediapipe_forwardpass(image_data, q_body, holistic, mp_holistic, lock, q_frame, r, num_joints, joints, fps=120, mp_face_mesh=None):
	"""
	function that runs in the thread for estimating pose online
	:param pose: object of Mediapipe class used to predict poses
	:param mp_pose: object of Mediapipe class for extracting body landmarks
	:param lock: lock for avoiding race condition on body vector
	:param q_frame: queue where to append current webcam frame
	:param r: object of Reaching class
	:return:
	"""
	# check for possible incorrectly represented fps values
	# (<FIX> some camera reporting fps = 0)
	if fps <= 0:
		fps = 120

	interframe_delay = float(1.0/fps)

	# DEBUG
	debug_frame_analyzed = 0
	debug_frame_skipped = 0

	keep_reading_queue = True

	# This first timeout waits for up to 10 seconds until the video source is up
	# and put something in the queue
	# NOTE: this eats up the first element in said queue, not a drama here
	# but neither so elegant it won't come bite us back later on.
	try:
		_ = q_frame.get(block=True, timeout=10.0)
	except queue.Empty:
		print("WARN: no image retrieved from the device after 10 seconds. Is it connected?")
		r.is_terminated = True


	while keep_reading_queue and not r.is_terminated:
		start_time = 0
		end_time = 0
		elapsed_time = 0

		if not r.is_paused:
			start_time = time.time()
			# get current frame from thread 
			try:
				# wait as the queue might be empty just due to fast
				# consumption, not for the end of the stream
				curr_frame = q_frame.get(block=True, timeout=1.0)

			except queue.Empty:
				keep_reading_queue = False # exit if the queue is empty (and has been so for >1 sec)
			else:
				if curr_frame is None:
					continue

				body_list = []
				debug_frame_analyzed+=1

				result = None
				with image_data.lock:
					# Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
					image_data.image_id = debug_frame_analyzed
					image_data.image = cv2.cvtColor(cv2.flip(curr_frame, 1), cv2.COLOR_BGR2RGB)
					# To improve performance, optionally mark the image as not writeable to pass by reference.
					image_data.image.flags.writeable = False
					image_data.result = result = holistic.process(image_data.image)

					# NOTE: this should not be needed as the holistic model already provides face landmarks' detection
					if mp_face_mesh is not None:
						image_data.result.face_landmarks = mp_face_mesh.process(image_data.image).multi_face_landmarks[0]

				if not result.pose_landmarks:
					continue
				if joints[0, 0] == 1:
					try:
						body_list.append(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x)
						body_list.append(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y)
					except AttributeError:
						# silent fail is landmark was not in view, ignore even good landmarks in this pass
						continue
				if joints[1, 0] == 1:
					try:
						body_list.append(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].x)
						body_list.append(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].y)
						body_list.append(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].x)
						body_list.append(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].y)
					except AttributeError:
						continue
				if joints[2, 0] == 1:
					try:
						body_list.append(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x)
						body_list.append(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y)
						body_list.append(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x)
						body_list.append(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y)
					except AttributeError:
						continue
				if joints[3, 0] == 1 or joints[4, 0] == 1:
					try:
						body_list.append(result.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x)
						body_list.append(result.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y)
					except AttributeError:
						continue
				if joints[4, 0] == 1:
					try:
						body_list.append(result.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].x)
						body_list.append(result.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].y)
						body_list.append(result.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].x)
						body_list.append(result.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y)
						body_list.append(result.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].x)
						body_list.append(result.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].y)
						body_list.append(result.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].x)
						body_list.append(result.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].y)
					except AttributeError:
						continue

				try:
					q_body.put(np.array(body_list), block=False)
				except queue.Full:
					# silent failing in case of queue not being read
					# (generally happens when exiting)
					pass

				# Skip the next n frames, in order to keep the same fps as the
				# video source
				elapsed_time = time.time() - start_time
				frames_to_be_skipped = math.ceil(elapsed_time*fps)
				#print('#DEBUG-PIPE: time elapsed {} frames to skip {} q_size {}'.format(elapsed_time, frames_to_be_skipped, q_frame.qsize()))
				for i in range(frames_to_be_skipped):
					try:
						_ = q_frame.get(block=False)
					except queue.Empty:
						pass
					else:
						debug_frame_skipped += 1
		
		time.sleep(max(0, interframe_delay - elapsed_time))

	# set the rec phase to be considered ended only after mediapipe finished consuming the input queue
	r.is_terminated = True

	#print('#DEBUG: frame processed {} frame skipped {} total frames {}'.format(debug_frame_analyzed, debug_frame_skipped, debug_frame_analyzed + debug_frame_skipped))
	print('Mediapipe_forwardpass thread terminated.')
