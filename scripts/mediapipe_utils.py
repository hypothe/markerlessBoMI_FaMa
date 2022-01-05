import math
import mediapipe as mp
from scripts.tk_utils import GREEN
import cv2
import queue
import numpy as np
import time


def mediapipe_forwardpass(image_data, body_wrap, holistic, mp_holistic, lock, q_frame, r, num_joints, joints, fps=120, mp_face_mesh=None):
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
	_ = q_frame.get(block=True, timeout=10.0)


	while keep_reading_queue and not r.is_terminated:
		start_time = 0
		end_time = 0
		elapsed_time = 0

		if not r.is_paused:
			start_time = time.time()
			# try:
			# get current frame from thread 
			try:
				# wait as the queue might be empty just due to fast
				# consumption, not for the end of the stream
				curr_frame = q_frame.get(block=True, timeout=1.0)
				#if curr_frame is None:
				#	raise queue.Empty
			except queue.Empty:
				keep_reading_queue = False # exit if the queue is empty (and has been so for >1 sec)
			else:
				if curr_frame is None:
					continue

				body_list = []
				debug_frame_analyzed+=1
				with image_data.lock:
					# Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
					image_data.image_id += 1
					image_data.image = cv2.cvtColor(cv2.flip(curr_frame, 1), cv2.COLOR_BGR2RGB)
					# To improve performance, optionally mark the image as not writeable to pass by reference.
					image_data.image.flags.writeable = False
					image_data.result = holistic.process(image_data.image)

					if mp_face_mesh is not None:
						image_data.result_face = mp_face_mesh.process(image_data.image)
						#print("result face {}".format(image_data.result_face.multi_face_landmarks))

				if not image_data.result.pose_landmarks:
					continue
				if joints[0, 0] == 1:
					body_list.append(image_data.result.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x)
					body_list.append(image_data.result.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y)
				if joints[1, 0] == 1:
					body_list.append(image_data.result.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].x)
					body_list.append(image_data.result.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].y)
					body_list.append(image_data.result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].x)
					body_list.append(image_data.result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].y)
				if joints[2, 0] == 1:
					body_list.append(image_data.result.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x)
					body_list.append(image_data.result.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y)
					body_list.append(image_data.result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x)
					body_list.append(image_data.result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y)
				if joints[3, 0] == 1 or joints[4, 0] == 1:
					body_list.append(image_data.result.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x)
					body_list.append(image_data.result.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y)
				if joints[4, 0] == 1:
					body_list.append(image_data.result.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].x)
					body_list.append(image_data.result.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].y)
					body_list.append(image_data.result.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].x)
					body_list.append(image_data.result.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y)
					body_list.append(image_data.result.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].x)
					body_list.append(image_data.result.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].y)
					body_list.append(image_data.result.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].x)
					body_list.append(image_data.result.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].y)

				body_mp = np.array(body_list)
				#q_frame.queue.clear()

				with lock:
					body_wrap.body = np.copy(body_mp)

				# consume the source at the correct frequecy      
				#end_time = time.time()

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


def landmarksDetection(img, results, draw=False):
    # landmark detection function

    img_height, img_width = img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]
    if draw:
        [cv2.circle(img, p, 2, GREEN, -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks
    return mesh_coord