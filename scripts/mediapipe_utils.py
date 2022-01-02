import mediapipe as mp
import cv2
import queue
import numpy as np

def mediapipe_forwardpass(image_data, body, holistic, mp_holistic, lock, q_frame, r, num_joints, joints):
	"""
	function that runs in the thread for estimating pose online
	:param pose: object of Mediapipe class used to predict poses
	:param mp_pose: object of Mediapipe class for extracting body landmarks
	:param lock: lock for avoiding race condition on body vector
	:param q_frame: queue where to append current webcam frame
	:param r: object of Reaching class
	:return:
	"""
	while not r.is_terminated:
			if not r.is_paused:
					# try:
					# get current frame from thread 
					try:
							# wait as the queue might be empty just due to fast
							# consumption, not for the end of the stream
							curr_frame = q_frame.get(block=True, timeout=1.0)
							body_list = []
					except queue.Empty:
							pass
					else:
							with image_data.lock:
									# Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
									image_data.image = cv2.cvtColor(cv2.flip(curr_frame, 1), cv2.COLOR_BGR2RGB)
									# To improve performance, optionally mark the image as not writeable to pass by reference.
									image_data.image.flags.writeable = False
									image_data.result = holistic.process(image_data.image)

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
							q_frame.queue.clear()
							with lock:
									body = np.copy(body_mp)

	print('Mediapipe_forwardpass thread terminated.')