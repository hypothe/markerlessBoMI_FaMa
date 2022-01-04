import cv2
import math

def check_available_videoio_backend(queryBackendName):
    availableBackends = [cv2.videoio_registry.getBackendName(b) for b in cv2.videoio_registry.getCameraBackends()]
    if queryBackendName in availableBackends:
        return True
    return False

def VideoCaptureOpt(videoSource):
    videoioBackend =  check_available_videoio_backend('DSHOW') and cv2.CAP_DSHOW or cv2.CAP_ANY
    try:
        cap = cv2.VideoCapture(videoSource, videoioBackend)#, cv2_OPENMODE)
    except:
        print("Unable to open source at {}, defaulting to camera 0".format(videoSource))
        cap = cv2.VideoCapture(0, videoioBackend)
    return cap


def get_data_from_camera(cap, q_frame, r, duration=None):
    '''
    function that runs in the thread to capture current frame and put it into the queue
    :param cap: object of OpenCV class
    :param q_frame: queue to store current frame
    :param r: object of Reaching class
    :return:
    '''

    # Check the timing from within this function
    # this allow to stop when either a camera has been recorded for at least 'duration' seconds
    # or when an input video ended (which reading generally takes less time)

    frames_read = 0

    keep_reading_cap = True

    fps_source = cap.get(cv2.CAP_PROP_FPS)
    frames_lenght_source = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps_source <= 0:
        fps_source = 30

    if duration is not None:
        frames_to_read = min(math.floor(duration*fps_source/1000), frames_lenght_source)
    else:
        frames_to_read = frames_lenght_source > 0 and frames_lenght_source or math.inf

    while keep_reading_cap and not r.is_terminated:
        if not r.is_paused:
            ret, frame = cap.read()
            if ret == True:
                q_frame.put(frame)
                frames_read += 1

            if frames_read > frames_to_read:
                keep_reading_cap = False
        #else:
        #    timer.pause()

    print('OpenCV thread terminated.')