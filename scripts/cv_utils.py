import cv2


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


def get_data_from_camera(cap, q_frame, r):
    '''
    function that runs in the thread to capture current frame and put it into the queue
    :param cap: object of OpenCV class
    :param q_frame: queue to store current frame
    :param r: object of Reaching class
    :return:
    '''
    
    while not r.is_terminated:
        if not r.is_paused:
            try:
                ret, frame = cap.read()
                q_frame.put(frame)
            except:
                r.is_terminated = True
        
    print('OpenCV thread terminated.')