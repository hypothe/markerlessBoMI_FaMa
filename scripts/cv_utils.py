import cv2
import time


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


def get_data_from_camera(cap, q_frame, r, cal, fps=120):
    '''
    function that runs in the thread to capture current frame and put it into the queue
    :param cap: object of OpenCV class
    :param q_frame: queue to store current frame
    :param r: object of Reaching class
    :return:
    '''
    # check for possible incorrectly represented fps values
    # (<FIX> some camera reporting fps = 0)
    if fps <= 0:
        fps = 120
    
    interframe_delay = float(1.0/fps)
    while not r.is_terminated:
        start_time = time.time()
        if not r.is_paused:
            try:
                ret, frame = cap.read()
                q_frame.put(frame)
            except:
                r.is_terminated = True
        # consume the source at the correct frequecy      
        end_time = time.time()
        
        time.sleep(max(0, interframe_delay - (end_time - start_time)))

    print('OpenCV thread terminated.')