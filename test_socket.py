import socketio
import time

class JointUpdater(socketio.Client):
    def __init__(self):
        super.__init__(self)

        self.joints_values = []

    #@sio.event
    def on_connect(self):
        print('Connection estabished')

    #@sio.event
    def on_getJointsValues(self, msg):

        print("Got joints message {}".format(msg))
        for id, joint_val in enumerate(msg):
            self.joints_values[id] = joint_val

    def subscribeJVA(self):
        self.emit('subscribeToJointsValuesUpdates')
    
    def unsubscribeJVA(self):
        self.emit('unsubscribeToJointsValuesUpdates')

    def sendJointsValues(self, jval=None):
        jval = jval is None and self.joints_values or jval

        msg = {"joints": jval}
        self.emit("jointsUpdate", msg)

if __name__ == "__main__":

    sio = JointUpdater()
    sio.connect('http://localhost:4242')

    time.sleep(2)

    print("Subscribing")
    sio.subscribeJVA()
    time.sleep(2)

    print("Sending")
    sio.sendJointsValues([0,1,2])
