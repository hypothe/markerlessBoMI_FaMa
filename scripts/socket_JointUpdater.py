import socketio
import time
from os import getenv

SERVER_NAME = getenv("BOMI_SERVER_NAME", 'localhost') 


class JointUpdaterNS(socketio.ClientNamespace):
    def __init__(self, n_joints=3):
        socketio.ClientNamespace.__init__(self)

        self.joints_values = [0]*n_joints

    #@sio.event
    def on_connect(self):
        print('Connection established')

    #@sio.event
    def on_getJointsValues(self, msg):
        for id, joint_val in enumerate(msg):
            try:
                self.joints_values[id] = joint_val
            except IndexError:
                pass

class JointUpdater(socketio.Client):
    def __init__(self, n_joints=3):
        socketio.Client.__init__(self)

        self.ns = JointUpdaterNS(n_joints=n_joints)
        self.register_namespace(self.ns)

    def subscribeJVA(self):
        self.emit('subscribeToJointsValuesUpdates')
    
    def unsubscribeJVA(self):
        self.emit('unsubscribeToJointsValuesUpdates')

    def sendJointsValues(self, jval=None):
        jval = jval is None and self.ns.joints_values or jval

        msg = {"joints": jval}
        self.emit("jointsUpdate", msg)

if __name__ == "__main__":
    # the following line is used to set the name of the server,
    # which will be resolved by the DNS.
    # Not setting the variable allows for a local testing


    sio = JointUpdater()
    sio.connect('http://'+SERVER_NAME+':4242')

    time.sleep(2)

    print("Subscribing")
    sio.subscribeJVA()
    time.sleep(2)

    print("Sending")
    sio.sendJointsValues()

    time.sleep(5)
    print("Stored values {}".format(sio.ns.joints_values))