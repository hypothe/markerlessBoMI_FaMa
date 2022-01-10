import socketio
import time
from os import getenv

class JointUpdaterNS(socketio.ClientNamespace):
    def __init__(self):
        socketio.ClientNamespace.__init__(self)

        self.joints_values = [0, 0, 0]

    #@sio.event
    def on_connect(self):
        print('Connection established')

    #@sio.event
    def on_getJointsValues(self, msg):

        print("Got joints message {}".format(msg))
        for id, joint_val in enumerate(msg):
            try:
                self.joints_values[id] = joint_val
            except IndexError:
                pass

class JointUpdater(socketio.Client):
    def __init__(self):
        socketio.Client.__init__(self)

        self.ns = JointUpdaterNS()
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

    server_name = getenv("BOMI_SERVER_NAME", 'localhost') 

    sio = JointUpdater()
    sio.connect('http://'+server_name+':4242')

    time.sleep(2)

    print("Subscribing")
    sio.subscribeJVA()
    time.sleep(2)

    print("Sending")
    sio.sendJointsValues()

    time.sleep(5)
    print("Stored values {}".format(sio.ns.joints_values))