#!/usr/bin/env python
# -*- coding: utf-8 -*-
#monitor the inbuilt SPEECHRECOGNIZED trigger of the respeaker mic array 2.0
#when it turns to 1, tell snips to listen via mosquitto
#receive snips recognition via mosquitto and broadcast it to ROS as well

#bonus: mute when pepper is speaking (pepper speaking node should be constantly rospy-written to a global variable, we can not make a call once we want to start recording, that is too slow)
#bonus: nice LED feedback (wie japaner)

#"ABSCHNEIDEN" wie der japaner falls snips zu langsam abschneidet

from tuning import Tuning
import usb.core
import usb.util
import time, json

#!/usr/bin/env python
import rospy
from std_msgs.msg import String




    
import paho.mqtt.client as mqtt
broker_address="192.168.1.140"
#Client(client_id="", clean_session=True, userdata=None, protocol=MQTTv311, transport="tcp")
client = mqtt.Client("P1")

client.connect(broker_address, keepalive=6000) #(host, port=1883, keepalive=6000, bind_address="")

#deactivate wake word on snips just in case
client.publish('hermes/hotword/toggleOff','{"siteId": "default"}')
client.publish('hermes/nlu/toggleOff','{"siteId": "default"}')

dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
#print dev

LAUNCH_INTERVAL = 1 #how often to we want to fire it
last_seen_speechdetected_status = 0
last_speechdetected_timestamp = time.time()

pepper_is_speaking = False

pub = rospy.Publisher('recognized_speech', String, queue_size=10)

#mqtt callback for recognized speech from snips via mqtt
def on_message(client, userdata, message):
    rospy.loginfo("\n")
    rospy.loginfo(message.payload.decode("utf-8"))
    recognized_string = json.loads(message.payload.decode("utf-8"))["text"]
    rospy.loginfo("RECOGNIZED: " + recognized_string)
    if recognized_string.strip() != "":
        rospy.loginfo("SENDING TO PUBLISHER: " + recognized_string)
        pub.publish(String(json.loads(message.payload.decode("utf-8"))["text"]))
    #rospy.loginfo("message received " ,str(message.payload.decode("utf-8")))
    #rospy.loginfo("message topic=",message.topic)
    #rospy.loginfo("message qos=",message.qos)
    #rospy.loginfo("message retain flag=",message.retain)
   
#rospy callback
def callback(data):
    global pepper_is_speaking
    global client
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    if data.data == "enqueued":
        rospy.loginfo("PEPPER IS SPEAKING!")
        rospy.loginfo(data)
        pepper_is_speaking = True
        client.publish('hermes/asr/stopListening','{"siteId": "default"}') #try to send our last recognition before we turn asr off
        client.publish('hermes/asr/toggleOff')
    elif data.data == "done":
        rospy.loginfo("PEPPER STOPPED SPEAKING!")
        rospy.loginfo(data)
        pepper_is_speaking = False
        client.publish('hermes/asr/toggleOn')
    elif data.data == "started":
        pass
    else:
        rospy.logerr("pepper_speech_status rospy subscriber did not understand what it saw:")
        rospy.logerr(data)
    
def ros_listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('speech_to_text', anonymous=True)
    rospy.Subscriber("pepper_speech_status", String, callback)
    
    # spin() simply keeps python from exiting until this node is stopped
    #rospy.spin() 
    #If there is another blocking call that keeps your program running, there is no need to call rospy.spin(). Unlike in C++ where spin() is needed to process all the threads, in python all it does is block.
    


if __name__ == '__main__':
    global pepper_is_speaking
    client.on_message=on_message #attach function to callback
    client.loop_start() #start the loop
    rospy.loginfo("Subscribing to topic","hermes/asr/textCaptured")
    client.subscribe("hermes/asr/textCaptured")

    ros_listener()
    #wait until microphone is correctly restarted from the other script
    rospy.loginfo("Snips script is sleeping for 20 seconds to wait for respeaker_node to finish launching")
    time.sleep(20)

    if dev:
        Mic_tuning = Tuning(dev)
        
        while True:
            try:
                #rospy.loginfo"VOICEACTIVITY:", Mic_tuning.is_voice()),
                speechdetected = Mic_tuning.read("VOICEACTIVITY") #SPEECHRECOGNIZED fires too often
                #rospy.loginfospeechdetected),
                
                if speechdetected:
                    if time.time() - last_speechdetected_timestamp > LAUNCH_INTERVAL and not pepper_is_speaking:
                        client.publish('hermes/asr/startListening','{"siteId": "default"}')
                        last_speechdetected_timestamp = time.time() #we don't want to trigger more than once per second or so
                        rospy.loginfo("\nPUBLISHED TO HERMES TO START LISTENING") #snips decides when to stop on its own, so no need to tell it to
                    
                last_seen_speechdetected_status = speechdetected
            except KeyboardInterrupt:
                break
