#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import actionlib
import rospy
import speech_recognition as SR

from actionlib_msgs.msg import GoalStatus, GoalStatusArray
from audio_common_msgs.msg import AudioData
from sound_play.msg import SoundRequest, SoundRequestAction, SoundRequestGoal
from speech_recognition_msgs.msg import SpeechRecognitionCandidates


class SpeechToText(object):
    def __init__(self):
        # format of input audio data
        self.sample_rate = rospy.get_param("~sample_rate", 16000)
        self.sample_width = rospy.get_param("~sample_width", 2L)
        # language of STT service
        self.language = rospy.get_param("~language", "en-US")
        self.asr_engine = rospy.get_param("~asr_engine", "google_legacy_single_utterance")
        # ignore voice input while the robot is speaking
        self.self_cancellation = rospy.get_param("~self_cancellation", True)
        # time to assume as SPEAKING after tts service is finished
        self.tts_tolerance = rospy.Duration.from_sec(
            rospy.get_param("~tts_tolerance", 1.0))

        self.recognizer = SR.Recognizer()

        self.tts_action = None
        self.last_tts = None
        self.is_canceling = False
        if self.self_cancellation:
            self.tts_action = actionlib.SimpleActionClient(
                "sound_play", SoundRequestAction)
            if self.tts_action.wait_for_server(rospy.Duration(5.0)):
                self.tts_timer = rospy.Timer(rospy.Duration(0.1), self.tts_timer_cb)
            else:
                rospy.logerr("action '%s' is not initialized." % rospy.remap_name("sound_play"))
                self.tts_action = None

        self.pub_speech = rospy.Publisher(
            "speech_to_text", SpeechRecognitionCandidates, queue_size=1)
        #confusing: ros launch file contains: <remap from="audio" to="speech_audio"/>
        #so actually here we are subscribing to speech_audio, which makes much more sense - if you just read the python code
        #and do not know about the remapping in the ros launch file, nothing makes sense anymore. hours were lost.
        self.sub_audio = rospy.Subscriber("audio", AudioData, self.audio_cb)

    def tts_timer_cb(self, event):
        stamp = event.current_real
        active = False
        for st in self.tts_action.action_client.last_status_msg.status_list:
            if st.status == GoalStatus.ACTIVE:
                active = True
                break
        if active:
            if not self.is_canceling:
                rospy.logdebug("START CANCELLATION")
                self.is_canceling = True
                self.last_tts = None
        elif self.is_canceling:
            if self.last_tts is None:
                self.last_tts = stamp
            if stamp - self.last_tts > self.tts_tolerance:
                rospy.logdebug("END CANCELLATION")
                self.is_canceling = False

    def audio_cb(self, msg):
        if self.is_canceling:
            rospy.loginfo("Speech is cancelled")
            return
        data = SR.AudioData(msg.data, self.sample_rate, self.sample_width)
        try:
            if self.asr_engine == "google_legacy_single_utterance":
                rospy.loginfo("Waiting for result %d" % len(data.get_raw_data()))
                result = self.recognizer.recognize_google(
                    data, language=self.language)
                msg = SpeechRecognitionCandidates(transcript=[result])
                rospy.loginfo("Got result: %s" % msg)
                self.pub_speech.publish(msg)

            elif self.asr_engine == "google_cloud_single_utterance":
                rospy.loginfo("Waiting for result %d" % len(data.get_raw_data()))
                result = self.recognizer.recognize_google_cloud(
                    data, language=self.language)
                msg = SpeechRecognitionCandidates(transcript=[result])
                rospy.loginfo("Got result {}: {}".format(len(data.get_raw_data()), msg))
                self.pub_speech.publish(msg)

            else:
                rospy.logerr("No valid Recognizer was set (in the roslaunch file): %s" % self.asr_engine)

        except SR.UnknownValueError as e:
            rospy.logerr("Failed to recognize: %s" % str(e))
        except SR.RequestError as e:
            rospy.logerr("Failed to recognize: %s" % str(e))


if __name__ == '__main__':
    rospy.init_node("speech_to_text")
    stt = SpeechToText()
    rospy.spin()
