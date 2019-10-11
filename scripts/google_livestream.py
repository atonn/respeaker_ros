#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import re
import sys
import time

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import pyaudio
from six.moves import queue
import usb.core
import usb.util
import rospy
import numpy as np
from speech_recognition_msgs.msg import SpeechRecognitionCandidates

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
pub_speech = None #dummy global variable for publisher

class MicrophoneStream(object):
    VENDOR_ID = 0x2886
    PRODUCT_ID = 0x0018

    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self.dev = usb.core.find(idVendor=self.VENDOR_ID,
                                 idProduct=self.PRODUCT_ID)
        if not self.dev:
            raise RuntimeError("Failed to find Respeaker device")
        print("Initializing Respeaker device")
        self.dev.reset()
        time.sleep(5)  # it will take 5 seconds to re-recognize as audio device
        self.pyaudio = pyaudio.PyAudio()

        #we want channel 0 (processed audio from respeaker array v2.0)
        self.channels = None
        self.channel = 0
        self.device_index = None
        #self.rate = 16000
        self.bitwidth = 2
        self.bitdepth = 16

        # find device
        count = self.pyaudio.get_device_count()
        print("%d audio devices found" % count)
        for i in range(count):
            info = self.pyaudio.get_device_info_by_index(i)
            name = info["name"].encode("utf-8")
            chan = info["maxInputChannels"]
            print(" - %d: %s" % (i, name))
            if name.lower().find("respeaker") >= 0:
                self.channels = chan
                self.device_index = i
                print("Found %d: %s (channels: %d)" % (i, name, chan))
                break
        if self.device_index is None:
            print("Failed to find respeaker device by name. Using default input")
            info = self.pyaudio.get_default_input_device_info()
            self.channels = info["maxInputChannels"]
            self.device_index = info["index"]

        if self.channels != 6:
            print("%d channel is found for respeaker" % self.channels)
            print("You may have to update firmware.")
        self.channel = min(self.channels - 1, max(0, self.channel))
        print("Channel set to {}".format(self.channel))
        print("Channels: ", self.channels)
        #frames_per_buffer=1024,


        self._audio_interface = self.pyaudio #ugly XXX
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            ######channels=1, rate=self._rate,
            ######input=True, frames_per_buffer=self._chunk,
            channels=self.channels, 
            rate=self._rate,
            input=True, 
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
            input_device_index=self.device_index,
        )

        print("Pyaudio interface opened.")
        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        data = np.fromstring(in_data, dtype=np.int16)
        chunk_per_channel = int( len(data) / self.channels )
        data = np.reshape(data, (chunk_per_channel, self.channels))
        chan_data = data[:, self.channel]
        #self._buff.put(in_data)
        self._buff.put(chan_data.tostring())
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)


def listen_print_loop(responses):
    global pub_speech
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + '\r')
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            print(transcript + overwrite_chars)
            #print(type(result))
            #print(result.alternatives[0].transcript.join(""))
            msg = SpeechRecognitionCandidates(transcript=transcript) #XXX stringify
            pub_speech.publish(msg)
            print(response)
            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            #if re.search(r'\b(exit|quit)\b', transcript, re.I):
            #    print('Exiting..')
            #    break

            num_chars_printed = 0


def main():
    rospy.init_node("speech_to_text")
    global pub_speech
    pub_speech = rospy.Publisher(
            "speech_to_text", SpeechRecognitionCandidates, queue_size=1)

    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = 'en-US'  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code)
    streaming_config = types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    print("RATE: ", RATE)
    print("CHUNK: ", CHUNK) #apparently 1600, lets try 1024
    
    #cheap way to dodge google's "exceeded 300 seconds of audio in one go"
    #better way: XXX https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/speech/microphone/transcribe_streaming_infinite.py
    while True:
        try:
            with MicrophoneStream(RATE, CHUNK) as stream:
                audio_generator = stream.generator()
                requests = (types.StreamingRecognizeRequest(audio_content=content)
                            for content in audio_generator)

                responses = client.streaming_recognize(streaming_config, requests)

                # Now, put the transcription responses to use.
                listen_print_loop(responses)
        except Exception as e:
            print(e)
        time.sleep(5)


if __name__ == '__main__':
    main()
