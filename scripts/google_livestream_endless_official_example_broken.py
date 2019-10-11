#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Speech API sample application using the streaming API.
NOTE: This module requires the dependencies `pyaudio` and `termcolor`.
To install using pip:
    pip install pyaudio
    pip install termcolor
Example usage:
    python transcribe_streaming_infinite.py
"""

# [START speech_transcribe_infinite_streaming]

import time
import re
import sys

# uses result_end_time currently only avaialble in v1p1beta, will be in v1 soon
from google.cloud import speech_v1p1beta1 as speech
import pyaudio
from six.moves import queue

import usb.core
import usb.util
import rospy
import numpy as np
from speech_recognition_msgs.msg import SpeechRecognitionCandidates

# Audio recording parameters
STREAMING_LIMIT = 10000
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms

RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[0;33m'

pub_speech = None

def get_current_time():
    """Return Current Time in MS."""

    return int(round(time.time() * 1000))


class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""
    VENDOR_ID = 0x2886
    PRODUCT_ID = 0x0018

    def __init__(self, rate, chunk_size):
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

        self._rate = rate
        self.chunk_size = chunk_size
        self._num_channels = 1
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

    def __enter__(self):

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

    def _fill_buffer(self, in_data, *args, **kwargs):
        """Continuously collect data from the audio stream, into the buffer."""
        data = np.fromstring(in_data, dtype=np.int16)
        chunk_per_channel = int( len(data) / self.channels )
        data = np.reshape(data, (chunk_per_channel, self.channels))
        chan_data = data[:, self.channel]
        #self._buff.put(in_data)
        self._buff.put(chan_data.tostring())
        return None, pyaudio.paContinue

    def generator(self):
        """Stream Audio from microphone to API and to local buffer"""

        while not self.closed:
            data = []

            if self.new_stream and self.last_audio_input:

                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)

                if chunk_time != 0:

                    if self.bridging_offset < 0:
                        self.bridging_offset = 0

                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time

                    chunks_from_ms = round((self.final_request_end_time -
                                            self.bridging_offset) / chunk_time)

                    self.bridging_offset = (round((
                        len(self.last_audio_input) - chunks_from_ms)
                                                  * chunk_time))

                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        data.append(self.last_audio_input[i])

                self.new_stream = False

            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            self.audio_input.append(chunk)

            if chunk is None:
                return
            data.append(chunk)
            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)

                    if chunk is None:
                        return
                    data.append(chunk)
                    self.audio_input.append(chunk)

                except queue.Empty:
                    break

            yield b''.join(data)


def listen_print_loop(responses, stream):
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
    #somehow this is broken in google's examples (seriously... why), so we put a try catch around it
    try:
        for response in responses:
            try:
                if get_current_time() - stream.start_time > STREAMING_LIMIT:
                    stream.start_time = get_current_time()
                    break

                if not response.results:
                    continue

                result = response.results[0]

                if not result.alternatives:
                    continue

                transcript = result.alternatives[0].transcript

                result_seconds = 0
                result_nanos = 0

                if result.result_end_time.seconds:
                    result_seconds = result.result_end_time.seconds

                if result.result_end_time.nanos:
                    result_nanos = result.result_end_time.nanos

                stream.result_end_time = int((result_seconds * 1000)
                                            + (result_nanos / 1000000))

                corrected_time = (stream.result_end_time - stream.bridging_offset
                                + (STREAMING_LIMIT * stream.restart_counter))
                # Display interim results, but with a carriage return at the end of the
                # line, so subsequent lines will overwrite them.

                if result.is_final:

                    sys.stdout.write(GREEN)
                    sys.stdout.write('\033[K')
                    sys.stdout.write(str(corrected_time) + ': ' + transcript + '\n')

                    stream.is_final_end_time = stream.result_end_time
                    stream.last_transcript_was_final = True

                    #global pub_speech
                    #msg = SpeechRecognitionCandidates(transcript=transcript)
                    #pub_speech.publish(msg)
                    
                    # Exit recognition if any of the transcribed phrases could be
                    # one of our keywords.
                    #if re.search(r'\b(exit|quit)\b', transcript, re.I):
                    #    sys.stdout.write(YELLOW)
                    #    sys.stdout.write('Exiting...\n')
                    #    stream.closed = True
                    #    break

                else:
                    sys.stdout.write(RED)
                    sys.stdout.write('\033[K')
                    sys.stdout.write(str(corrected_time) + ': ' + transcript + '\r')

                    stream.last_transcript_was_final = False
            except Exception as g:
                print("Weird Google Exception inner loop: ", str(g))
    except Exception as e:
        print("Weird Google Exception: ", str(e))

def main():
    """start bidirectional streaming from microphone input to speech API"""
    rospy.init_node("speech_to_text")
    global pub_speech
    pub_speech = rospy.Publisher(
            "speech_to_text", SpeechRecognitionCandidates, queue_size=1)

    client = speech.SpeechClient()
    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code='en-US',
        max_alternatives=1)
    streaming_config = speech.types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)
    print(mic_manager.chunk_size)
    sys.stdout.write(YELLOW)
    sys.stdout.write('\nListening, say "Quit" or "Exit" to stop.\n\n')
    sys.stdout.write('End (ms)       Transcript Results/Status\n')
    sys.stdout.write('=====================================================\n')

    with mic_manager as stream:

        while not stream.closed:
            sys.stdout.write(YELLOW)
            sys.stdout.write('\n' + str(
                STREAMING_LIMIT * stream.restart_counter) + ': NEW REQUEST\n')

            stream.audio_input = []
            audio_generator = stream.generator()

            requests = (speech.types.StreamingRecognizeRequest(
                audio_content=content)for content in audio_generator)

            responses = client.streaming_recognize(streaming_config,
                                                   requests)

            # Now, put the transcription responses to use.
            listen_print_loop(responses, stream)

            if stream.result_end_time > 0:
                stream.final_request_end_time = stream.is_final_end_time
            stream.result_end_time = 0
            stream.last_audio_input = []
            stream.last_audio_input = stream.audio_input
            stream.audio_input = []
            stream.restart_counter = stream.restart_counter + 1

            if not stream.last_transcript_was_final:
                sys.stdout.write('\n')
            stream.new_stream = True


if __name__ == '__main__':
    main()

# [END speech_transcribe_infinite_streaming]
