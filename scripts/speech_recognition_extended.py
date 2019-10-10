from speech_recognition import Recognizer

import io
import os

# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

class PepperProjectRecognizer(recognizer)
    def recognize_google_single_utterance(self, audio_data, language="en-US", pfilter=0, show_all=False):
        """
        Uses the actual google cloud speech API, unlike the standard speech recognition python package.
        Requires a google account to work, with Credentials downloaded and Exported as a System Variable.
        Example:
        export GOOGLE_APPLICATION_CREDENTIALS="/home/spoid/Downloads/googlePepperCloud-355fb60430ac.json"
        """
        assert isinstance(audio_data, AudioData), "``audio_data`` must be audio data"
        assert isinstance(language, str), "``language`` must be a string"
        

        flac_data = audio_data.get_flac_data(
            convert_rate=None if audio_data.sample_rate >= 8000 else 8000,  # audio samples must be at least 8 kHz
            convert_width=2  # audio samples must be 16-bit
        )
        
        client = speech.SpeechClient()
        
        #if key is None: key = "AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw"
        #url = "http://www.google.com/speech-api/v2/recognize?{}".format(urlencode({
        #    "client": "chromium",
        #    "lang": language,
        #    "key": key,
        #    "pFilter": pfilter
        #}))
        
        request = Request(url, data=flac_data, headers={"Content-Type": "audio/x-flac; rate={}".format(audio_data.sample_rate)})
        
        config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=audio_data.sample_rate,
        language_code=language)

        audio = types.RecognitionAudio(content=flac_data)

        # obtain audio transcription results
        try:
            #response = urlopen(request, timeout=self.operation_timeout)
            response = client.recognize(config, audio)
        except HTTPError as e:
            raise RequestError("recognition request failed: {}".format(e.reason))
        except URLError as e:
            raise RequestError("recognition connection failed: {}".format(e.reason))
        response_text = response.read().decode("utf-8")

        # ignore any blank blocks
        actual_result = []
        for line in response_text.split("\n"):
            if not line: continue
            result = json.loads(line)["result"]
            if len(result) != 0:
                actual_result = result[0]
                break

        # return results
        if show_all: return actual_result
        if not isinstance(actual_result, dict) or len(actual_result.get("alternative", [])) == 0: raise UnknownValueError()

        if "confidence" in actual_result["alternative"]:
            # return alternative with highest confidence score
            best_hypothesis = max(actual_result["alternative"], key=lambda alternative: alternative["confidence"])
        else:
            # when there is no confidence available, we arbitrarily choose the first hypothesis.
            best_hypothesis = actual_result["alternative"][0]
        if "transcript" not in best_hypothesis: raise UnknownValueError()
        return best_hypothesis["transcript"]
