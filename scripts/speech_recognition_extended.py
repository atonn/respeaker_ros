from speech_recognition import Recognizer, AudioData

import io
import os
import base64
import json

try:  # attempt to use the Python 2 modules
    from urllib import urlencode
    from urllib2 import Request, urlopen, URLError, HTTPError
except ImportError:  # use the Python 3 modules
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError

class WaitTimeoutError(Exception): pass


class RequestError(Exception): pass


class UnknownValueError(Exception): pass

#from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
#from ibm_watson import SpeechToTextV1
#from ibm_watson.websocket import RecognizeCallback, AudioSource

###
# we try to extend the official uberi/SpeechRecognition library with modern watson (its outdated in the uberi repo)
# but along the way we notice that ibm-watson only supports python 3.5+ which we won't get into ROS kinetic without building a whole second ROS kinetic
# from source with lots of hackarounds just for this purpose. so work in progress, for now we can use the outdated uberi watson method again
# XXX
###
class PepperProjectRecognizer(Recognizer):
#     def recognize_ibm_single_utterance_python3(self, audio_data, language="en-US", pfilter=0, show_all=False):
#         assert isinstance(audio_data, AudioData), "Data must be audio data"

#         flac_data = audio_data.get_flac_data(
#             convert_rate=None if audio_data.sample_rate >= 16000 else 16000,  # audio samples should be at least 16 kHz
#             convert_width=None if audio_data.sample_width >= 2 else 2  # audio samples should be at least 16-bit
#         )

#         speech_to_text = SpeechToTextV1()

#         model = service.get_model('en-US_BroadbandModel').get_result()
        
#         response_text = service.recognize(
#             audio=flac_data,
#             content_type='audio/x-flac',
#             timestamps=True,
#             word_confidence=True).get_result()

#         result = json.loads(response_text)

#         # return results
#         if show_all: return result
#         if "results" not in result or len(result["results"]) < 1 or "alternatives" not in result["results"][0]:
#             raise UnknownValueError()

#         transcription = []
#         for utterance in result["results"]:
#             if "alternatives" not in utterance: raise UnknownValueError()
#             for hypothesis in utterance["alternatives"]:
#                 if "transcript" in hypothesis:
#                     transcription.append(hypothesis["transcript"])
#         return "\n".join(transcription)

    def recognize_ibm_single_utterance(self, audio_data, key, language="en-US", url="https://gateway-lon.watsonplatform.net/speech-to-text/api/v1/recognize", show_all=False):
        """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using the IBM Speech to Text API.
        The IBM Speech to Text username and password are specified by ``username`` and ``password``, respectively. Unfortunately, these are not available without `signing up for an account <https://console.ng.bluemix.net/registration/>`__. Once logged into the Bluemix console, follow the instructions for `creating an IBM Watson service instance <https://www.ibm.com/watson/developercloud/doc/getting_started/gs-credentials.shtml>`__, where the Watson service is "Speech To Text". IBM Speech to Text usernames are strings of the form XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX, while passwords are mixed-case alphanumeric strings.
        The recognition language is determined by ``language``, an RFC5646 language tag with a dialect like ``"en-US"`` (US English) or ``"zh-CN"`` (Mandarin Chinese), defaulting to US English. The supported language values are listed under the ``model`` parameter of the `audio recognition API documentation <https://www.ibm.com/watson/developercloud/speech-to-text/api/v1/#sessionless_methods>`__, in the form ``LANGUAGE_BroadbandModel``, where ``LANGUAGE`` is the language value.
        Returns the most likely transcription if ``show_all`` is false (the default). Otherwise, returns the `raw API response <https://www.ibm.com/watson/developercloud/speech-to-text/api/v1/#sessionless_methods>`__ as a JSON dictionary.
        Raises a ``speech_recognition.UnknownValueError`` exception if the speech is unintelligible. Raises a ``speech_recognition.RequestError`` exception if the speech recognition operation failed, if the key isn't valid, or if there is no internet connection.
        """
        assert isinstance(audio_data, AudioData), "Data must be audio data"
        assert isinstance(key, str), "``key`` must be a string"

        flac_data = audio_data.get_flac_data(
            convert_rate=None if audio_data.sample_rate >= 16000 else 16000,  # audio samples should be at least 16 kHz
            convert_width=None if audio_data.sample_width >= 2 else 2  # audio samples should be at least 16-bit
        )

        # url = "https://stream.watsonplatform.net/speech-to-text/api/v1/recognize?{}".format(urlencode({
        #     "profanity_filter": "false",
        #     "model": "{}_BroadbandModel".format(language),
        #     "inactivity_timeout": -1,  # don't stop recognizing when the audio stream activity stops
        # }))

        request = Request(url, data=flac_data, headers={
            "Content-Type": "audio/x-flac",
            "model": "{}_BroadbandModel".format(language),
            #XXX LANGUAGE IS MISSING / NOT IN THE CORRECT SPOT?
            "X-Watson-Learning-Opt-Out": "true",  # prevent requests from being logged, for improved privacy
        })
        request.get_method = lambda: 'POST'
        username = 'apikey'
        password = key
        authorization_value = base64.standard_b64encode("{}:{}".format(username, password).encode("utf-8")).decode("utf-8")
        request.add_header("Authorization", "Basic {}".format(authorization_value))
        try:
            response = urlopen(request, timeout=self.operation_timeout)
        except HTTPError as e:
            raise RequestError("recognition request failed: {}".format(e.reason))
        except URLError as e:
            raise RequestError("recognition connection failed: {}".format(e.reason))
        response_text = response.read().decode("utf-8")
        result = json.loads(response_text)

        # return results
        if show_all: return result
        if "results" not in result or len(result["results"]) < 1 or "alternatives" not in result["results"][0]:
            raise UnknownValueError(response_text)

        transcription = []
        for utterance in result["results"]:
            if "alternatives" not in utterance: raise UnknownValueError()
            for hypothesis in utterance["alternatives"]:
                if "transcript" in hypothesis:
                    transcription.append(hypothesis["transcript"])
        return "\n".join(transcription)