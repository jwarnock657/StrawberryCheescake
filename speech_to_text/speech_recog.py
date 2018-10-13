import speech_recognition as sr
import csv
import wave
import scipy.io.wavfile
import struct

#def save_wav(audio):
 #   try:
  #      # Open up a wav file
   #     print("Exporting Wav...")
    #    wav_file=wave.open("output_wav.wav","w")
#
 #       # wav params
  #      nchannels = 1
#
 #       sampwidth = 4
#
 #       # 44100 is the industry standard sample rate - CD quality.  If you need to
  #      # save on file size you can adjust it downwards. The stanard for low quality
   #     # is 8000 or 8kHz.
    #    nframes = 4 #len(audio)
     #   comptype = "NONE"
      #  compname = "not compressed"
       # print("Setting Params...")
        #wav_file.setparams((nchannels, sampwidth, 44100, nframes, comptype, compname))
#
 #       print("Params Set...")
        # WAV files here are using short, 16 bit, signed integers for the 
        # sample size.  So we multiply the floating point data we have by 32767, the
        # maximum value for a short integer.  NOTE: It is theortically possible to
        # use the floating point -1.0 to 1.0 data directly in a WAV file but not
        # obvious how to do that using the wave module in python.

  #      print("Writing...")
        #wav_file.writeframes(audio)
        
   #     wav_file.writeframes(struct.pack('h', audio ))
#
 #       print("Written and Closing...")
  #      wav_file.close()
   # except:
    #    print("WAVE Export Error")

    #return

def recognize_speech(recognizer, microphone):
# check that recognizer and microphone arguments are appropriate type
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    # adjust the recognizer sensitivity to ambient noise and record audio
    # from the microphone
    with microphone as source:
        print("Started Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.record(source, 4)
        print("Finished Listening....")
    
    # set up the response object
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    # try recognizing the speech in the recording
    # if a RequestError or UnknownValueError exception is caught,
    #     update the response object accordingly
    try:
        print("Getting Response....")
        response["transcription"] = recognizer.recognize_google(audio)
        print(audio)
        save_wav(audio)
        print("Got Response....")
    except sr.RequestError:
        # API was unreachable or unresponsive
        print("Request Error...")
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        print("Unknown Value Error....")
        # speech was unintelligible
        response["error"] = "Unable to recognize speech"

    return response

if __name__ == "__main__":
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    understood = False
    while(understood == False):
        print("Say something :)")

        user_said = recognize_speech(recognizer, microphone)

        print("Value / Error Returned")

        #if what was said was understoo
        if user_said["transcription"]:

            #change from type object to string
            said = str(user_said["transcription"])

            #write said to csv
            with open('csvfile.csv','w') as file:
                file.write(said)
                file.write('\n')

            #print what was said
            print(user_said["transcription"])
            understood = True
        elif user_said["success"]:
            print("I didn't catch that. What did you say?\n")
        else:
            print("Error")
            print(user_said["error"])

    
        
    


    
