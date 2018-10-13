import speech_recognition as sr
import csv
import pandas as pd

def recognize_speech(recognizer, microphone):
# check that recognizer and microphone arguments are appropriate type
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    # adjust the recognizer sensitivity to ambient noise and record audio
    # from the microphone
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

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

    print("Say something :)")

    user_said = recognize_speech(recognizer, microphone)

    print("Value / Error Returned")
    
    if user_said["transcription"]:

        said = str(user_said["transcription"])
        
        with open('csvfile.csv','w') as file:
            file.write(said)
            file.write('\n')

        print(user_said["transcription"])
    elif user_said["success"]:
        print("I didn't catch that. What did you say?\n")
    else:
        print("Error")
        print(user_said["error"])

    
        
    


    
