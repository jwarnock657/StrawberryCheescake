from flask import Flask, render_template, request, send_from_directory
import os
import csv
app = Flask(__name__)

def predict():
    # Specify file paths FROM TOP LEVEL - !!! NOT WITHIN WEBAPP FOLDER !!!
    transcript = os.path.join("soundAnalysis", "transcript.txt")
    fourSSpeech2Txt = os.path.join("soundAnalysis", "4sSpeech2Txt.txt")
    textAnalysisOutput = os.path.join("soundAnalysis", "textAnalysisOutput.txt")
    voiceAnalysis = os.path.join("soundAnalysis", "voiceAnalysis.csv")

    with open(transcript, 'r') as file:
        t = ""
        for r in file.readlines():
            t+=r
        transcript = t

    with open(fourSSpeech2Txt, 'r') as file:
        t = ""
        for r in file.readlines():
            t += r
        fourSSpeech2Txt = t

    with open(textAnalysisOutput, 'r') as file:
        t = ""
        for r in file.readlines():
            t += r
        textAnalysisOutput = int(t)

    with open(voiceAnalysis, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
           voiceAnalysis = row

    return {'transcript': transcript, 'fourSSPeech2Txt': fourSSpeech2Txt, "textAnalysisOutput": textAnalysisOutput, "voiceAnalysis": voiceAnalysis}


answer = predict()
maxmood = "None"
confidence = 0
for key, value in answer["voiceAnalysis"].iteritems():
    if value > confidence:
        confidence = value
        maxmood = key
    else: 
        pass

confidence = round(float(confidence), 3) * 100

if (answer["textAnalysisOutput"] == 1):
    answer["textAnalysisOutput"] = "positive"
elif(answer["textAnalysisOutput"] == 0):
    answer["textAnalysisOutput"] = "Negative"

@app.route('/')
def index():
    return render_template('index.html', positivity=answer["textAnalysisOutput"], transcript=answer['fourSSPeech2Txt'], mood=maxmood, confidence = confidence)

# @app.route('/')
# def devs():
#     return render_template('developers..html')


if __name__ == "__main__":
    app.run()
