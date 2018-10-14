from flask import Flask, render_template, request, send_from_directory
import os
app = Flask(__name__)

def predict():
    result = "Male, Happy"
    return result

answer = predict()
@app.route('/')
def index():
    return render_template('index.html', positivity=answer, transcript="hello dom")



if __name__ == "__main__":
    app.run()
