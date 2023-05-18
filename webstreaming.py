from flask import Response
from flask import Flask
from flask import render_template, flash, redirect, url_for, request
from flask_login import login_user, logout_user, current_user, login_required
from flask_login import LoginManager
from flask_login import UserMixin
from forms import LoginForm
from werkzeug.urls import url_parse
import threading
import argparse
import datetime
import time
import cv2
from queue import Queue
from streams import Camera, NormalVideoStream, FPSOverlayVideoStream, ObjectsDetectionVideoStream
import queue
import os


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
# initialize a flask object
app = Flask(__name__)
app.config.from_object(Config)
login = LoginManager(app)
login.login_view = 'login'

class MockUser(UserMixin):
    def get_id(self):
        return 1
user = MockUser()

username = "user"
password = "1234"

class NukeOldDataQueue(queue.Queue):
    def put(self,*args,**kwargs):
        queue.Queue.put(self,*args,**kwargs)
        if self.full():
            try:
                oldest_data = self.get()
                # print('[WARNING]: throwing away old data:'+repr(oldest_data))
            # a True value from `full()` does not guarantee
            # that anything remains in the queue when `get()` is called
            except Queue.Empty:
                pass

MAX_QUE_SIZE = 30
# Queues for streams
framesNormalQue = NukeOldDataQueue(maxsize=MAX_QUE_SIZE)
framesFPSQue = NukeOldDataQueue(maxsize=MAX_QUE_SIZE)
framesCarQue = NukeOldDataQueue(maxsize=MAX_QUE_SIZE)
framesPeopleQue = NukeOldDataQueue(maxsize=MAX_QUE_SIZE)
print('Queues created')

url = "https://www.youtube.com/watch?v=y3sMI1HtZfE"
camera0 = Camera(url, [framesNormalQue])
camera0.start()

url = "https://youtu.be/mwN6l3O1MNI"
camera1 = Camera(url, [framesFPSQue])
camera1.start()

camera2 = Camera("https://youtu.be/qP1y7Tdab7Y", [framesCarQue])
camera2.start()

camera3 = Camera("https://youtu.be/v9rQqa_VTEY", [framesPeopleQue]) #https://youtu.be/1EiC9bvVGnk
camera3.start()

print('Cameras started')

# Streams
normalStream = NormalVideoStream(framesNormalQue)
fpsStream = FPSOverlayVideoStream(framesFPSQue)
carStream = ObjectsDetectionVideoStream(framesCarQue, camera2)
peopleStream = ObjectsDetectionVideoStream(framesPeopleQue, camera3, objects=['person'], count=True)
print('Streams created')

normalStream.start()
fpsStream.start()
carStream.start()
peopleStream.start()


@login.user_loader
def load_user(id):
    return user

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        if form.username.data != username or form.password.data != password:
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route("/", defaults={'number': 1})
@app.route("/<number>")
@login_required
def index(number):
    # return the rendered template
    return render_template("index.html", number=number)

@login_required
@app.route("/video_feed/<stream_id>")
def video_feed(stream_id):
    # return the response generated along with the specific media
    # type (mime type)
    streams = {
            1: fpsStream,
            2: normalStream,
            3: carStream,
            4: peopleStream,
            }
    s = streams[int(stream_id)]
    s.enable(True)
    return Response(s.gen(),
                    mimetype = "multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
        help="ip address of the device")
    ap.add_argument("--login", type=str, required=False,
                    help="login")
    ap.add_argument("--pswd", type=str, required=False,
                    help="password")
    ap.add_argument("-o", "--port", type=int, required=True,
        help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
        help="# of frames used to construct the background model")
    args = vars(ap.parse_args())
    if args['login'] is not None: 
        username = args['login']
    if args['pswd'] is not None:
        password = args['pswd']
    # start a thread that will perform motion detection
    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)
# release the video stream pointer
camera0.stopCamera()
camera1.stopCamera()
camera2.stopCamera()
camera3.stopCamera()
