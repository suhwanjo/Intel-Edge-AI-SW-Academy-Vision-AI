from django.contrib import auth
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate
from django import forms
import cv2
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
from ultralytics import YOLO
import dlib
import numpy as np
from scipy.spatial import distance as dist

import torch

# Create your views here.


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("C:/Users/USER/Desktop/Intel-Edge-AI-SW-Academy-Vision-AI/accounts/shape_predictor_68_face_landmarks.dat")
        self.eye_closed_long_count = 0
        self.eye_closed = False
        self.prev_eye_closed = False
        (self.lStart, self.lEnd) = (42, 48)
        (self.rStart, self.rEnd) = (36, 42)
        self.EAR_THRESHOLD = 0.25
        self.fps = 30
        self.SECONDS_TO_COUNT = 2
        self.THRESHOLD_FRAMES = int(self.fps * self.SECONDS_TO_COUNT)
        self.eye_closed_count = 0

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        if not ret:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        for face in faces:
            landmarks = self.predictor(gray, face)
            shape = np.array([[p.x, p.y] for p in landmarks.parts()])
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < self.EAR_THRESHOLD:
                self.eye_closed = True
            else:
                self.eye_closed = False

            if self.eye_closed:
                if self.prev_eye_closed:
                    self.eye_closed_count += 1
                else:
                    self.eye_closed_count = 1
            else:
                self.eye_closed_count = 0

            self.prev_eye_closed = self.eye_closed

            if self.eye_closed_count >= self.THRESHOLD_FRAMES:
                self.eye_closed_long_count += 1
                self.eye_closed_count = 0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        cv2.putText(frame, f"Long Blink Count: {self.eye_closed_long_count, self.THRESHOLD_FRAMES}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@gzip.gzip_page
def webcam_feed(request):
    try:
        camera = VideoCamera()
        return StreamingHttpResponse(gen(camera), content_type="multipart/x-mixed-replace;boundary=frame")
    except Exception as e:
        print(e)
        pass

def board_view(request):
    return render(request, 'accounts/board.html')

def webcam_view(request):
    return render(request, 'accounts/webcam.html')
def stop_webcam(request):
    try:
        del request.session['camera']
    except KeyError:
        pass
    return redirect('board')
class SignupForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta(UserCreationForm.Meta):
        model = User
        fields = UserCreationForm.Meta.fields + ('email',)

# views.py
def signup(request):
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth.login(request, user)
            return redirect('/')
    else:
        form = SignupForm()
    return render(request, 'accounts/signup.html', {'form': form})


def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return redirect('board')
        else:
            return render(request, 'accounts/login.html', {'error': 'username or password is incorrect.'})
    else:
        return render(request, 'accounts/login.html')


def logout(request):
    auth.logout(request)
    return redirect('home')


def home(request):
    return render(request, 'accounts/home.html')


def board(request):
    # 게시판 렌더링 로직 작성
    return render(request, 'accounts/board.html')