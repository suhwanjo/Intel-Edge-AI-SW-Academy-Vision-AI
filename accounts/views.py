from django.contrib import auth
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate
from django import forms
import cv2
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
import mediapipe as mp
import time
from ultralytics import YOLO
import numpy as np
from scipy.spatial import distance as dist
import torch
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .models import BehaviorLog
from django.contrib.auth.models import User


# Create your views here.


class VideoCamera(object):
    def __init__(self, request):
        self.video = cv2.VideoCapture(0)
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # Constants
        self.EAR_THRESHOLD = 0.20
        self.MAR_THRESHOLD = 0.5  # Adjusted for more accurate detection
        self.SECONDS_TO_COUNT = 3
        self.YAWN_DURATION = 3  # 3 seconds for yawning count

        # Variables to track counts
        self.eye_closed_long_count = 0
        self.yawn_count = 0
        self.yawn_prev = 0
        self.eye_closed = False
        self.prev_eye_closed = False
        self.yawning = False
        self.yawn_start_time = None
        self.eye_closed_count = 0
        self.yawn_frame_count = 0

        # Frame rate
        self.THRESHOLD_FRAMES = 30

        self.last_save_time = time.time()
        self.save_interval = 10  # 5분 = 300초
        self.request = request

    def __del__(self):
        self.video.release()

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def mouth_aspect_ratio(self, mouth):
        W = dist.euclidean(mouth[0], mouth[1])  # 61-67
        H = dist.euclidean(mouth[2], mouth[3])  # 62-66
        mar = H/W
        return mar
        # 57, 287 = 입 너비, 0, 17 = 입 높이

    def save_behavior_log(self):
        BehaviorLog.objects.create(
            user=self.request.user,
            yawn_count=self.yawn_count,
            sleepy_count=self.eye_closed_long_count
            # 다른 행동 데이터도 추가할 수 있음
        )
        self.yawn_count = 0
        self.eye_closed_long_count = 0

    def get_frame(self):
        success, image = self.video.read()
        if not success:
            return None

        start = time.time()
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            self.save_behavior_log()
            self.last_save_time = current_time

        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False

        # Get the result
        results = self.mp_face_mesh.process(image)

        # To improve performance
        image.flags.writeable = True

        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                # Check if all necessary landmarks are detected
                necessary_indices = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380, 61, 146, 91, 181, 84,
                                     17, 314, 405, 321, 375, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
                if not all(idx < len(landmarks) for idx in necessary_indices):
                    print("Error: Not all necessary landmarks are detected.")
                    self.video.release()
                    cv2.destroyAllWindows()
                    exit(1)

                # Left and right eye landmarks
                leftEyeIndices = [33, 160, 158, 133, 153, 144]
                rightEyeIndices = [362, 385, 387, 263, 373, 380]
                mouthIndices = [57,287,0,17]

                leftEye = [landmarks[i] for i in leftEyeIndices]
                rightEye = [landmarks[i] for i in rightEyeIndices]
                mouth = [landmarks[i] for i in mouthIndices]

                leftEye = [(int(p.x * img_w), int(p.y * img_h)) for p in leftEye]
                rightEye = [(int(p.x * img_w), int(p.y * img_h)) for p in rightEye]
                mouth = [(int(p.x * img_w), int(p.y * img_h)) for p in mouth]

                # Calculate eye aspect ratio (EAR)
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                # Calculate mouth aspect ratio (MAR)
                mar = self.mouth_aspect_ratio(mouth)

                # Check if eyes are closed
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

                if self.eye_closed_count >= self.THRESHOLD_FRAMES * 2:
                    self.eye_closed_long_count += 1
                    cv2.putText(image, "Sleeping count increased", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                2)
                    self.eye_closed_count = 0

                if mar > self.yawn_prev * 2:
                    if not self.yawning:
                        self.yawning = True
                        self.yawn_frame_count = 0
                        self.yawn_prev = mar
                    else:
                        self.yawn_frame_count += 1
                        if self.yawn_frame_count >= self.THRESHOLD_FRAMES:
                            self.yawn_count += 1
                            cv2.putText(image, "Yawning count increased", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2)
                            self.yawning = False
                            self.yawn_frame_count = 0
                            self.yawn_prev = mar  # 이전 mar 값을 현재 mar 값으로 업데이트
                else:
                    if self.yawning:
                        self.yawn_frame_count += 1
                        if self.yawn_frame_count >= self.THRESHOLD_FRAMES:
                            self.yawning = False
                            self.yawn_frame_count = 0
                    else:
                        self.yawn_frame_count = 0
                        self.yawn_prev = 0  # 이전 mar 값을 현재 mar 값으로 업데이트

                for point in leftEye:
                    cv2.circle(image, point, 1, (0, 255, 0), -1)
                for point in rightEye:
                    cv2.circle(image, point, 1, (0, 255, 0), -1)
                for point in mouth:
                    cv2.circle(image, point, 1, (255, 0, 0), -1)

                cv2.putText(image, f"EAR: {ear:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(image, f"EAR Threshold: {self.EAR_THRESHOLD}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
                cv2.putText(image, f"MAR: {mar:.2f}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, f"MAR Threshold: {self.MAR_THRESHOLD}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 0), 2)

                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                        # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                # See where the user's head tilting
                if y < -10:
                    text = "Looking Left"
                elif y > 10:
                    text = "Looking Right"
                elif x < -10:
                    text = "Looking Down"
                else:
                    text = "Forward"

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix,
                                                                 dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                cv2.line(image, p1, p2, (255, 0, 0), 3)

                # Add the text on the image
                cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2)
                cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)

            cv2.putText(image, f"Sleeping Count: {self.eye_closed_long_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
            cv2.putText(image, f"Yawning Count: {self.yawn_count}", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@login_required
@gzip.gzip_page
def webcam_feed(request):
    try:
        camera = VideoCamera(request)
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
            messages.error(request, '아이디 또는 비밀번호가 잘못되었습니다.')
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