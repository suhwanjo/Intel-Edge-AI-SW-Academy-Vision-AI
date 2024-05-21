from django.contrib import auth
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate
from django import forms
import cv2
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
import mediapipe as mp
import time as pytime
from ultralytics import YOLO
import numpy as np
from scipy.spatial import distance as dist
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
import pickle
from datetime import datetime, time
from .models import BehaviorLog
from .forms import DateTimeForm
from django.db.models import Sum
from datetime import timedelta
# Create your views here.


class VideoCamera(object):
    def __init__(self, request):
        self.video = cv2.VideoCapture(0)
        self.model = YOLO("models/yolov8n-pose.pt")
        with open('models/model_knn.h5', 'rb') as f:
            self.model_knn = pickle.load(f)
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # 졸음, 하품 관련 변수
        self.EAR_THRESHOLD = 0.2
        self.MAR_THRESHOLD = 0.5

        self.eye_closed_long_count = 0
        self.eye_closed = False
        self.prev_eye_closed = False
        self.eye_closed_count = 0

        self.yawn_count = 0
        self.yawn_prev = 0
        self.yawning = False
        self.yawn_frame_count = 0

        # Variables to track counts
        self.DIRECTION_DURATION = 3

        self.right_count = 0
        self.right_start_time = None
        self.bRight = False

        self.left_count = 0
        self.left_start_time = None
        self.bLeft = False

        self.down_count = 0
        self.down_start_time = None
        self.bDown = False

        self.down_cnt_dir = 0
        # Frame rate
        self.THRESHOLD_FRAMES = 30

        self.last_save_time = pytime.time()
        self.last_pose_inference_time = pytime.time()
        self.save_interval = 300  # 저장 및 자세 추정 주기
        self.request = request

        self.jungjasae_count = 0
        self.gubujung_count = 0

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

    # 로그 DB에 저장
    def save_behavior_log(self):
        BehaviorLog.objects.create(
            user=self.request.user,
            yawn_count=self.yawn_count,
            sleepy_count=self.eye_closed_long_count,
            gaze_left_count=self.left_count,
            gaze_right_count=self.right_count,
            gaze_down_count=self.down_cnt_dir,
            gaze_down_long_count=self.down_count,
            pose_good=self.jungjasae_count,
            pose_bad=self.gubujung_count
            # 다른 행동 데이터도 추가할 수 있음
        )
        self.yawn_count = 0
        self.eye_closed_long_count = 0
        self.right_count = 0
        self.left_count = 0
        self.down_count = 0
        self.down_cnt_dir = 0
        self.jungjasae_count=0
        self.gubujung_count=0

    def get_frame(self):
        success, image = self.video.read()
        if not success:
            return None

        # 특정 시간 마다 로그 DB에 저장
        start = pytime.time()
        current_time = pytime.time()
        if current_time - self.last_save_time >= self.save_interval:
            self.save_behavior_log()
            self.last_save_time = current_time



        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.mp_face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        # 졸음, 하품, 시선 인식
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                necessary_indices = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380, 61, 146, 91, 181, 84,
                                     17, 314, 405, 321, 375, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
                if not all(idx < len(landmarks) for idx in necessary_indices):
                    self.video.release()
                    cv2.destroyAllWindows()
                    exit(1)

                # 눈 랜드마크
                leftEyeIndices = [33, 160, 158, 133, 153, 144]
                leftEye = [landmarks[i] for i in leftEyeIndices]
                leftEye = [(int(p.x * img_w), int(p.y * img_h)) for p in leftEye]

                rightEyeIndices = [362, 385, 387, 263, 373, 380]
                rightEye = [landmarks[i] for i in rightEyeIndices]
                rightEye = [(int(p.x * img_w), int(p.y * img_h)) for p in rightEye]
                # EAR 계산
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                # 입 랜드마크
                mouthIndices = [57, 287, 0, 17]
                mouth = [landmarks[i] for i in mouthIndices]
                mouth = [(int(p.x * img_w), int(p.y * img_h)) for p in mouth]
                # MAR 계산
                mar = self.mouth_aspect_ratio(mouth)

                # 졸음 인식 알고리즘
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

                if self.eye_closed_count >= self.THRESHOLD_FRAMES * 3:
                    self.eye_closed_long_count += 1
                    self.eye_closed_count = 0
                # 졸음 시각화
                for point in leftEye:
                    cv2.circle(image, point, 1, (0, 255, 0), -1)
                for point in rightEye:
                    cv2.circle(image, point, 1, (0, 255, 0), -1)
                cv2.putText(image, f"EAR: {ear:.2f}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # 하품 인식 알고리즘
                if mar > self.yawn_prev * 2:
                    if not self.yawning:
                        self.yawning = True
                        self.yawn_frame_count = 0
                        self.yawn_prev = mar
                    else:
                        self.yawn_frame_count += 1
                        if self.yawn_frame_count >= self.THRESHOLD_FRAMES:
                            self.yawn_count += 1
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
                # 하품 시각화
                for point in mouth:
                    cv2.circle(image, point, 1, (255, 0, 0), -1)
                cv2.putText(image, f"MAR: {mar:.2f}", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # 시선 방향 인식 알고리즘
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        face_2d.append([x, y])

                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)

                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                rmat, jac = cv2.Rodrigues(rot_vec)

                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                if y < -10:
                    text = "Looking Left"
                    self.bRight = False
                    if self.bDown: self.down_cnt_dir += 1; self.bDown = False
                    if not self.bLeft:
                        self.bLeft = True
                        self.left_start_time = pytime.time()
                    elif pytime.time() - self.left_start_time > self.DIRECTION_DURATION:
                        self.left_count += 1
                        self.bLeft = False
                elif y > 10:
                    text = "Looking Right"
                    self.bLeft = False
                    if self.bDown: self.down_cnt_dir += 1; self.bDown = False
                    if not self.bRight:
                        self.bRight = True
                        self.right_start_time = pytime.time()
                    elif pytime.time() - self.right_start_time > self.DIRECTION_DURATION:
                        self.right_count += 1
                        self.bRight = False
                elif x < -10:
                    text = "Looking Down"
                    self.bLeft = False
                    self.bRight = False
                    if not self.bDown:
                        self.bDown = True
                        self.down_start_time = pytime.time()
                    elif pytime.time() - self.down_start_time > self.DIRECTION_DURATION:
                        self.down_count += 1
                        self.bDown = False
                else:
                    text = "Forward"
                    if self.bDown: self.down_cnt_dir += 1; self.bDown = False
                    self.bLeft = False
                    self.bRight = False
                # 시선 방향 시각화
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix,
                                                                 dist_matrix)
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
                cv2.line(image, p1, p2, (255, 0, 0), 3)
                # Add the text on the image
                cv2.putText(image, text, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, "x: " + str(np.round(x, 2)), (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(image, "y: " + str(np.round(y, 2)), (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                            2)
                cv2.putText(image, "z: " + str(np.round(z, 2)), (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),

                            2)
        # 임계값마다 포즈 추론
        if current_time - self.last_pose_inference_time >= 5:#(self.save_interval / 5):
            results = self.model(image, conf=0.7)
            #image = results[0].plot()
            if results[0].keypoints is not None:
                keypoints = results[0].keypoints.xy[0]
            # YOLOv8-pose에서 얻은 keypoints가 유효한 경우
            if keypoints is not None and keypoints.shape[0] == 17:
                # 필요한 keypoints가 모두 검출된 경우
                if keypoints[2, 1] != 0 and keypoints[4, 1] != 0 and keypoints[5, 1] != 0:
                    # 추출한 좌표에서 필요한 y좌표 추출
                    nose_y = keypoints[0, 1]  # 코 y좌표
                    L_eye_y = keypoints[2, 1]  # 왼쪽 눈 y좌표
                    L_ear_y = keypoints[4, 1]  # 왼쪽 귀 y좌표
                    R_shoulder_y = keypoints[5, 1]  # 오른쪽 어깨 y좌표

                    # KNN 모델로 자세 분류
                    input_data = np.array([[nose_y, L_eye_y, L_ear_y, R_shoulder_y]])
                    pred = self.model_knn.predict(input_data)

                    # 결과에 따라 count 증가
                    if pred[0] == 1:
                        self.jungjasae_count += 1
                    elif (pred[0] == 0 or pred[0] == 2) and text == "Looking Down":
                        self.jungjasae_count += 1
                    else:
                        self.gubujung_count += 1
                    # 정 자세                 -> 정 자세
                    # 고개를 숙임 and 시선 아래 -> 정 자세
                    # 구부정 and 시선 아래     -> 정 자세

            self.last_pose_inference_time = current_time

        cv2.putText(image, f"Sleeping Count: {self.eye_closed_long_count}", (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        cv2.putText(image, f"Yawning Count: {self.yawn_count}", (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255),
                    2)
        cv2.putText(image, f"Gaze Direction: R:{self.right_count}, L:{self.left_count} D:{self.down_count}, D_cnt:{self.down_cnt_dir}", (10, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255),
                    2)
        cv2.putText(image,
                    f"Pose: Good:{self.jungjasae_count}, Bad:{self.gubujung_count}",(10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255),
                    2)

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


def get_period_time(period):
    period_mapping = {
        '': (time(9, 0), time(16, 50)),
        '1': (time(9, 0), time(9, 50)),
        '2': (time(10, 0), time(10, 50)),
        '3': (time(11, 0), time(11, 50)),
        '4': (time(13, 0), time(13, 50)),
        '5': (time(14, 0), time(14, 50)),
        '6': (time(15, 0), time(15, 50)),
        '7': (time(16, 0), time(16, 50)),
    }
    return period_mapping.get(period)


@login_required
def dashboard_view(request):
    if request.method == 'POST':
        form = DateTimeForm(request.POST)
        if form.is_valid():
            date = form.cleaned_data['date']
            period = form.cleaned_data.get('period')

            logs = BehaviorLog.objects.filter(
                user=request.user,
                timestamp__date=date
            )

            if logs.exists():
                # 5분 간격으로 데이터 가져오기
                start_time, end_time = get_period_time(period) #

                time_intervals = []
                current_time = start_time
                current_datetime = datetime.combine(date, current_time)
                while current_datetime.time() <= end_time:
                    time_intervals.append((current_datetime, current_datetime + timedelta(minutes=5)))
                    current_datetime += timedelta(minutes=5)

                aggregated_data = {
                    'time_labels': [],
                    'total_yawn': [],
                    'total_sleepy': [],
                    'total_gaze_left': [],
                    'total_gaze_right': [],
                    'total_gaze_down': [],
                    'total_gaze_down_long': [],
                    'total_pose_good': [],
                    'total_pose_bad': [],
                }

                for start, end in time_intervals:
                    # 해당 시간 범위 내의 데이터 가져오기
                    interval_logs = logs.filter(timestamp__range=(start, end))

                    # 해당 시간에 대한 행동 정보 합산
                    total_yawn = interval_logs.aggregate(total_yawn=Sum('yawn_count'))['total_yawn'] or 0
                    total_sleepy = interval_logs.aggregate(total_sleepy=Sum('sleepy_count'))['total_sleepy'] or 0
                    total_gaze_left = interval_logs.aggregate(total_gaze_left=Sum('gaze_left_count'))['total_gaze_left'] or 0
                    total_gaze_right = interval_logs.aggregate(total_gaze_right=Sum('gaze_right_count'))['total_gaze_right'] or 0
                    total_gaze_down = interval_logs.aggregate(total_gaze_down=Sum('gaze_down_count'))['total_gaze_down'] or 0
                    total_gaze_down_long = interval_logs.aggregate(total_gaze_down_long=Sum('gaze_down_long_count'))['total_gaze_down_long'] or 0
                    total_pose_good = interval_logs.aggregate(total_pose_good=Sum('pose_good'))['total_pose_good'] or 0
                    total_pose_bad = interval_logs.aggregate(total_pose_bad=Sum('pose_bad'))['total_pose_bad'] or 0

                    # 결과를 aggregated_data에 추가
                    aggregated_data['time_labels'].append(start.strftime('%H:%M'))
                    aggregated_data['total_yawn'].append(total_yawn)
                    aggregated_data['total_sleepy'].append(total_sleepy)
                    aggregated_data['total_gaze_left'].append(total_gaze_left)
                    aggregated_data['total_gaze_right'].append(total_gaze_right)
                    aggregated_data['total_gaze_down'].append(total_gaze_down)
                    aggregated_data['total_gaze_down_long'].append(total_gaze_down_long)
                    aggregated_data['total_pose_good'].append(total_pose_good)
                    aggregated_data['total_pose_bad'].append(total_pose_bad)

                aggregated_data2 = {
                    'total_yawn': sum(aggregated_data['total_yawn']),
                    'total_sleepy': sum(aggregated_data['total_sleepy']),
                    'total_gaze_left': sum(aggregated_data['total_gaze_left']),
                    'total_gaze_right': sum(aggregated_data['total_gaze_right']),
                    'total_gaze_down': sum(aggregated_data['total_gaze_down']),
                    'total_gaze_down_long': sum(aggregated_data['total_gaze_down_long']),
                    'total_pose_good': sum(aggregated_data['total_pose_good']),
                    'total_pose_bad': sum(aggregated_data['total_pose_bad']),
                }
                return render(request, 'accounts/dashboard.html', {'form': form, 'aggregated_data': aggregated_data, 'aggregated_data2': aggregated_data2})
            else:
                # 데이터가 없을 때 처리
                return render(request, 'accounts/dashboard.html', {'form': form, 'message': 'No data available for the selected date.'})
    else:
        form = DateTimeForm()
        return render(request, 'accounts/dashboard.html', {'form': form})


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
    return render(request,
                  'accounts/signup.html',
                  {'form': form})


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