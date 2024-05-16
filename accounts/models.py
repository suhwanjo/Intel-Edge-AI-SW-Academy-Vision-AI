from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class BehaviorLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='behavior_logs')
    yawn_count = models.PositiveIntegerField(default=0)
    sleepy_count = models.PositiveIntegerField(default=0)
    gaze_left_count = models.PositiveIntegerField(default=0)
    gaze_right_count = models.PositiveIntegerField(default=0)
    gaze_down_count = models.PositiveIntegerField(default=0)
    gaze_down_long_count = models.PositiveIntegerField(default=0)
    pose_good = models.PositiveIntegerField(default=0)
    pose_bad = models.PositiveIntegerField(default=0)
    timestamp = models.DateTimeField(auto_now_add=True)