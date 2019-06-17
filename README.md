# deepsbd
This is a package to do shot boundary detection on videos. 

Required packages:
------------------
keras-squeezenet (https://github.com/BitFloyd/keras-squeezenet)

cv2

skimage

Keras

Tensorflow

clockshortenstream (https://github.com/BitFloyd/clockshortenstream)


**Installation**
===================
Navigate to the deepsbd folder


pip install -e .

Usage Example:
------------------

```

from deepsbd.video_to_shots import VideoToShots

video_path = 'video.mp4'
video_srt_path = 'video.srt' #Set this to None if you dont need to crop subtitles.
directory_to_save_shots_in = 'shots'
# Step 3: Convert video to shots
vts = VideoToShots(path_to_video=video_path, path_to_video_srt=video_srt_path)


#Perform fit
vts.fit()

print "Transition Candidates: " vts.candidates
print "Cut Transitions: ",vts.cuts
print "Gradual Transitions:", vts.grads

print "##############################################"
print "ALL_TRANSITION_FRAMES:"
print vts.full_trans
print "##############################################"

vts.save_video_as_shots(directory_to_save_shots_in)

```
