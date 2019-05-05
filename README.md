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
vts = VideoToShots(path_to_video)

#Perform fit
vts.fit()

print "Transition Candidates: " vts.candidates
print "Cut Transitions: ",vts.cuts
print "Gradual Transitions:", vts.grads

```
