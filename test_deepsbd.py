from deepsbd.video_to_shots import VideoToShots

video_name = 'FP.mp4'

vts = VideoToShots(path_to_video=video_name)
vts.fit()

print "##############################################"
print "FULL_TRANS:"
print vts.full_trans
print "##############################################"

vts.save_video_as_shots('FP')
