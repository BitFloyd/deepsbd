from deepsbd.video_to_shots import VideoToShots
from clockshortenstream.process_video_pkg.clock_processes import ShortenVideoStream


input_video = '/usr/local/data/sejacob/PhD_Repo/test_data/2018-09-19-tor-ott-home.mp4'
input_srt = '/usr/local/data/sejacob/PhD_Repo/test_data/2018-09-19-tor-ott-home.srt'
directory_to_save_clips = '/usr/local/data/sejacob/PhD_Repo/test_data/MATCH_CLIPS'

shortened_video = '/usr/local/data/sejacob/PhD_Repo/test_data/shortened_video.mp4'
shortened_video_srt = '/usr/local/data/sejacob/PhD_Repo/test_data/shortened_video.srt'

svs = ShortenVideoStream(path_to_input_video=input_video,path_to_input_srt=input_srt,commercial_removed=False)
svs.shorten_video_stream(path_to_output_video=shortened_video,path_to_output_srt=shortened_video_srt)

# Step 3: Convert video to shots
vts = VideoToShots(path_to_video=shortened_video, path_to_video_srt=shortened_video_srt)
vts.fit()

print ("##############################################")
print ("FULL_TRANS:")
print (vts.full_trans)
print ("##############################################")
