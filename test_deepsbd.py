from deepsbd.video_to_shots import VideoToShots
from clockshortenstream.process_video_pkg.clock_processes import ShortenVideoStream


video_name_full_match = '/usr/local/data/sejacob/PhD_Repo/test_data/2018-09-19-tor-ott-home.ts'
video_name_wo_commercial =  '/usr/local/data/sejacob/PhD_Repo/test_data/tor_ott_no_com.mp4'
shortened_video = '/usr/local/data/sejacob/PhD_Repo/test_data/tor_ott_shortened.mp4'
directory_to_save_clips = '/usr/local/data/sejacob/PhD_Repo/test_data/MATCH_CLIPS'


svs = ShortenVideoStream(path_to_input_video=video_name_wo_commercial)
svs.shorten_video_stream(path_to_output_video=shortened_video)

vts = VideoToShots(path_to_video=shortened_video)
vts.fit()

print ("##############################################")
print ("FULL_TRANS:")
print (vts.full_trans)
print ("##############################################")

vts.save_video_as_shots(directory_to_save_clips)
