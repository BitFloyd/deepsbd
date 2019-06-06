from deepsbd.video_to_shots import VideoToShots
from clockshortenstream.process_video_pkg.clock_processes import ShortenVideoStream


input_video = '/usr/local/data/sejacob/PhD_Repo/test_data/2018-09-19-tor-ott-home.mp4'
input_srt = '/usr/local/data/sejacob/PhD_Repo/test_data/2018-09-19-tor-ott-home.srt'
directory_to_save_clips = '/usr/local/data/sejacob/PhD_Repo/test_data/MATCH_CLIPS'

output_video = '/usr/local/data/sejacob/PhD_Repo/test_data/shortened_video.mp4'
output_srt = '/usr/local/data/sejacob/PhD_Repo/test_data/shortened_video.srt'

svs = ShortenVideoStream(path_to_input_video=input_video,path_to_input_srt=input_srt,commercial_removed=False)
svs.shorten_video_stream(path_to_output_video=output_video,path_to_output_srt=output_srt)

vts = VideoToShots(path_to_video=output_video)
vts.fit()

print ("##############################################")
print ("FULL_TRANS:")
print (vts.full_trans)
print ("##############################################")

vts.save_video_as_shots(directory_to_save_clips)
