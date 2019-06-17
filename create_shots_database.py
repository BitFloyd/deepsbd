from deepsbd.video_to_shots import VideoToShots
import os
import sys
from clockshortenstream.process_video_pkg.clock_processes import ShortenVideoStream

videos_directory = '/usr/local/data/sejacob/DATASETS/HOCKEY/ALL_GAMES/GAME_VIDEOS'
scrap_directory = '/usr/local/data/sejacob/DATASETS/HOCKEY/ALL_GAMES/SCRAP_DIRECTORY'
clips_database_directory = '/usr/local/data/sejacob/DATASETS/HOCKEY/ALL_GAMES/CLIPS_DATABASE'

list_of_videos = [os.path.join(videos_directory,i) for i in os.listdir(videos_directory)]

for video in list_of_videos:

    #Step 1: Convert video to MP4 file and SRT file

    ts_file = video
    filename, ext = os.path.splitext(ts_file)

    _, filename_short = os.path.split(filename)
    directory_to_save_clips = os.path.join(clips_database_directory, filename_short)

    if(os.path.exists(directory_to_save_clips)):
        continue

    mp4_file = filename+'.mp4'
    _, mp4_file = os.path.split(mp4_file)
    mp4_file = os.path.join(scrap_directory,mp4_file)

    ffmpeg_command_video_convert = 'ffmpeg -i {ts_file} -c:v libx264 -c:a copy {mp4_file}'.format(ts_file=ts_file, mp4_file=mp4_file)

    if(not os.path.exists(mp4_file)):
        os.system(ffmpeg_command_video_convert)

    srt_file = filename+'.srt'
    _, srt_file = os.path.split(srt_file)
    srt_file = os.path.join(scrap_directory,srt_file)

    ffmpeg_command_video_to_srt = 'ffmpeg -f lavfi -i movie={ts_file}[out+subcc]  -map 0:1  {sub_file}'.format(ts_file=ts_file, sub_file=srt_file)

    if(not os.path.exists(srt_file)):
        os.system(ffmpeg_command_video_to_srt)

    #Step 2: Shorten this video to a file without commercials and random useless scenes

    short_video = os.path.join(scrap_directory,'shortened_video.mp4')
    short_srt = os.path.join(scrap_directory,'short_srt.srt')

    try:
        svs = ShortenVideoStream(path_to_input_video=mp4_file, path_to_input_srt=srt_file, commercial_removed=False)
        svs.shorten_video_stream(path_to_output_video=short_video, path_to_output_srt=short_srt)

        #Step 3: Convert video to shots
        vts = VideoToShots(path_to_video=short_video,path_to_video_srt=short_srt)
        vts.fit()

        print ("##############################################")
        print ("FULL_TRANS:")
        print (vts.full_trans)
        print ("##############################################")

        _, filename_short = os.path.split(filename)
        directory_to_save_clips = os.path.join(clips_database_directory, filename_short)
        os.mkdir(directory_to_save_clips)

        vts.save_video_as_shots(directory_to_save_clips)
    except Exception as e:
        with open('./error_log.txt', 'a+') as f:
            f.write("----------------------------------\n")
            f.write('{filename} did not work because of this error: {error} \n'.format(filename=filename_short,error=e))
            f.write("-----------------------------------\n")

    #Step 4: Cleanup the residual files created from each video.
    remove_residual_commands = 'rm -rf {scrap_folder}/*'.format(scrap_folder=scrap_directory)
    os.system(remove_residual_commands)


