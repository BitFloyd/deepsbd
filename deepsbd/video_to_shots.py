from framefilter import perform_frame_filtration
from keras.models import load_model
from config import n_frame_added,cut_detector,grad_detector,grad_n_frames_per_sample
from read_video_cuboids import read_cuboid_from_video_cut_detection,get_SIM_for_grad_candidate
from read_video_cuboids import get_frame_start_for_grad_cuboids
import os
import numpy as np
from clockshortenstream.process_video_pkg.frame_writer import FrameWriter
from clockshortenstream.process_video_pkg.frame_reader import Stream



class VideoToShots:

    def __init__(self,path_to_video):
        self.path_to_video = path_to_video
        self.candidates = []
        self.cuts = []
        self.grads = []
        self.candidates_no_cut = []
        self.full_trans = []

    def perform_frame_filtration(self):
        self.candidates = perform_frame_filtration(self.path_to_video)

        return self.candidates

    def perform_cut_detection(self):

        cut_model = load_model(cut_detector)
        self.cuts = []

        for i in self.candidates:
            range_start = i - n_frame_added
            range_end = i + n_frame_added

            frame_nums_list = range(range_start, range_end + 1)

            cuboid = read_cuboid_from_video_cut_detection(video_path=self.path_to_video, frame_nums_list=frame_nums_list)

            prediction = cut_model.predict(cuboid)[0][0]

            if(prediction>0.5):
                self.cuts.append(i)
            else:
                self.candidates_no_cut.append(i)


        return True

    def perform_grad_detection(self):

        grad_model = load_model(grad_detector)
        self.grads = []

        for i in self.candidates_no_cut:
            frame_start = get_frame_start_for_grad_cuboids(i)
            sim_image = get_SIM_for_grad_candidate(video_path=self.path_to_video,frame_candidate=i)

            prediction = grad_model.predict(sim_image)

            class_output = prediction[0][0]
            reg_output = prediction[1][0]

            if(class_output>0.5):
                self.grads.append(frame_start+np.int(reg_output*grad_n_frames_per_sample))


        return True

    def fit(self):

        self.perform_frame_filtration()
        self.perform_cut_detection()
        self.perform_grad_detection()

        self.full_trans = []
        self.full_trans.extend(self.cuts)
        self.full_trans.extend(self.grads)
        self.full_trans.sort()

        return self.full_trans

    def get_video_name_from_id(self,video_id):

        return 'shot_'+str(video_id)+'.mp4'


    def save_video_as_shots(self,out_path_for_video):

        if not (os.path.exists(out_path_for_video)):
            os.mkdir(out_path_for_video)

        video_id = 0
        video_name = self.get_video_name_from_id(video_id)

        fReader = Stream(self.path_to_video,time_resolution=None)
        frame = fReader.readNextFrameFromVideo()

        fWriter = FrameWriter(out_path_for_video,video_name,frame_size=(frame.shape[0],frame.shape[1]),video_fps=fReader.frameReader.videoFPS)
        fWriter.openVideoStream()

        trans_to_write = list(self.full_trans)

        trans_to_write.append(fReader.frameReader.numFrames)


        while len(trans_to_write) > 0 and not fReader.videoFinished:
            fWriter.writeFrameToVideo(frame)

            if(fReader.frameReader.frameNumber ==trans_to_write[0]):
                fWriter.closeVideoStream()
                trans_to_write.pop(0)
                if(not len(trans_to_write)>0):
                    break
                video_id+=1
                video_name=self.get_video_name_from_id(video_id)
                fWriter=FrameWriter(out_path_for_video,video_name,frame_size=(frame.shape[0],frame.shape[1]),video_fps=fReader.frameReader.videoFPS)
                fWriter.openVideoStream()

            frame = fReader.readNextFrameFromVideo()


        return True
