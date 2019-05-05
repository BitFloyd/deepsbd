from framefilter import perform_frame_filtration
from keras.models import load_model
from config import n_frame_added,cut_detector,grad_detector,grad_n_frames_per_sample
from read_video_cuboids import read_cuboid_from_video_cut_detection,get_SIM_for_grad_candidate
from read_video_cuboids import get_frame_start_for_grad_cuboids
import numpy as np



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

        return self.full_trans
