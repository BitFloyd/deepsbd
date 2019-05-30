import os
import time

import numpy as np
from clockshortenstream.process_video_pkg.frame_reader import Stream
from moviepy.editor import VideoFileClip
from keras.models import load_model
from tqdm import tqdm

from config import cut_detector, grad_detector, grad_n_frames_per_sample, grad_n_threads, cut_n_threads
from framefilter import perform_frame_filtration
from read_video_cuboids import AppendCUBThread, AppendCUTCUBThread


class VideoToShots:

    def __init__(self, path_to_video, verbose=True):
        self.path_to_video = path_to_video
        self.candidates = []
        self.cuts = []
        self.grads = []
        self.candidates_no_cut = []
        self.full_trans = []
        self.verbose = verbose

    def perform_frame_filtration(self):
        self.candidates = perform_frame_filtration(self.path_to_video)

        return self.candidates

    def perform_cut_detection(self):

        cut_model = load_model(cut_detector)
        self.cuts = []
        cubs = []

        if (self.verbose):
            print ("###################################################")
            print ("PERFORMING CUT DETECTION")
            print ("###################################################")

        n_threads_running = []

        candidates_cut_split = np.array_split(self.candidates,
                                              len(self.candidates) / cut_n_threads + 1)

        for split in tqdm(candidates_cut_split):

            for i in tqdm(split):
                n_threads_running.append(AppendCUTCUBThread(cubs, i, self.path_to_video))
                n_threads_running[-1].start()

            while self.are_any_threads_running(n_threads_running):
                time.sleep(0.0001)

            n_threads_running = []

        for i in cubs:
            cub, frame = i
            prediction = cut_model.predict(cub)

            class_output = prediction[0][0]

            if (class_output > 0.5):
                self.cuts.append(frame)
            else:
                self.candidates_no_cut.append(frame)

        return True

    def are_any_threads_running(self, n_threads_running):
        for thread in n_threads_running:
            if thread.finished == False:
                return True

        return False

    def perform_grad_detection(self):

        grad_model = load_model(grad_detector)
        self.grads = []
        cubs = []

        if (self.verbose):
            print ("###################################################")
            print ("PERFORMING GRAD DETECTION")
            print ("###################################################")

        n_threads_running = []

        candidates_no_cut_split = np.array_split(self.candidates_no_cut,
                                                 len(self.candidates_no_cut) / grad_n_threads + 1)

        for split in tqdm(candidates_no_cut_split):

            for i in tqdm(split):
                n_threads_running.append(AppendCUBThread(cubs, i, self.path_to_video))
                n_threads_running[-1].start()

            while self.are_any_threads_running(n_threads_running):
                time.sleep(0.0001)

            n_threads_running = []

        for i in cubs:
            cub, frame_start = i
            prediction = grad_model.predict(cub)

            class_output = prediction[0][0]
            reg_output = prediction[1][0]

            if (class_output > 0.5):
                self.grads.append(frame_start + np.int(reg_output * grad_n_frames_per_sample))

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

    def get_video_name_from_id(self, video_id):

        return 'shot_' + str(video_id) + '.mp4'

    def save_video_as_shots(self, out_path_for_video):

        if not (os.path.exists(out_path_for_video)):
            os.mkdir(out_path_for_video)

        video_id = 0

        video_name = self.get_video_name_from_id(video_id)

        fReader = Stream(self.path_to_video, time_resolution=None)

        time_resolution = fReader.time_resolution

        fReader.close_Stream()

        trans_to_write = list(self.full_trans)

        trans_to_write = [-1]+[trans_to_write]

        trans_to_write.append(fReader.frameReader.numFrames)

        for i,j in zip(trans_to_write,trans_to_write[1:]):

            print i,j
            clip = VideoFileClip(self.path_to_video).subclip(time_resolution*(i+1), time_resolution*(j))
            clip.write_videofile(video_name)
            video_id += 1
            video_name = self.get_video_name_from_id(video_id)


        return True
