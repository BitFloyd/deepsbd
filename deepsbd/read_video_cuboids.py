from threading import Thread

import cv2
import numpy as np
from clockshortenstream.process_video_pkg.frame_reader import FrameReader
from skimage.transform import resize

from config import frame_size, grad_n_frames_per_sample, n_frame_added


def read_cuboid_from_video_cut_detection(video_path, frame_nums_list):
    fReader = FrameReader(pathToVideo=video_path)

    frames = []

    for frame_num in frame_nums_list:
        frame = fReader.getFrameAtFrameNumber(frame_num)
        frame_resized = resize(frame, frame_size)
        frames.append(frame_resized)

    fReader.closeFrameReader()

    cuboid = np.array(frames)
    cuboid = np.expand_dims(cuboid, axis=0)

    return cuboid


def get_frame_start_for_grad_cuboids(frame_candidate):
    frame_start = frame_candidate - grad_n_frames_per_sample / 2

    return frame_start


def read_frame_cuboid_from_video_grad(video_path, frame_candidate):
    fReader = FrameReader(pathToVideo=video_path)
    frame_start = get_frame_start_for_grad_cuboids(frame_candidate)
    frames = fReader.getNumberOfFramesFromPosition(start_frame_id=frame_start,
                                                   num_frames=grad_n_frames_per_sample)
    for idx, frame in enumerate(frames):
        frames[idx] = cv2.resize(frame, frame_size)

    fReader.closeFrameReader()

    return frames


def get_cuboid_for_grad_candidate(video_path, frame_candidate):
    grad_cuboid = read_frame_cuboid_from_video_grad(video_path, frame_candidate)
    grad_cuboid = np.expand_dims(grad_cuboid, axis=0)

    return grad_cuboid


class AppendCUBThread(Thread):

    def __init__(self, cubs, candidate, path_to_video):
        Thread.__init__(self)
        self.daemon = True
        self.cubs = cubs
        self.path_to_video = path_to_video
        self.candidate = candidate
        self.finished = False

    def run(self):
        frame_start = get_frame_start_for_grad_cuboids(self.candidate)
        grad_cuboid = get_cuboid_for_grad_candidate(self.path_to_video, self.candidate)
        self.cubs.append((grad_cuboid, frame_start))

        self.finished = True


class AppendCUTCUBThread(Thread):

    def __init__(self, cubs, candidate, path_to_video):
        Thread.__init__(self)
        self.daemon = True
        self.cubs = cubs
        self.path_to_video = path_to_video
        self.candidate = candidate
        self.finished = False

    def run(self):
        range_start = self.candidate - n_frame_added
        range_end = self.candidate + n_frame_added

        frame_nums_list = range(range_start, range_end + 1)

        cut_cuboid = read_cuboid_from_video_cut_detection(video_path=self.path_to_video,
                                                          frame_nums_list=frame_nums_list)
        self.cubs.append((cut_cuboid, self.candidate))

        self.finished = True
