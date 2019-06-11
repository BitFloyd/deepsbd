import matplotlib.pyplot as plt
import numpy as np
from config import scale_merge_interval, frame_size
import cv2
from clockshortenstream.process_video_pkg.frame_reader import Stream
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras_squeezenet import SqueezeNet
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import cosine
from tqdm import tqdm

t = 0.5
sigma = 0.05


def features_from_video(path_to_video):
    stream = Stream(path_to_input_video=path_to_video, time_resolution=None)

    fps = int(stream.frameReader.videoFPS)

    sqnet = get_sqnet()

    list_features = []
    pbar = tqdm(total=stream.num_read_iterations)

    frame = stream.readNextFrameFromVideo()

    while (not stream.videoFinished):
        frame = cv2.resize(frame, frame_size)
        list_features.append(squeezenet_feature_from_frame(sqnet, frame))
        frame = stream.readNextFrameFromVideo()
        pbar.update(1)

    pbar.close()

    stream.close_Stream()

    return list_features, fps


def get_sqnet():
    sqnet = SqueezeNet(include_top=False,pooling='max')

    return sqnet


def squeezenet_feature_from_frame(sqnet, frame):
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    feature = sqnet.predict(frame)
    feature = feature.flatten()

    return feature


def multiscale_sampling(list_features, scale):
    return list_features[::scale]


def get_list_of_dissimilarity_metric(list_of_features):
    list_dissimilarity_metric = []

    for pairs in zip(list_of_features, list_of_features[1:]):
        list_dissimilarity_metric.append(cosine(pairs[0], pairs[1]))

    return list_dissimilarity_metric


def adaptive_threshold(list_dissimilarity_metric, n):
    T = t + sigma * np.mean(np.array(list_dissimilarity_metric))

    if (list_dissimilarity_metric[n] > T):
        return True
    else:
        return False


def merge_scales(candidates_scale_dict, interval):
    list_merged = []
    scales = candidates_scale_dict.keys()
    scales.sort(reverse=True)

    for i in range(0, len(scales)):

        list_high_scale = candidates_scale_dict[scales[i]]

        list_lower_scales = []

        for j in range(i + 1, len(scales)):
            list_lower_scales.extend(candidates_scale_dict[scales[j]])

        # Compare the higher scale list with the combined lower scale list.

        for list_index, frame_index in enumerate(list_high_scale):
            frame_present_in_lower = False
            for f in list_lower_scales:
                if frame_index in range(f - interval, f + interval + 1):
                    frame_present_in_lower = True
                    break

            if (not frame_present_in_lower):
                list_merged.append(frame_index)

    list_merged.sort()

    return list_merged


def get_scale_candidates(features_from_scale, frame_indexes_from_scale, a=20):
    list_scale_candidates = []

    initial_n = a - 1
    final_n = len(features_from_scale) - 1 - a

    idx_to_threshold = (2 * a - 1) / 2

    for n in range(initial_n, final_n + 1):
        sample_from_window = features_from_scale[n - a + 1: n + a + 1]
        list_dissimilarity_metric = get_list_of_dissimilarity_metric(sample_from_window)

        if adaptive_threshold(list_dissimilarity_metric, idx_to_threshold):
            list_scale_candidates.append(frame_indexes_from_scale[n] + 1)

    return list_scale_candidates


def remove_consecutive_candidates_from_scale(scale, candidates):

    i = 0
    while i < len(candidates)-1:
        if ((candidates[i + 1] - candidates[i]) <= scale):
            candidates.pop(i + 1)
            continue
        else:
            i += 1

    return candidates

def perform_frame_filtration(path_to_video):
    list_features, fps = features_from_video(path_to_video)

    scales = [1, 2, 4, 8, 16, 32]

    frame_indexes = range(0, len(list_features))

    candidates_scale_dict = {}

    for scale in scales:
        features_from_scale = multiscale_sampling(list_features, scale)
        frame_indexes_from_scale = multiscale_sampling(frame_indexes, scale)

        candidates_in_scale = get_scale_candidates(features_from_scale, frame_indexes_from_scale, a=fps / 4)
        candidates_in_scale.sort()

        if(scale<scale_merge_interval):
            candidates_in_scale = remove_consecutive_candidates_from_scale(scale,candidates_in_scale)

        candidates_scale_dict[scale] = candidates_in_scale

    list_frame_candidates = merge_scales(candidates_scale_dict, interval=scale_merge_interval)
    print "----------------------------------"
    print "LIST_FRAME_CANDIDATES:"
    print "----------------------------------"
    print list_frame_candidates
    return list_frame_candidates


def debug_plot_CandidatesinPDF(path_to_video):
    list_candidates = perform_frame_filtration(path_to_video)

    stream = Stream(path_to_input_video=path_to_video, time_resolution=None)

    frame_id = 0

    pbar = tqdm(total=stream.num_read_iterations)

    frame = stream.readNextFrameFromVideo()
    frame_id += 1

    with PdfPages('debug_frames.pdf') as pdf:
        while (not stream.videoFinished):

            if frame_id in list_candidates:
                fig, (ax1, ax2) = plt.subplots(2, 1)
                ax1.imshow(frame)

                next_frame = stream.readNextFrameFromVideo()
                frame_id += 1
                ax2.imshow(next_frame)

                pdf.savefig()
                plt.close()

            frame = stream.readNextFrameFromVideo()
            frame_id += 1
            pbar.update(1)

    pbar.close()

    stream.close_Stream()
