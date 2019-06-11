import multiprocessing
import os
# ================================== #
# Frame Filter
# ================================== #
scale_merge_interval = 30
# ================================== #
# Cut Detector
# ================================== #
frame_size = (128, 128)
n_frame_added = 6
batch_size = 32
cut_detector = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'cut_detector.h5')
num_iteration_in_one_epoch = 1e3
cut_n_threads = multiprocessing.cpu_count() - 3
cut_confidence_threshold = 0.35

# ================================== #
# Grad Detector
# ================================== #
grad_frame_size = (128, 128)
grad_n_frames_per_sample = 60
grad_batch_size = 256
grad_detector = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'grad_detector.h5')
grad_num_iteration_in_one_epoch = 1e2
grad_n_threads = multiprocessing.cpu_count() - 3
grad_confidence_threshold = 0.35
