# ================================== #
# Cut Detector
# ================================== #
frame_size = (128, 128)
n_frame_added = 6
batch_size = 32
cut_detector = './cut_detector.h5'
num_iteration_in_one_epoch = 1e3

# ================================== #
# Grad Detector
# ================================== #
grad_frame_size = (128, 128)
grad_n_frames_per_sample = 60
grad_batch_size = 256
grad_detector = './grad_detector.h5'
grad_num_iteration_in_one_epoch = 1e2
