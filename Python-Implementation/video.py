import glob

from numpy import *

from frame import *


class Video(object):
    def __init__(self, path, numb_frames=35):
        self.path = path
        self.numb_frames = numb_frames

        # Getting gloss
        splitted_path = path.split('-')
        self.gloss = splitted_path[len(splitted_path) - 1]
        self.frames = []
        self.read_frames()

    def read_frames(self):
        frames_paths = glob.glob(self.path + '/*')
        frames_paths.reverse()

        for i in range(self.numb_frames):
            frame_path = frames_paths[i]
            frame = Frame(frame_path)
            self.frames.append(frame)

        self.frames.reverse()

    def get_path(self):
        return self.path

    def get_frames(self):
        return self.frames

    def get_frames_matrix(self):
        frame_matrix = []
        for frame in self.frames:
            frame_matrix.append(frame.get_img())
        return frame_matrix

    def get_reduced_frames_matrix(self, numb_groups, frames_matrix=None):
        if frames_matrix == None:
            frames_matrix = self.get_frames_matrix()
        group_size = len(frames_matrix) / (numb_groups * 1.0)

        reduced_frames_matrix = []
        count = 1
        cur_composed_frame = zeros(frames_matrix[0].shape)

        for frame in frames_matrix:
            cur_composed_frame += frame

            if count < group_size:
                count += 1
            else:
                count = 1
                reduced_frames_matrix.append(cur_composed_frame)
                cur_composed_frame = zeros(frames_matrix[0].shape)

        if count > 1:
            reduced_frames_matrix.append(cur_composed_frame)

        return reduced_frames_matrix

    def get_frame_at(self, pos):
        return self.frames[pos]

    def get_frame_matrix_at(self, pos):
        return self.frames[pos].get_img()

    def get_gloss(self):
        return self.gloss

    def set_path(self, path):
        self.path = path

    def set_frames(self, frames):
        self.frames = frames

    def set_frame_at(self, frame):
        self.frames[pos] = frame

    def set_gloss(self, gloss):
        self.gloss = gloss

    def __str__(self):
        return self.gloss + '\n' + self.path + '\n' + str(len(self.frames)) + ' frames'

    def __len__(self):
        return len(self.frames)

    def get_reduced_frames_matrix2(self, numb_groups):
        frames_matrix = array(self.get_frames_matrix())
        frames_matrix.astype('float32')

        # group_size = len(frames_matrix)/numb_groups
        mean = frames_matrix.mean()

        reduced_frames_matrix = []
        cur_composed_frame = zeros(frames_matrix[0].shape)

        count = 1
        for frame in frames_matrix:
            cur_composed_frame += frame

            if cur_composed_frame.sum() >= mean:
                reduced_frames_matrix.append(cur_composed_frame)
                cur_composed_frame = zeros(frames_matrix[0].shape)
                count = 1
            else:
                count += 1

        if count > 1:
            reduced_frames_matrix[-1] += cur_composed_frame

        if len(reduced_frames_matrix) < numb_groups:
            reduced_frames_matrix += [reduced_frames_matrix[-1]] * (numb_groups - len(reduced_frames_matrix))
        else:
            reduced_frames_matrix = self.get_reduced_frames_matrix(numb_groups, reduced_frames_matrix)

        return reduced_frames_matrix
