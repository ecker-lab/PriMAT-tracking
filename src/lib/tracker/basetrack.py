from collections import OrderedDict, defaultdict

import numpy as np


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count_dict = defaultdict(int)

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id(cls_id):
        BaseTrack._count_dict[cls_id] += 1
        return BaseTrack._count_dict[cls_id]

    # @even: reset track id
    @staticmethod
    def init_count(num_classes):
        """
        Initiate _count for all object classes
        :param num_classes:
        """
        for cls_id in range(num_classes):
            BaseTrack._count_dict[cls_id] = 0

    @staticmethod
    def reset_track_count(cls_id):
        BaseTrack._count_dict[cls_id] = 0

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed
