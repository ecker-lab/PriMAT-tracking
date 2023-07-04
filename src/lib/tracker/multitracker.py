from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn.functional as F
from models.decode import mot_decode
from models.model import create_model, load_model
from models.utils import _tranpose_and_gather_feat
from tracker import matching
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.log import logger
from tracking_utils.utils import *

from models import *

from .basetrack import BaseTrack, TrackState


class Track(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, cls_id, buff_size=30, gc=None):
        """
        :param tlwh:
        :param score:
        :param temp_feat:
        :param num_classes:
        :param cls_id:
        :param buff_size:
        :param gc:
        """
        
        # object class id
        self.cls_id = cls_id

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.track_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buff_size)
        self.alpha = 0.9

        self.gc = gc

    def update_features(self, feat):
        # L2 normalizing
        feat /= np.linalg.norm(feat)

        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1.0 - self.alpha) * feat

        self.features.append(feat)

        # L2 normalizing
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(tracks):
        if len(tracks) > 0:
            multi_mean = np.asarray([track.mean.copy() for track in tracks])
            multi_covariance = np.asarray([track.covariance for track in tracks])

            for i, st in enumerate(tracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = Track.shared_kalman.multi_predict(multi_mean, multi_covariance)

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                tracks[i].mean = mean
                tracks[i].covariance = cov

    def reset_track_id(self):
        self.reset_track_count(self.cls_id)

    def activate(self, kalman_filter, frame_id):
        """Start a new track"""
        self.kalman_filter = kalman_filter  # assign a filter to each track?

        # update track id for the object class
        self.track_id = self.next_id(self.cls_id)

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.track_len = 0
        self.state = TrackState.Tracked  # set flag 'tracked'

        self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        # kalman update
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_track.tlwh))

        # feature vector update
        self.update_features(new_track.curr_feat)

        self.track_len = 0
        self.frame_id = frame_id

        self.state = TrackState.Tracked  # set flag 'tracked'
        self.is_activated = True

        if new_id:  # update track id for the object class
            self.track_id = self.next_id(self.cls_id)

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: Track
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.track_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked  # set flag 'tracked'
        self.is_activated = True  # set flag 'activated'

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

        self.gc = new_track.gc

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()

        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_({}-{})_({}-{})'.format(self.cls_id, self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        if opt.use_gc:
            self.model = create_model(opt.arch, opt.heads, opt.head_conv, num_gc_cls=opt.num_gc_cls, clsID4GC=opt.clsID4GC)
        else:
            self.model = create_model(opt.arch, opt.heads, opt.head_conv, num_gc_cls=None, clsID4GC=None)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.tracked_tracks_dict = defaultdict(list)
        self.lost_tracks_dict = defaultdict(list)
        self.last_seen_tracks_dict = defaultdict(list)
        self.removed_tracks_dict = defaultdict(list)

        self.frame_id = 0
        self.max_frames_between_det = int(frame_rate * self.opt.track_buffer)
        self.max_per_image = opt.K
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.det_thres = opt.det_thres
        self.kalman_filter = KalmanFilter()
        self.proportion_iou = opt.proportion_iou
        self.new_overlap_thres = opt.new_overlap_thres
        self.buffered_iou = opt.buffered_iou

    def reset(self):
        """
        :return:
        """
        # Reset tracks dict
        self.tracked_tracks_dict = defaultdict(list)
        self.lost_tracks_dict = defaultdict(list)
        self.last_seen_tracks_dict = defaultdict(list)
        self.removed_tracks_dict = defaultdict(list)

        # Reset frame id
        self.frame_id = 0

        # Reset kalman filter to stabilize tracking
        self.kalman_filter = KalmanFilter()


    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def update(self, im_blob, img0):
        self.frame_id += 1

        # ----- reset the track ids for all object classes in the first frame
        if self.frame_id == 1:
            Track.init_count(self.opt.num_classes)

        # record tracking results, key: class_id
        activated_tracks_dict = defaultdict(list)
        lost_tracks_dict = defaultdict(list)
        removed_tracks_dict = defaultdict(list)
        output_tracks_dict = defaultdict(list)

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        h_out = inp_height // self.opt.down_ratio
        w_out = inp_width // self.opt.down_ratio

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            # L2 normalize the reid feature vector
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.opt.reg_offset else None
            

            dets, inds, cls_inds_mask = mot_decode(heatmap=hm,
                                                   wh=wh,
                                                   reg=reg,
                                                   num_classes=self.opt.num_classes,
                                                   cat_spec_wh=self.opt.cat_spec_wh,
                                                   K=self.opt.K)
            
            # ----- get ReID feature vector by object class
            cls_id_feats = []  # topK feature vectors of each object class
            for cls_id in range(self.opt.num_classes):
                cls_inds = inds[:, cls_inds_mask[cls_id]]

                # gather feats for each object class
                cls_id_feature = _tranpose_and_gather_feat(id_feature, cls_inds)
                cls_id_feature = cls_id_feature.squeeze(0)  # n × FeatDim
                cls_id_feature = cls_id_feature.cpu().numpy()
                cls_id_feats.append(cls_id_feature)

      
        if self.opt.use_gc:
            gc_inds = inds[:, cls_inds_mask[self.opt.clsID4GC]]
            output['gc_pred'] = self.model.gc_lin(output['gc'], gc_inds)
            output['gc_pred'] = output['gc_pred'].squeeze().cpu().numpy()
 

        # translate and scale
        dets = map2orig(dets, h_out, w_out, height, width, self.opt.num_classes)

        # from here on everything is detached and numpy and not torch.tensor!


        # ----- parse each object class
        for cls_id in range(self.opt.num_classes):
            cls_dets = dets[cls_id]

            # low threshold (0.01) to allow many potential detections
            remain_inds = cls_dets[:, 4] > self.opt.conf_thres
            cls_dets = cls_dets[remain_inds]
            # Remove detections where the center is closer than X=10 pixels to the border
            # Reason for this is that there are usually many low-confidence detections close
            # to the borders which prevent bounding boxes from disappearing
            no_border_inds = ((cls_dets[:, 0] + cls_dets[:, 2]) / 2 > 10) & ((cls_dets[:, 0] + cls_dets[:, 2]) / 2 < (width - 10)) & ((cls_dets[:, 1] + cls_dets[:, 3]) / 2 > 10) & ((cls_dets[:, 1] + cls_dets[:, 3]) / 2 < (height - 10))
            cls_dets = cls_dets[no_border_inds]
            cls_id_feature = cls_id_feats[cls_id][remain_inds]
            cls_id_feature = cls_id_feature[no_border_inds]
            
            
            if self.opt.use_gc and cls_id == self.opt.clsID4GC:
                output['gc_pred'] = output['gc_pred'].reshape(-1, self.opt.num_gc_cls)[remain_inds.squeeze()].reshape(-1, self.opt.num_gc_cls)
                # FIXME should we use this option?
                output['gc_pred'] = output['gc_pred'][no_border_inds]


            if len(cls_dets) > 0:
                '''Detections'''
                if self.opt.use_gc and cls_id == self.opt.clsID4GC:
                    cls_detects = [Track(Track.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], feat, cls_id, self.opt.track_buffer, gc)
                    for (tlbrs, feat, gc) in zip(cls_dets[:, :5], cls_id_feature, output['gc_pred'])]
                else:
                    cls_detects = [Track(Track.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], feat, cls_id, self.opt.track_buffer)
                        for (tlbrs, feat) in zip(cls_dets[:, :5], cls_id_feature)]
            else:
                cls_detects = []

            ''' Add newly detected tracklets to tracked_stracks'''
            unconfirmed_dict = defaultdict(list)
            tracked_tracks_dict = defaultdict(list)
            for track in self.tracked_tracks_dict[cls_id]:
                tracked_tracks_dict[cls_id].append(track)

            ''' Step 2: Calculate embedding distance and IoU distance'''
            track_pool_dict = defaultdict(list)
            track_pool_last_seen_dict = defaultdict(list)
            track_pool_dict[cls_id] = joint_stracks(tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])
            track_pool_last_seen_dict[cls_id] = joint_stracks(tracked_tracks_dict[cls_id], self.last_seen_tracks_dict[cls_id])
            # Predict the current location with KF
            Track.multi_predict(track_pool_dict[cls_id])


            emb_dists = matching.embedding_distance(track_pool_dict[cls_id], cls_detects)
            iou_dists = matching.buffered_iou_distance(track_pool_dict[cls_id], cls_detects)

            dists = self.proportion_iou * iou_dists + (1 - self.proportion_iou) * emb_dists

            
            if iou_dists.size == 0:
                min_dist = np.array([])
            else:
                min_dist = np.amin(iou_dists, axis = 0)

            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.opt.sim_thres)

            for itracked, idet in matches:
                track = track_pool_dict[cls_id][itracked]
                det = cls_detects[idet]
                track.update(cls_detects[idet], self.frame_id)
                activated_tracks_dict[cls_id].append(track)

            """ Step 3: What happens to non-matches: Will be compared to last position of tracks"""
            cls_detects = [cls_detects[i] for i in u_detection]
            r_tracked_tracks = [track_pool_last_seen_dict[cls_id][i] for i in u_track]
            dists = matching.iou_distance(r_tracked_tracks, cls_detects)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.9) # more relaxed threshold

            for i_tracked, i_det in matches:
                track = r_tracked_tracks[i_tracked]
                det = cls_detects[i_det]
                track.update(det, self.frame_id)
                activated_tracks_dict[cls_id].append(track)

            for it in u_track:
                track = r_tracked_tracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_tracks_dict[cls_id].append(track)


            """ Step 4: Init new tracks"""
            # detections - if they have a high score and low overlap: add a new track
            for inew in u_detection:
                track = cls_detects[inew]
                if (track.score > self.det_thres): # with 0.7 relatively high to only allow real new detections
                    if (len(min_dist) > 0):
                        if min_dist[inew] > self.opt.new_overlap_thres:
                            track.activate(self.kalman_filter, self.frame_id)
                            activated_tracks_dict[cls_id].append(track)
                    else:
                        track.activate(self.kalman_filter, self.frame_id)
                        activated_tracks_dict[cls_id].append(track)


            """ Step 5: Remove lost tracks after some time"""
            for track in self.lost_tracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_frames_between_det:
                    track.mark_removed()
                    removed_tracks_dict[cls_id].append(track)
                    
            for track in self.last_seen_tracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_frames_between_det:
                    track.mark_removed()


            self.tracked_tracks_dict[cls_id] = [t for t in self.tracked_tracks_dict[cls_id] if
                                                t.state == TrackState.Tracked]
            self.tracked_tracks_dict[cls_id] = joint_stracks(self.tracked_tracks_dict[cls_id],
                                                           activated_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_stracks(self.lost_tracks_dict[cls_id],
                                                       self.tracked_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_stracks(self.lost_tracks_dict[cls_id],
                                                       self.removed_tracks_dict[cls_id])
            
            self.last_seen_tracks_dict[cls_id] = sub_stracks(self.last_seen_tracks_dict[cls_id],
                                                       self.tracked_tracks_dict[cls_id])
            self.last_seen_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
            self.last_seen_tracks_dict[cls_id] = sub_stracks(self.last_seen_tracks_dict[cls_id],
                                                       self.removed_tracks_dict[cls_id])
            
            self.removed_tracks_dict[cls_id].extend(removed_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = remove_duplicate_tracks(
                self.tracked_tracks_dict[cls_id],
                self.lost_tracks_dict[cls_id])
            
            self.tracked_tracks_dict[cls_id], self.last_seen_tracks_dict[cls_id] = remove_duplicate_tracks(
                self.tracked_tracks_dict[cls_id],
                self.last_seen_tracks_dict[cls_id])

            # get scores of lost tracks
            output_tracks_dict[cls_id] = [track for track in self.tracked_tracks_dict[cls_id] if track.is_activated]

            logger.debug('===========Frame {}=========='.format(self.frame_id))
            logger.debug('Activated: {}'.format(
                [track.track_id for track in activated_tracks_dict[cls_id]]))
            logger.debug('Lost: {}'.format(
                [track.track_id for track in lost_tracks_dict[cls_id]]))
            logger.debug('Removed: {}'.format(
                [track.track_id for track in removed_tracks_dict[cls_id]]))

        # FIXME look up if I can include output['gc_pred'] in output_tracks_dict
        # TODO directly append gc to Track, then it is also part of output_tracks_dict...
        # if self.opt.use_gc:
        #     return output_tracks_dict, output['gc_pred']
        # else:
        return output_tracks_dict


class JDESpecializedTracker(JDETracker):
    
    def update(self, im_blob, img0):
        self.frame_id += 1

        # ----- reset the track ids for all object classes in the first frame
        if self.frame_id == 1:
            Track.init_count(self.opt.num_classes)

        # record tracking results, key: class_id
        activated_tracks_dict = defaultdict(list)
        refined_tracks_dict = defaultdict(list)
        lost_tracks_dict = defaultdict(list)
        removed_tracks_dict = defaultdict(list)
        output_tracks_dict = defaultdict(list)

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        h_out = inp_height // self.opt.down_ratio
        w_out = inp_width // self.opt.down_ratio

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            # L2 normalize the reid feature vector
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.opt.reg_offset else None
            
            dets, inds, cls_inds_mask = mot_decode(heatmap=hm,
                                                   wh=wh,
                                                   reg=reg,
                                                   num_classes=self.opt.num_classes,
                                                   cat_spec_wh=self.opt.cat_spec_wh,
                                                   K=self.opt.K)
            

            # ----- get ReID feature vector by object class
            cls_id_feats = []  # topK feature vectors of each object class
            for cls_id in range(self.opt.num_classes):
                cls_inds = inds[:, cls_inds_mask[cls_id]]

                # gather feats for each object class
                cls_id_feature = _tranpose_and_gather_feat(id_feature, cls_inds)
                cls_id_feature = cls_id_feature.squeeze(0)  # n × FeatDim
                cls_id_feature = cls_id_feature.cpu().numpy()
                cls_id_feats.append(cls_id_feature)

        if self.opt.use_gc:
            mnk_inds = inds[:, cls_inds_mask[self.opt.clsID4GC]]
            output['gc_pred'] = self.model.gc_lin(output['gc'], mnk_inds)
            output['gc_pred'] = output['gc_pred'].squeeze().cpu().numpy()


        # translate and scale
        dets = map2orig(dets, h_out, w_out, height, width, self.opt.num_classes)
        

        # ----- parse each object class
        for cls_id in range(self.opt.num_classes):
            cls_dets = dets[cls_id]

            # low threshold (0.01) to allow many potential detections
            remain_inds = cls_dets[:, 4] > self.opt.conf_thres
            cls_dets = cls_dets[remain_inds]
            
            no_border_inds = ((cls_dets[:, 0] + cls_dets[:, 2]) / 2 > 0) & ((cls_dets[:, 0] + cls_dets[:, 2]) / 2 < width) & ((cls_dets[:, 1] + cls_dets[:, 3]) / 2 > 0) & ((cls_dets[:, 1] + cls_dets[:, 3]) / 2 < height)
            cls_dets = cls_dets[no_border_inds]
            cls_id_feature = cls_id_feats[cls_id][remain_inds]
            cls_id_feature = cls_id_feature[no_border_inds]

            if self.opt.use_gc and cls_id == self.opt.clsID4GC:
                output['gc_pred'] = output['gc_pred'].reshape(-1, self.opt.num_gc_cls)[np.bitwise_and(remain_inds.squeeze(), no_border_inds.squeeze())].reshape(-1, self.opt.num_gc_cls)

            if len(cls_dets) > 0:
                '''Detections'''
                if self.opt.use_gc and cls_id == self.opt.clsID4GC:
                    cls_detects = [Track(Track.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], feat, cls_id, self.opt.track_buffer, gc)
                    for (tlbrs, feat, gc) in zip(cls_dets[:, :5], cls_id_feature, output['gc_pred'])]
                else:
                    cls_detects = [Track(Track.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], feat, cls_id, self.opt.track_buffer)
                        for (tlbrs, feat) in zip(cls_dets[:, :5], cls_id_feature)]
            else:
                cls_detects = []

            ''' Add newly detected tracklets to tracked_stracks'''
            unconfirmed_dict = defaultdict(list)
            tracked_tracks_dict = defaultdict(list)
            for track in self.tracked_tracks_dict[cls_id]:
                # seperation in unconfirmed and not is not ther in richards code
                if not track.is_activated:
                    unconfirmed_dict[cls_id].append(track)
                else:
                    tracked_tracks_dict[cls_id].append(track)

            ''' Step 2: Calculate embedding distance and IoU distance'''
            track_pool_dict = defaultdict(list)
            track_pool_dict[cls_id] = joint_stracks(tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])
            # Predict the current location with KF
            Track.multi_predict(track_pool_dict[cls_id])


            emb_dists = matching.embedding_distance(track_pool_dict[cls_id], cls_detects)
            iou_dists = matching.buffered_iou_distance(track_pool_dict[cls_id], cls_detects, factor = self.buffered_iou)


            #dists = np.multiply(emb_dists, iou_dists)
            dists = self.proportion_iou * iou_dists + (1 - self.proportion_iou) * emb_dists
            
            if iou_dists.size == 0:
                min_dist = np.array([])
            else:
                min_dist = np.amin(iou_dists, axis = 0)

            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.opt.sim_thres)

            for itracked, idet in matches:
                track = track_pool_dict[cls_id][itracked]
                det = cls_detects[idet]
                if track.state == TrackState.Tracked:
                    track.update(cls_detects[idet], self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)

         
            for it in u_track:
                track = track_pool_dict[cls_id][it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_tracks_dict[cls_id].append(track)

          


            """ Step 4: Init new tracks"""
            # detections - if they have a high score and low overlap: add a new track
            for inew in u_detection:
                track = cls_detects[inew]
                if (track.score > self.det_thres): # with 0.7 relatively high to only allow real new detections
                    if (len(min_dist) > 0):
                        if min_dist[inew] > self.new_overlap_thres:
                            track.activate(self.kalman_filter, self.frame_id)
                            activated_tracks_dict[cls_id].append(track)
                    else:
                        track.activate(self.kalman_filter, self.frame_id)
                        activated_tracks_dict[cls_id].append(track)



            """ Step 5: Remove lost tracks after some time"""
            for track in self.lost_tracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_frames_between_det:
                    track.mark_removed()
                    removed_tracks_dict[cls_id].append(track)


            self.tracked_tracks_dict[cls_id] = [t for t in self.tracked_tracks_dict[cls_id] if
                                                t.state == TrackState.Tracked]
            self.tracked_tracks_dict[cls_id] = joint_stracks(self.tracked_tracks_dict[cls_id],
                                                           activated_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id] = joint_stracks(self.tracked_tracks_dict[cls_id],
                                                           refined_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_stracks(self.lost_tracks_dict[cls_id],
                                                       self.tracked_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_stracks(self.lost_tracks_dict[cls_id],
                                                       self.removed_tracks_dict[cls_id])
            self.removed_tracks_dict[cls_id].extend(removed_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = remove_duplicate_tracks(
                self.tracked_tracks_dict[cls_id],
                self.lost_tracks_dict[cls_id])

            # get scores of lost tracks
            output_tracks_dict[cls_id] = [track for track in self.tracked_tracks_dict[cls_id] if track.is_activated]

            logger.debug('===========Frame {}=========='.format(self.frame_id))
            logger.debug('Activated: {}'.format(
                [track.track_id for track in activated_tracks_dict[cls_id]]))
            logger.debug('Refind: {}'.format(
                [track.track_id for track in refined_tracks_dict[cls_id]]))
            logger.debug('Lost: {}'.format(
                [track.track_id for track in lost_tracks_dict[cls_id]]))
            logger.debug('Removed: {}'.format(
                [track.track_id for track in removed_tracks_dict[cls_id]]))

        # if self.opt.use_gc:
        #     return output_tracks_dict, output['gc_pred']
        # else:
        return output_tracks_dict


def map2orig(dets, h_out, w_out, h_orig, w_orig, num_classes):
    """
    :param dets:
    :param h_out:
    :param w_out:
    :param h_orig:
    :param w_orig:
    :param num_classes:
    :return: dict of detections(key: cls_id)
    """

    def get_padding():
        """
        :return: pad_1, pad_2, pad_type('pad_x' or 'pad_y'), new_shape(w, h)
        """
        ratio_x = float(w_out) / w_orig
        ratio_y = float(h_out) / h_orig
        ratio = min(ratio_x, ratio_y)
        new_shape = (round(w_orig * ratio), round(h_orig * ratio))  # new_w, new_h

        pad_x = (w_out - new_shape[0]) * 0.5  # width padding
        pad_y = (h_out - new_shape[1]) * 0.5  # height padding
        top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
        left, right = round(pad_x - 0.1), round(pad_x + 0.1)
        if ratio == ratio_x:  # pad_y
            return top, bottom, 'pad_y', new_shape
        else:  # pad_x
            return left, right, 'pad_x', new_shape

    pad_1, pad_2, pad_type, new_shape = get_padding()

    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])  # default: 1×128×6
    dets = dets[0]  # 128×6

    dets_dict = {}

    if pad_type == 'pad_x':
        dets[:, 0] = (dets[:, 0] - pad_1) / new_shape[0] * w_orig  # x1
        dets[:, 2] = (dets[:, 2] - pad_1) / new_shape[0] * w_orig  # x2
        dets[:, 1] = dets[:, 1] / h_out * h_orig  # y1
        dets[:, 3] = dets[:, 3] / h_out * h_orig  # y2
    else:  # 'pad_y'
        dets[:, 0] = dets[:, 0] / w_out * w_orig  # x1
        dets[:, 2] = dets[:, 2] / w_out * w_orig  # x2
        dets[:, 1] = (dets[:, 1] - pad_1) / new_shape[1] * h_orig  # y1
        dets[:, 3] = (dets[:, 3] - pad_1) / new_shape[1] * h_orig  # y2

    classes = dets[:, -1]
    for cls_id in range(num_classes):
        inds = (classes == cls_id)
        dets_dict[cls_id] = dets[inds, :]

    return dets_dict


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_tracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
