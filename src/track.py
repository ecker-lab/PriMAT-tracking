from __future__ import absolute_import

import itertools
import logging
import os
import os.path as osp
from collections import defaultdict

import cv2
import datasets.jde as datasets
import motmetrics as mm
import numpy as np
import torch
from numpy.core._multiarray_umath import ndarray
from opts import opts
from tracker.multitracker import JDESpecializedTracker, JDETracker
from tracking_utils import visualization as vis
from tracking_utils.evaluation import Evaluator
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.utils import mkdir_if_missing


def write_results_dict(
    filename, results_dict, data_type, num_classes=1, use_gc=False, clsID4GC=0
):
    if data_type == "mot":
        save_format = "{frame}, {id}, {x1}, {y1}, {w}, {h}, {score}, {cls_id}\n"
        if use_gc:
            save_format_gc = "{frame}, {id}, {x1}, {y1}, {w}, {h}, {score}, {cls_id}, {gc}, {gc_score}\n"
    elif data_type == "kitti":
        save_format = "{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n"
    else:
        raise ValueError(data_type)

    with open(filename, "w") as f:
        for cls_id in range(num_classes):  # process each object class
            cls_results = results_dict[cls_id]

            for frame_id, tlwhs, track_ids, scores, gcs, gc_scores in cls_results:
                frame_id += 1
                for tlwh, track_id, score, gc, gc_score in itertools.zip_longest(
                    tlwhs, track_ids, scores, gcs, gc_scores, fillvalue=None
                ):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh

                    if use_gc and cls_id == clsID4GC:
                        line = save_format_gc.format(
                            frame=frame_id,
                            id=track_id,
                            x1=x1,
                            y1=y1,
                            w=w,
                            h=h,
                            score=score,  # detection score
                            cls_id=cls_id,
                            gc=gc,
                            gc_score=gc_score,
                        )
                    else:
                        line = save_format.format(
                            frame=frame_id,
                            id=track_id,
                            x1=x1,
                            y1=y1,
                            w=w,
                            h=h,
                            score=score,  # detection score
                            cls_id=cls_id,
                        )

                    f.write(line)
    logger.info("save results to {}".format(filename))


def eval_seq(
    opt,
    dataloader,
    data_type,
    result_filename,
    save_dir=None,
    show_image=True,
    frame_rate=30,
    use_cuda=True,
):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDESpecializedTracker(opt, frame_rate)
    timer = Timer()
    results_dict = defaultdict(list)
    frame_id = 0
    for i, (path, img, img0) in enumerate(dataloader):
        if frame_id % 200 == 0:
            logger.info(
                "Processing frame {} ({:.2f} fps)".format(
                    frame_id, 1.0 / max(1e-5, timer.average_time)
                )
            )

        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)

        online_targets_dict = tracker.update(blob, img0)

        online_tlwhs_dict = defaultdict(list)
        online_ids_dict = defaultdict(list)
        online_scores_dict = defaultdict(list)
        online_gc_scores_dict = defaultdict(list)
        online_gcs_dict = defaultdict(list)

        # process each class separately
        for cls_id in range(opt.num_classes):
            online_targets = online_targets_dict[cls_id]
            for t in online_targets:
                if cls_id == opt.clsID4GC and opt.use_gc:
                    gc_score = t.gc
                    gc_cls = np.argmax(gc_score)
                tlwh = t.tlwh
                tid = t.track_id
                score = t.score
                if tlwh[2] * tlwh[3] > opt.min_box_area:
                    online_tlwhs_dict[cls_id].append(tlwh)
                    online_ids_dict[cls_id].append(tid)
                    online_scores_dict[cls_id].append(score)
                    if cls_id == opt.clsID4GC and opt.use_gc:
                        online_gc_scores_dict[cls_id].append(gc_score)
                        online_gcs_dict[cls_id].append(gc_cls)

        timer.toc()

        # collect results
        for cls_id in range(opt.num_classes):
            results_dict[cls_id].append(
                (
                    frame_id,
                    online_tlwhs_dict[cls_id],
                    online_ids_dict[cls_id],
                    online_scores_dict[cls_id],
                    online_gcs_dict[cls_id],
                    online_gc_scores_dict[cls_id],
                )
            )

        if show_image or save_dir is not None:
            online_im: ndarray = vis.plot_tracking(
                image=img0,
                tlwhs_dict=online_tlwhs_dict,
                obj_ids_dict=online_ids_dict,
                num_classes=opt.num_classes,
                class_names=opt.class_names,
                clsID4GC=opt.clsID4GC,
                gcs_dict=online_gcs_dict,
                gc_cls_names=opt.gc_cls_names,
                gc_scores_dict=online_gc_scores_dict,
                frame_id=frame_id,
                fps=1.0 / timer.average_time,
                show_image=show_image,
                line_thickness=opt.line_thickness,
                id_inline=opt.id_inline,
                debug_info=opt.debug_info,
            )

        if show_image:
            cv2.imshow("online_im", online_im)
        if save_dir is not None:
            cv2.imwrite(
                os.path.join(save_dir, "{:05d}.jpg".format(frame_id)), online_im
            )
        frame_id += 1

    # save results
    write_results_dict(
        result_filename,
        results_dict,
        data_type,
        opt.num_classes,
        use_gc=opt.use_gc,
        clsID4GC=opt.clsID4GC,
    )

    return frame_id, timer.average_time, timer.calls


def main(
    opt,
    data_root="/data/MOT16/train",
    det_root=None,
    seqs=("MOT16-05",),
    exp_name="demo",
    save_images=False,
    save_videos=False,
    show_image=True,
):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, "..", "results", exp_name)
    mkdir_if_missing(result_root)
    data_type = "mot"

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = (
            os.path.join(data_root, "..", "outputs", exp_name, seq)
            if save_images or save_videos
            else None
        )
        logger.info("start seq: {}".format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, "img1"), opt.img_size)
        result_filename = os.path.join(result_root, "{}.txt".format(seq))
        meta_info = open(os.path.join(data_root, seq, "seqinfo.ini")).read()
        frame_rate = int(
            meta_info[meta_info.find("frameRate") + 10 : meta_info.find("\nseqLength")]
        )
        nf, ta, tc = eval_seq(
            opt,
            dataloader,
            data_type,
            result_filename,
            save_dir=output_dir,
            show_image=show_image,
            frame_rate=frame_rate,
        )
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info("Evaluate seq: {}".format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, "{}.mp4".format(seq))
            cmd_str = "ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}".format(
                output_dir, output_video_path
            )
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info(
        "Time elapsed: {:.2f} seconds, FPS: {:.2f}".format(all_time, 1.0 / avg_time)
    )

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(
        summary, os.path.join(result_root, "summary_{}.xlsx".format(exp_name))
    )


# TODO Richard is going to do a selection on the data sets
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = """KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte"""
        # seqs_str = '''TUD-Campus'''
        data_root = os.path.join(opt.data_dir, "MOT15/images/train")
    else:
        seqs_str = """MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13"""
        data_root = os.path.join(opt.data_dir, "MOT16/train")
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(
        opt,
        data_root=data_root,
        seqs=seqs,
        exp_name="MOT17_test_public_dla34",
        show_image=False,
        save_images=False,
        save_videos=False,
    )
