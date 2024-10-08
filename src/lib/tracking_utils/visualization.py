import numpy as np
import cv2


def tlwhs_to_tlbrs(tlwhs):
    tlbrs = np.copy(tlwhs)
    if len(tlbrs) == 0:
        return tlbrs
    tlbrs[:, 2] += tlwhs[:, 0]
    tlbrs[:, 3] += tlwhs[:, 1]
    return tlbrs


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def resize_image(image, max_size=800):
    if max(image.shape[:2]) > max_size:
        scale = float(max_size) / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale)
    return image


# TODO look into how to activate and use this
def plot_trajectory(image, tlwhs, track_ids):
    image = image.copy()
    for one_tlwhs, track_id in zip(tlwhs, track_ids):
        color = get_color(int(track_id))
        for tlwh in one_tlwhs:
            x1, y1, w, h = tuple(map(int, tlwh))
            cv2.circle(image, (int(x1 + 0.5 * w), int(y1 + h)), 2, color, thickness=2)

    return image


def plot_detection(image, dets_dict, num_classes, frame_id, fps=0.0):
    img = np.ascontiguousarray(np.copy(image))
    # im_h, im_w = img.shape[:2]

    text_scale = max(1.0, image.shape[1] / 1200.0)  # 1600.
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 600.0))

    for cls_id in range(num_classes):
        # plot each object class
        cls_dets = dets_dict[cls_id]

        cv2.putText(
            img,
            "frame: %d fps: %.2f" % (frame_id, fps),
            (0, int(15 * text_scale)),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            (0, 0, 255),
            thickness=2,
        )

        # plot each object of the object class
        for obj_i, obj in enumerate(cls_dets):
            # left, top, right, down, score, cls_id
            x1, y1, x2, y2, score, cls_id = obj
            cls_name = opt.class_names[int(cls_id)]
            box_int = tuple(map(int, (x1, y1, x2, y2)))
            cls_color = get_color(abs(cls_id))

            # draw bbox for each object
            cv2.rectangle(
                img,
                box_int[0:2],
                box_int[2:4],
                color=cls_color,
                thickness=line_thickness,
            )

            # draw class name
            cv2.putText(
                img,
                cls_name,
                (box_int[0], box_int[1]),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale,
                [0, 255, 255],
                thickness=text_thickness,
            )

    return img


def plot_tracking(
    image,
    tlwhs_dict,
    obj_ids_dict,
    num_classes,
    class_names,
    clsID4GC=0,
    gcs_dict=None,
    gc_cls_names=None,
    gc_scores_dict=None,
    scores=None,
    frame_id=0,
    fps=0.0,
    show_image=False,
    line_thickness=1,
    id_inline=False,
    debug_info=False,
):
    
    

    img = np.ascontiguousarray(np.copy(image))
    im_h, im_w = img.shape[:2]

    text_scale = max(1.0, image.shape[1] / 1200.0)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.0 * line_thickness))

    if debug_info:
        collected_scores = []

    for cls_id in range(num_classes):
        cls_tlwhs = tlwhs_dict[cls_id]
        obj_ids = obj_ids_dict[cls_id]

        if show_image:
            cv2.putText(
                img,
                f"frame: {frame_id} fps: {fps:.2f}",
                (0, int(15 * text_scale)),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale,
                (0, 0, 255),
                thickness=2,
            )
        
        

        for i, tlwh_i in enumerate(cls_tlwhs):
            x1, y1, w, h = tlwh_i
            int_box = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            obj_id = int(obj_ids[i])
            id_text = "<{}>".format(int(obj_id))

            color = get_color(abs(obj_id))

            # draw bbox
            cv2.rectangle(
                img=img,
                pt1=int_box[0:2],  # (x1, y1)
                pt2=int_box[2:4],  # (x2, y2)
                color=color,
                thickness=line_thickness,
            )

            # draw class name, index and applying gc-labels
            box_text = class_names[cls_id]
            if id_inline:
                box_text += f" {id_text}"
            if len(gcs_dict[cls_id]) > 0:
                if debug_info:
                    collected_scores.append(box_text)
                a = gc_scores_dict[clsID4GC][i][gcs_dict[cls_id][i]]
                a = str(round(a, 2)) #if a>0.1 else ""
                #a=1
                box_text += f" : {gc_cls_names[gcs_dict[cls_id][i]]}{str(a)}"

            cv2.putText(
                img,
                box_text,
                (int(x1), int(y1)),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale,
                (0, 255, 255),
                thickness=text_thickness,
            )

            # Put ID above of Class name
            if not id_inline:
                if debug_info and len(gcs_dict[cls_id]) > 0:
                    collected_scores[-1] += f" {id_text}"
                txt_w, txt_h = cv2.getTextSize(
                    box_text,
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=text_scale,
                    thickness=text_thickness,
                )

                cv2.putText(
                    img,
                    id_text,
                    (int(x1), int(y1) - (txt_h + 8)),
                    cv2.FONT_HERSHEY_PLAIN,
                    text_scale,
                    (0, 255, 255),
                    thickness=text_thickness,
                )

    # print GC scores in top right corner
    if debug_info:
        uv_top_left = np.array([im_w - 170, 20], dtype=float)
        for (row, scores) in zip(collected_scores, gc_scores_dict[clsID4GC]):
            order = np.argsort(scores)[::-1]
            text = "GC Label\n{}\n{}\n{}\n{}\n{}".format(
                gc_cls_names[order[0]],
                gc_cls_names[order[1]],
                gc_cls_names[order[2]],
                gc_cls_names[order[3]],
                gc_cls_names[order[4]],
            )
            text2 = "Score\n{:.2f}\n{:.2f}\n{:.2f}\n{:.2f}\n{:.2f}".format(
                scores[order[0]],
                scores[order[1]],
                scores[order[2]],
                scores[order[3]],
                scores[order[4]],
            )

            assert uv_top_left.shape == (2,)

            (w, h), _ = cv2.getTextSize(
                text=row,
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=text_scale,
                thickness=text_thickness,
            )
            org = tuple((uv_top_left - [w + 20, -h]).astype(int))
            cv2.putText(img, row, org, cv2.FONT_HERSHEY_PLAIN, 1, text_thickness)

            for i, (line, line2) in enumerate(zip(text.split("\n"), text2.split("\n"))):
                (w, h), _ = cv2.getTextSize(
                    text=line,
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=text_scale,
                    thickness=text_thickness,
                )
                uv_bottom_left_i = uv_top_left + [0, h]
                org1 = tuple((uv_bottom_left_i + [0, 0]).astype(int))
                org2 = tuple((uv_bottom_left_i + [100, 0]).astype(int))
                cv2.putText(img, line, org1, cv2.FONT_HERSHEY_PLAIN, 1, text_thickness)
                cv2.putText(img, line2, org2, cv2.FONT_HERSHEY_PLAIN, 1, text_thickness)
                line_spacing = 1.5
                uv_top_left += [0, h * line_spacing]

            uv_top_left += [0, h * line_spacing * 1.2]

    return img
