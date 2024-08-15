check_point_inside_box <- function(box, point, scale_boxes = 0) {
  box_x <- box[1]
  box_y <- box[2]
  width <- box[3]
  height <- box[4]
  
  x <- point[1]
  y <- point[2]
  
  if ((x >= box_x - (width * (1+scale_boxes))/2) && (x <= box_x + (width * (1 + scale_boxes))/2) &&
      (y >= box_y - (height * (1+scale_boxes))/2 && y <= box_y + (height * (1 + scale_boxes))/2)) {
    return(TRUE)
  } else {
    return(FALSE)
  }
}

calculate_iou <- function(gt_box, pred_box) {
  # Extracting the coordinates and dimensions of gt_box
  x1 <- gt_box[1]
  y1 <- gt_box[2]
  width1 <- gt_box[3]
  height1 <- gt_box[4]
  
  # Extracting the coordinates and dimensions of pred_box
  x2 <- pred_box[1] 
  y2 <- pred_box[2]
  width2 <- pred_box[3]
  height2 <- pred_box[4]
  
  # Calculating the coordinates of the overlapping region
  x_overlap <- max(0, min(x1 + width1/2, x2 + width2/2) - max(x1 - width1/2, x2- width2/2))
  y_overlap <- max(0, min(y1 + height1/2, y2 + height2/2) - max(y1 - height1/2, y2- height2/2))
  
  # Calculating the area of overlap
  overlap_area <- x_overlap * y_overlap
  
  area_gt_box <- width1 * height1
  
  # Calculating the area of pred_box
  area_pred_box <- width2 * height2
  
  # Calculating the union area
  union_area <- area_gt_box + area_pred_box - overlap_area
  
  # Calculating the IoU
  iou <- overlap_area / union_area
  
  return(iou)
}
