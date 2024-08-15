library(dplyr)
library(ggplot2)
library(tidyr)
library(ggchicklet)

source("helpers.R")


path_to_dlc <- "for_Z02_LemurKeypointModels/full_labels/dlcrnet_ms5/videos_iou0.6_pcutoff0.0/"

## Load data

## gt is in format cx, cy, w, h
## pred is in format t, l, w, h

vids <- seq(8, 19)


### ground truth
read_txt_files <- function(filename) {
  temp <- read.table(paste0(path_to_gt, filename))
  temp$V7 <- i
  i <<- i + 1
  return(temp)
}

for(vid in vids) {
  path_to_gt <- paste0("Eval", vid, "/labels_with_ids/")
  
  all_text <- list.files(path_to_gt)
  
  i <- 1
  
  
  assign(paste0("gt", vid), lapply(all_text, read_txt_files) %>%
    do.call(rbind, .) %>%
    rename(zero = V1, id = V2, x = V3, y = V4, w = V5, h = V6, frame = V7))
  
}


### predictions

for(vid in vids) {
  assign(paste0("pred", vid), read.table(paste0("Predictions/0.01_0.4_0.8_0.5/Eval", vid, ".txt"), sep = ",") %>%
           rename(frame = V1, id = V2, x = V3, y = V4, w = V5, h = V6, conf = V7, class = V8) %>%
           mutate(x = x / 1920, y = y / 1080, w = w / 1920, h = h / 1080))
  
}


### DLC keypoints

for(vid in vids) {

  
  collapse_colnames <- function(x) {
    paste(x, collapse = "_")
  }
  
  colnames <- read.csv(paste0(path_to_dlc, "Eval8DLC_dlcrnetms5_Lemur_fieldVideosMar28shuffle1_75000_el.csv")) %>%
    slice(1:3)
  
  colnames_vec <- apply(colnames, 2, collapse_colnames)
  
  file_path <- paste0(path_to_dlc, "Eval", vid, "DLC_dlcrnetms5_Lemur_fieldVideosMar28shuffle1_75000_el.csv")
  
  if (!file.exists(file_path)) {
    print(paste0("File does not exist: ", file_path))
    # Return an empty dataframe with specified column names
    assign(paste0("dlc", vid), data.frame(frame = integer(), 
                      individual = character(), 
                      bodypart = character(), 
                      x = numeric(), 
                      y = numeric(), 
                      likelihood = numeric(), 
                      stringsAsFactors = FALSE))
  } else {
    
    # Attempt to read the CSV file
    dlc <- read.csv(paste0(path_to_dlc, "Eval", vid, "DLC_dlcrnetms5_Lemur_fieldVideosMar28shuffle1_75000_el.csv"),
                    skip = 3)
    
    names(dlc) <- colnames_vec
    
    
    
    dlc <- dlc %>% rename(frame = individuals_bodyparts_coords)
    
    
    dlc_long <- dlc %>%
      pivot_longer(
        cols = -frame,
        names_to = c("individual","bodypart", "variable"),
        names_pattern = "(ind.)_(.*)_(.*)", #has to be "monkey" for the spine version, "ind" for the full version 
        values_to = "value"
      )
    
    assign(paste0("dlc", vid), dlc_long %>%
             pivot_wider(names_from = variable) %>%
             mutate(frame = frame + 1) %>%
             mutate(x = x / 1920, y = y / 1080))
    
    
  }
  
  
  
  
}

  
## Visualize

vid <- 10

gt <- eval(parse(text = paste0("gt", vid))) %>% filter(id < 10)
pred <- eval(parse(text = paste0("pred", vid))) %>% filter(class == 0)
dlc <- eval(parse(text = paste0("dlc", vid))) %>% 
  filter(!(bodypart %in% c("M_tail", "E_tail")), !is.na(x))

### Visualize one frame

sel_frame <- 320

gt %>%
  filter(frame == sel_frame) %>%
  ggplot() + 
  geom_rect(aes(xmin = x - w/2, xmax = x + w/2, ymin = 1 - (y - h/2), ymax = 1 - (y + h/2),
                col = id > 9), fill = NA) +
  geom_rect(data = pred %>% filter(frame == sel_frame), 
            aes(xmin = x, xmax = x + w, ymin = 1 - y, ymax = 1 - (y + h),
                col = as.character(class)),
            fill = NA) +
  geom_point(data = dlc %>% filter(frame == sel_frame), 
             aes(x = x, y = 1- y, col = individual)) +
  coord_cartesian(xlim = c(0,1), ylim = c(0,1)) +
  scale_y_continuous(limits = c(0,1)) +
  #scale_color_discrete(labels = c("pred_lemur", "gt_lemur", "DLC_Ind1", "DLC_Ind2")) +
  theme_light() +
  theme(axis.text = element_blank(),
        axis.ticks = element_blank())



### Visualize many frames (skip every xth frame)
num_frames <- 4
skip <- 60


gt %>%
  filter(frame %% skip == 0, frame <= skip * num_frames) %>%
  ggplot() + 
  geom_rect(aes(xmin = x - w/2, xmax = x + w/2, ymin = 1 - (y - h/2), ymax = 1 - (y + h/2),
                col = id > 9), fill = NA) +
  geom_rect(data = pred %>% filter(frame %% skip == 0, frame <= skip * num_frames), 
            aes(xmin = x, xmax = x + w, ymin = 1 - y, ymax = 1 - (y + h),
                col = as.character(class)),
            fill = NA, linetype = "dashed") +
  geom_point(data = dlc %>% filter(frame %% skip == 0, frame <= skip * num_frames), aes(x = x, y = 1- y, col = "z")) +
  coord_cartesian(xlim = c(0,1), ylim = c(0,1)) +
  scale_y_continuous(limits = c(0,1)) +
  scale_color_manual(labels = c("our model", "ground truth", "DLC"),
                       values = c("grey20", "green3", "grey")) +
  facet_wrap(~frame, nrow = 2) +
  theme_light() +
  labs(color = "") +
  theme(axis.text = element_blank(),
        axis.ticks = element_blank(),
        axis.title = element_blank(),
        legend.position = "bottom")

ggsave("boxes_keypoints.png", width = 7, height = 5)


### Calculate precision and recall

recalls_boxes <- list()
recalls_dlc <- list()
precision_boxes <- list()
precision_dlc <- list()

ious <- list()


for(vid in vids) {
  print(vid)
  gt <- eval(parse(text = paste0("gt", vid))) %>% filter(id < 10)
  pred <- eval(parse(text = paste0("pred", vid))) %>% filter(class == 0)
  dlc <- eval(parse(text = paste0("dlc", vid))) %>% 
    filter(!(bodypart %in% c("M_tail", "E_tail")), !is.na(x))
  
  
  
  iou_thres <- 0.5
  
  
  total <- 0
  box_matches <- 0
  dlc_matches <- 0
  
  for(i in 1:nrow(gt)) {
    gt_line <- gt[i, ]
    gt_frame <- unlist(gt_line[, "frame"])
    gt_box <- unlist(gt_line[, c("x", "y", "w", "h")])
    gt_box[1] <- gt_box[1] - gt_box[3] / 2
    gt_box[2] <- gt_box[2] - gt_box[4] /2
    
    pred_filtered <- pred %>%
      filter(frame == gt_frame)
    
    if(nrow(pred_filtered) != 0) {
      for(j in 1:nrow(pred_filtered)) {
        pred_box <- unlist(pred_filtered[j, c("x", "y", "w", "h")])
        if(calculate_iou(gt_box, pred_box) > iou_thres) {
          box_matches <- box_matches + 1
          break
        }
      }
    }
    
    
    

    dlc_filtered <- dlc %>%
      filter(frame == gt_frame)
    counter <- 0
    
    if(nrow(dlc_filtered) != 0) {
      for(j in 1:nrow(dlc_filtered)) {
        point <- unlist(dlc_filtered[j, c("x", "y")])
        if(check_point_inside_box(gt_box, point, 0.5)) {
          counter <- counter + 1
          
          if(counter>=2) {
            dlc_matches <- dlc_matches + 1
            break
          }
          
        }
      }
    }
    
    total <- total + 1
    
  }
  
  recalls_boxes <- append(recalls_boxes, box_matches / total)
  recalls_dlc <- append(recalls_dlc, dlc_matches / total)
  
  
  total <- 0
  box_matches <- 0
  
  for(i in 1:nrow(pred)) {
    pred_line <- pred[i, ]
    pred_frame <- unlist(pred_line[, "frame"])
    pred_box <- unlist(pred_line[, c("x", "y", "w", "h")])
    
    gt_filtered <- gt %>%
      filter(frame == pred_frame)
    
    
    if(nrow(gt_filtered) != 0) {
      for(j in 1:nrow(gt_filtered)) {
        gt_box <- unlist(gt_filtered[j, c("x", "y", "w", "h")])
        gt_box[1] <- gt_box[1] - gt_box[3] / 2
        gt_box[2] <- gt_box[2] - gt_box[4] /2
        if(calculate_iou(gt_box, pred_box) > iou_thres) {
          box_matches <- box_matches + 1
          break
        }
      }
    }
    
    total <- total + 1
  }
  
  precision_boxes <- append(precision_boxes, box_matches / total)
  
  total <- 0
  dlc_matches <- 0
  
  if(nrow(dlc) == 0) {
    precision_dlc <- append(precision_dlc, 0)
  } else {
    for(i in 1:nrow(dlc)) {
      
      dlc_line <- dlc[i, ]
      dlc_bodypart <- dlc[i, 3]
      
      dlc_frame <- unlist(dlc_line[, "frame"])
      dlc_point <- unlist(dlc_line[, c("x", "y")])
      
      gt_filtered <- gt %>%
        filter(frame == dlc_frame)
      counter <- 0
      
      if(nrow(gt_filtered) != 0) {
        for(j in 1:nrow(gt_filtered)) {
          gt_box <- unlist(gt_filtered[j, c("x", "y", "w", "h")])
          
          if(check_point_inside_box(gt_box, dlc_point, 0.5)) {
            counter <- counter + 1
            
            if(counter>=1) {
              dlc_matches <- dlc_matches + 1
              break
            }
            
          }
        }
      }
      
      total <- total + 1
      
      
    }
    
    precision_dlc <- append(precision_dlc, dlc_matches / total)
  }
  
  
}

cam <- c("close", "close", "close", "close", "far", "far", "far", "far", "top", "top", "top", "top")

results <- data.frame(cam = cam, 
                      vid = unlist(vids), 
                      recall_boxes = unlist(recalls_boxes),
                      recall_dlc = unlist(recalls_dlc),
                      precision_boxes = unlist(precision_boxes),
                      precision_dlc = unlist(precision_dlc))

results <- results %>%
  mutate(f1_boxes = 2 * (recall_boxes * precision_boxes) / (recall_boxes + precision_boxes),
         f1_dlc = 2 * (recall_dlc * precision_dlc) / (recall_dlc + precision_dlc))

## ignore this line (unless you want to compare spine and full model, then rerun the part above)
#results_spine_iou0.6_pcutoff0.0 <- results
results$f1_spine <- results_spine_iou0.6_pcutoff0.0$f1_dlc
results <- results %>%
  mutate(f1_zfull = ifelse(is.nan(f1_dlc), 0, f1_dlc),
        f1_spine = ifelse(is.nan(f1_spine), 0, f1_spine),
        f1_dlc = NULL)

results_summ <- results %>%
  summarise(f1_zfull = mean(f1_zfull),
            f1_spine = mean(f1_spine),
            f1_boxes = mean(f1_boxes)) %>%
  pivot_longer(cols = c("f1_boxes", "f1_zfull", "f1_spine"))

results %>%
  pivot_longer(cols = c("f1_boxes", "f1_zfull", "f1_spine")) %>% 
  ggplot(aes(x = name, y = value)) + 
  geom_point(aes(group = vid, col = cam), size = 2, alpha = 0.8) + 
  geom_line(aes(group = vid, col = cam)) +
  geom_point(data = results_summ, size = 3) +
  geom_line(data = results_summ, aes(group = 1)) +
  #lims(y = c(0,1)) +
  scale_color_manual(values = c("blue", "plum3", "orange")) +
  scale_x_discrete(labels = c("our model", "DLC_spine", "DLC_full")) +
  labs(y = "F1 Score", x = "", color = "Camera \nperspective") +
  theme_light() +
  theme(text = element_text(size = 18, color = "black"),
        panel.grid.major.x = element_blank(),  # Remove grid lines
        panel.border = element_blank(),
        axis.line.x = element_line(color = "black", size = 0.5),
        axis.line.y = element_line(color = "black", size = 0.5),
        axis.text = element_text(color = "black"),
        panel.grid.minor = element_blank())  # Remove grid lines)


ggsave("comparison.pdf", width = 7, height = 4)

data <- data.frame(
  Model = factor(c("our model / DLC-full", "DLC-spine"), levels = c("our model / DLC-full", "DLC-spine")),
  Hours = c(5, 12)
)

# Create the bar plot
ggplot(data, aes(x = Model, y = Hours)) +
  geom_chicklet(width = 0.5) +
  labs(x = "", y = "hours") +
  scale_y_continuous(breaks = c(0, 4, 8, 12)) +
  theme_light() +
  theme(text = element_text(size = 18, color = "black"),
        panel.grid.major.x = element_blank(),  # Remove grid lines
        panel.border = element_blank(),
        axis.line.x = element_line(color = "black", size = 0.5),
        axis.line.y = element_line(color = "black", size = 0.5),
        axis.text = element_text(color = "black"),
        panel.grid.minor = element_blank())  # Remove grid lines)

ggsave("time_comp.pdf", width = 7, height = 4)

### show images


library(magick)

vid <- 10

gt <- eval(parse(text = paste0("gt", vid))) %>% filter(id < 10)
pred <- eval(parse(text = paste0("pred", vid))) %>% filter(class == 0)
dlc <- eval(parse(text = paste0("dlc", vid))) %>% 
  filter(!(bodypart %in% c("M_tail", "E_tail")), !is.na(x))

sel_frame <- 120

for(sel_frame in c(80, 90, 100, 110, 120, 130, 140, 150, 160)) {
  img_path <- sprintf("images/obj_train_data/frame_%06d.PNG", sel_frame - 1)
  img <- image_read(paste0(img_path))
  
  w <- image_info(img)$width 
  h <- image_info(img)$height
  
  img1 <- img %>%
    image_colorize(opacity = 40, color = 'white') %>%
    image_draw()
  
  
  r <- filter(gt, frame == sel_frame)
  for(i in 1:nrow(r)) {
    rect((r[i, 'x'] - r[i, 'w'] / 2) * w ,(r[i, 'y'] - r[i, 'h'] / 2) * h , (r[i, 'x'] + r[i, 'w'] / 2) * w, (r[i, 'y'] + r[i, 'h'] / 2) * h,
         border = "green3", lwd = 12)
  }
  r <- filter(pred, frame == sel_frame)
  for(i in 1:nrow(r)) {
    rect((r[i, 'x']) * w ,(r[i, 'y']) * h , (r[i, 'x'] + r[i, 'w']) * w, (r[i, 'y'] + r[i, 'h']) * h,
         border = "black", lwd = 12, lty = "dashed")
  }
  r <- filter(dlc, frame == sel_frame)
  for(i in 1:nrow(r)) {
    symbols(r[i, 'x'] * w, r[i, 'y'] * h, circles = 12,
            bg = 2, inches = FALSE, add = TRUE)
  }
  
  image_write(img1, path = sprintf("images/output%03d.png", sel_frame), format = "png")
  
  dev.off()
  
}
