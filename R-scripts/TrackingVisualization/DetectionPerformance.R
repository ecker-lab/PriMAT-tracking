library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(glue)

df <- read.csv("DetectionResults/models200_detection.csv")

df_summ <- df %>% group_by(model, valset) %>%
  summarise(APsd = sd(AP), AP = mean(AP), AP50 = mean(AP50), AP75 = mean(AP75)) %>%
  mutate(in_domain = ifelse((model == "macaquecp" & valset == "MacaqueCopyPaste") |
                  (model == "macaquecpw" & valset == "MacaqueCopyPasteWild") |
                  (model == "macaquepose" & valset == "MacaquePose"), TRUE, FALSE))

det_plot <- df %>%
  ggplot(aes(x = valset, y = AP, group = interaction(model, seed), col = model)) + 
  geom_line(alpha = 0.5) +
  geom_line(data = df_summ, aes(group = model), size = 1) + 
  geom_point(data = df_summ, aes(group = model), size = 2) +
  scale_x_discrete(limits = c("MacaquePose", "MacaqueCopyPaste", "MacaqueCopyPasteWild"), 
                   labels = c("MacaquePose", "MacaqueCopyPaste", "MacaqueCopyPasteWild")) +
  labs(subtitle = "Detection", x = "Validation set") +
  theme_light() +
  theme(axis.text = element_text(size = 12),
        legend.position = "none")

df %>%
  ggplot(aes(x = valset, y = AP, group = interaction(model, seed), col = model)) + 
  geom_point(alpha = 0.5) +
  #geom_col(data = df_summ, aes(group = model, fill = model), size = 1, position = "dodge") + 
  geom_point(data = df_summ, aes(group = model), size = 3) +
  geom_line(data = df_summ, aes(group = model), size = 1) +
  scale_x_discrete(limits = c("MacaquePose", "MacaqueCopyPaste", "MacaqueCopyPasteWild"), 
                   labels = c("MacaquePose", "MacaqueCopyPaste", "MacaqueCopyPasteWild")) +
  labs(subtitle = "Detection", x = "Validation set") +
  theme_light() +
  theme(axis.text = element_text(size = 12),
        legend.position = "none",
        axis.text.x.top = element_text(size = 10))


det_plot <- df_summ %>%
  ggplot(aes(x = valset, y = model, fill = AP)) + 
  geom_tile() + 
  geom_tile(aes(x = 0.3, y = model, width = 0.2, height = 0.2, col = model), size = 2) +
  #geom_tile(aes(x = valset, y = 0.4, width = 0.2, height = 0.2, col = model), size = 2) +
  #geom_text(data = df, aes(group = interaction(model, valset), label = round(min(AP), 2)), nudge_y = -0.2) +
  geom_text(aes(label = round(AP,2))) +
  scale_x_discrete(limits = c("MacaqueCopyPasteWild", "MacaqueCopyPaste", "MacaquePose"), 
                   labels =  c("MacaqueCopy\nPasteWild", "MacaqueCopy\nPaste", "MacaquePose"),
                   expand = expansion(mult = c(0.5, 0.3))) +
  scale_y_discrete(
    limits = c("macaquepose", "macaquecp","macaquecpw"),
    labels = c( "MacaquePose", "MacaqueCopy\nPaste", "MacaqueCopy\nPasteWild")) +
  labs(subtitle = "Detection (Average Precision)", x = "Validation set", y = "Training set") +
  scale_fill_gradient(low = "white", high = "grey") +
  coord_flip() +
  # scale_color_manual(values = c("white", "black")) +
  theme_minimal() +
  theme(text = element_text(size = 14),
        plot.subtitle = element_text(size = 20),
        legend.position = "none")

det_plot <- df_summ %>%
  ggplot(aes(x = model, y = AP, group = valset, col = valset)) + 
  #geom_point(alpha = 0.5, position = position_dodge(0.3)) +
  geom_errorbar(aes(ymin = AP - APsd/sqrt(3), ymax = AP + APsd/sqrt(3)), width = 0.2, position = position_dodge(0.3), size = 1) +
  #geom_line(data = df_summ, aes(group = valset), size = 1, position = position_dodge(0.3)) +
  geom_point(data = df_summ, aes(size = in_domain), pch = 21, fill = NA, position = position_dodge(0.3)) +
  scale_x_discrete(limits = c("macaquepose", "macaquecp", "macaquecpw"), 
                   labels = c("MacaquePose", "MacaqueCopy\nPaste", "MacaqueCopy\nPasteWild")) +
  scale_color_discrete(limits = c("MacaquePose", "MacaqueCopyPasteWild","MacaqueCopyPaste" ), 
                       labels = c("MacaquePose", "MacaqueCopy\nPasteWild", "MacaqueCopy\nPaste")) +
  scale_size_manual(values = c(0,3)) +
  labs(subtitle = "Detection", x = "Training set", y = "mAP", size = "In-domain", col = "Validation set") +
  theme_light() +
  theme(text = element_text(size = 14),
        plot.subtitle = element_text(size = 20))


det_plot

## Tracking 
track <- read.csv("DetectionResults/tracking_results.csv")

track_summ <- track %>% group_by(epoch, train_data) %>%
  summarise(HOTAsd = sd(HOTA), HOTA = mean(HOTA), IDF1 = mean(IDF1), MOTA = mean(MOTA),
            AssA = mean(AssA), DetA = mean(DetA), IDs = mean(IDs))


tr_plot <- track %>%
  #filter(seed==2) %>%
  ggplot(aes(x = epoch, y = HOTA, group = interaction(seed, train_data), col = train_data)) +
  geom_line(alpha = 0.3) +
  geom_point(size = 2, alpha = 0.5) +
  geom_line(data = track_summ, aes(group = train_data), size = 1) +
  geom_point(data = track_summ, aes(group = train_data), size = 2) +
  labs(subtitle = "Tracking") +
  theme_light() +
  theme(axis.text = element_text(size = 12),
        axis.text.x.top = element_text(size = 10))

#with points of repetitions
tr_plot <- track %>%
  filter(epoch == 150) %>%
  ggplot(aes(x = train_data, y = HOTA, col = train_data)) +
  geom_point(size = 2, alpha = 0.5) +
  geom_point(data = track_summ %>% filter(epoch == 150), aes(group = train_data), 
             size = 4) +
  labs(subtitle = "Tracking") +
  theme_light() +
  labs(x = "Training set", y = "HOTA") +
  scale_x_discrete(
    limits = c("macaquecpw", "macaquecp", "macaquepose"),
    labels = c( "MacaqueCopy\nPasteWild", "MacaqueCopy\nPaste", "MacaquePose")) +
  theme(text = element_text(size = 14),
        plot.subtitle = element_text(size = 20),
        legend.position = "none",
        #axis.text.y = element_blank(),
        #axis.title.y = element_blank()
        )

#with errorbars
tr_plot <- track_summ %>%
  filter(epoch == 150) %>%
  ggplot(aes(x = train_data, y = HOTA)) +
  #geom_point(size = 2, alpha = 0.5) +
  geom_errorbar(aes(ymin = HOTA - HOTAsd/sqrt(3), ymax = HOTA + HOTAsd/sqrt(3)), width = 0.1, position = position_dodge(0.9), alpha = 0.5) +
  geom_point(data = track_summ %>% filter(epoch == 150), aes(group = train_data), 
             size = 4) +
  labs(subtitle = "Tracking") +
  theme_light() +
  labs(x = "Training set", y = "HOTA") +
  scale_x_discrete(
    limits = c("macaquepose", "macaquecp", "macaquecpw"),
    labels = c("MacaquePose", "MacaqueCopy\nPaste", "MacaqueCopy\nPasteWild")) +
  theme(text = element_text(size = 14),
        plot.subtitle = element_text(size = 20),
        legend.position = "none",
        #axis.text.y = element_blank(),
        #axis.title.y = element_blank()
  )

tr_plot


#old
#track <- read.csv("TrackingPerformance.csv")

#track_summ <- track %>% group_by(epoch, model) %>%
#  summarise(mota = mean(mota), idf1 = mean(idf1_roll))

#tr_plot <- track %>%
#ggplot(aes(x = epoch, y = idf1, group = interaction(seed, model), col = model)) +
#  geom_line(alpha = 0.3) +
#  geom_point(size = 2, alpha = 0.5) +
 # geom_line(data = track_summ, aes(group = model), size = 1) +
#  geom_point(data = track_summ, aes(group = model), size = 2) +
#  theme_light()
  

det_plot + tr_plot + plot_layout(guides = "collect") +
  plot_annotation(title = "MacaqueCopyPaste generalizes well")



ggsave("mcq_performance_plot.png", width = 9, height = 4)

det_plot
ggsave("detection_performance.pdf", width = 6, height = 4)

tr_plot
ggsave("tracking_performance.pdf", width = 4, height = 4)


##test
a <- data.frame(from = c(rep("Pose", 3), rep("CP", 3), rep("CPW", 3)),
                to = rep(c("Pose", "CP", "CPW"), 3),
                mAP = c(0.679, 0.532, 0.604, 0.645, 0.790, 0.701, 0.702, 0.638, 0.806))


ggplot(a, aes(x = to, y = mAP, group = from, col = from)) + 
  geom_line() +
  geom_point(size = 2)


#Pretraining comparison####


df <- read.csv("DetectionResults/lemur_pretraining.txt") %>%
  filter(!grepl("buffer", metric))

df %>%
  filter(!grepl("buffer", metric)) %>%
  mutate(metric = stringr::str_replace(metric, "lemur_i", "i")) %>%
  mutate(metric = stringr::str_replace(metric, "lemur_n", "n")) %>%
  separate(col = metric, into = c("model", "epoch"), sep = "_", extra = "merge") %>% 
  mutate(epoch = as.numeric(epoch)) %>%
  ggplot(aes(x = epoch * 2, y = HOTA, group = model, col = model)) + 
  geom_line(size = 1) +
  scale_color_manual(values = c("blue", "orange", "lightblue"),
                     labels = c("ImageNet", "MacaquePose", "No pretraining")) +
  labs(title = "Tracking performance", 
       subtitle = "Training on lemur dataset with different pretrained models", 
       x = "Epoch")

#Pretraining Macaques#####

animal <- "lemur" #macaque or lemur
df <- read.table(glue("DetectionResults/summary_{animal}_tracking.txt"), sep = ",", header = TRUE)


df <- df %>%
  #mutate(metric = stringr::str_replace(metric, "lemurs_i", "i")) %>%
  #mutate(metric = stringr::str_replace(metric, "lemurs_m", "m")) %>% 
  #mutate(metric = stringr::str_replace(metric, "lemurs_n", "n")) %>% 
  separate(col = metric, into = c("animal", "model", "epoch", "lr"), sep = "_", extra = "merge") %>%  
  mutate(epoch = as.numeric(epoch)) %>% 
  mutate(lr = ifelse(is.na(lr), "1e-4", lr))

df %>%
  filter(lr == "5e-5") %>%
  #filter(model == "imagenet") %>%
  ggplot(aes(x = epoch * 2, y = HOTA, group = model, col = model)) + 
  geom_line(size = 1) +
  scale_color_manual(values = c("blue", "orange", "darkgreen", "grey"),
                     labels = c("ImageNet", "MacaqueCopyPaste", "MacaqueCopyPasteWild", "No pretraining")) +
  labs(title = "Tracking performance", 
       subtitle = glue("Training on {animal} dataset with different pretrained models"), 
       x = "Epoch",
       color = "Pretrained on")


# Pretraining Final results
animal <- "lemur" #macaque or lemur
df <- read.table(glue("DetectionResults/summary_{animal}_tracking_3seeds.txt"), sep = ",", header = TRUE) %>%
  separate(col = metric, into = c("animal", "model", "seed"), sep = "_", extra = "merge")

df_summ <- df %>%
  group_by(model) %>%
  summarise(HOTAsd = sd(HOTA), HOTA = mean(HOTA))

df_summ %>%
  filter(model != "macaquecpw") %>%
  ggplot(aes(x = model, y = HOTA)) + 
  geom_point(size = 2) + 
  geom_errorbar(aes(ymin = HOTA - HOTAsd/sqrt(3), ymax = HOTA + HOTAsd/sqrt(3)), width = 0.1, position = position_dodge(0.9))

df %>%
  filter(seed == 1, model != "macaquecpw") %>%
  select(model, HOTA, MOTA, IDF1)

#Ablation results#####

df_lemurs <- read.table("DetectionResults/summary_lemur_hyperparams_new.txt", sep = ",", header = TRUE) %>%
  tidyr::separate(metric, into = c("animal", "conf_thres", "det_thres", "assoc_thres", "iou_prop", "matching_method"), sep = "_") 


df_lemurs %>% group_by(iou_prop, matching_method) %>%
  summarise(meanHOTA = mean(HOTA), maxHOTA = max(HOTA)) %>%
  ggplot(aes(x= iou_prop, y = meanHOTA, col = matching_method)) + geom_point(size = 2) +
  geom_point(aes(y = maxHOTA), size = 1)


df_lemurs %>% group_by(det_thres, matching_method) %>%
  summarise(meanHOTA = mean(HOTA), maxHOTA = max(HOTA)) %>%
  ggplot(aes(x= det_thres, y = meanHOTA, col = matching_method)) + geom_point(size = 2) +
  geom_point(aes(y = maxHOTA), size = 1)

df_lemurs %>% group_by(conf_thres, matching_method) %>%
  summarise(meanHOTA = mean(HOTA), maxHOTA = max(HOTA)) %>%
  ggplot(aes(x= conf_thres, y = meanHOTA, col = matching_method)) + geom_point(size = 2) +
  geom_point(aes(y = maxHOTA), size = 1)

df_lemurs %>% group_by(assoc_thres, matching_method) %>%
  summarise(meanHOTA = mean(HOTA), maxHOTA = max(HOTA)) %>%
  ggplot(aes(x= assoc_thres, y = meanHOTA, col = matching_method)) + geom_point(size = 2) +
  geom_point(aes(y = maxHOTA), size = 1)




df_macaques <- read.table("DetectionResults/summary_macaque_hyperparams_new.txt", sep = ",", header = TRUE) %>%
  tidyr::separate(metric, into = c("animal", "conf_thres", "det_thres", "assoc_thres", "iou_prop", "matching_method"), sep = "_") 

df_macaques2 <- read.table("DetectionResults/summary_macaque_hyperparams2.txt", sep = ",", header = TRUE) %>%
  tidyr::separate(metric, into = c("animal", "conf_thres", "det_thres", "new_thres","matching_method"), sep = "_") 



df_macaques %>% group_by(iou_prop, matching_method) %>%
  summarise(meanHOTA = mean(HOTA), maxHOTA = max(HOTA)) %>%
  ggplot(aes(x= iou_prop, y = meanHOTA, col = matching_method)) + geom_point(size = 2) +
  geom_point(aes(y = maxHOTA), size = 1)


df_macaques %>% group_by(det_thres, matching_method) %>%
  summarise(meanHOTA = mean(HOTA), maxHOTA = max(HOTA)) %>%
  ggplot(aes(x= det_thres, y = meanHOTA, col = matching_method)) + geom_point(size = 2) +
  geom_point(aes(y = maxHOTA), size = 1)

df_macaques %>% group_by(conf_thres, matching_method) %>%
  summarise(meanHOTA = mean(HOTA), maxHOTA = max(HOTA)) %>%
  ggplot(aes(x= conf_thres, y = meanHOTA, col = matching_method)) + geom_point(size = 2) +
  geom_point(aes(y = maxHOTA), size = 1)

df_macaques %>% group_by(assoc_thres, matching_method) %>%
  summarise(meanHOTA = mean(HOTA), maxHOTA = max(HOTA)) %>%
  ggplot(aes(x= assoc_thres, y = meanHOTA, col = matching_method)) + geom_point(size = 2) +
  geom_point(aes(y = maxHOTA), size = 1)




# Tables

## max value
df_lemurs %>%
  group_by(matching_method) %>%
  top_n(1, HOTA) %>%
  select(animal, conf_thres, det_thres, assoc_thres, iou_prop,matching_method,HOTA, MOTA, IDF1)

df_macaques %>%
  group_by(matching_method) %>%
  top_n(1, HOTA) %>%
  select(animal, conf_thres, det_thres, assoc_thres, iou_prop,matching_method,HOTA, MOTA, IDF1)

df_lemurs %>%
  group_by(conf_thres) %>%
  top_n(1, HOTA) %>%
  select(animal, conf_thres, det_thres, assoc_thres, iou_prop,matching_method,HOTA, MOTA, IDF1) 

df_macaques %>%
  group_by(conf_thres) %>%
  top_n(1, HOTA) %>%
  select(animal, conf_thres, det_thres, assoc_thres, iou_prop,matching_method,HOTA, MOTA, IDF1)


## mean value
df_lemurs %>%
  group_by(animal, matching_method) %>%
  summarise(HOTA = mean(HOTA), MOTA = mean(MOTA), IDF1= mean(IDF1)) %>%
  select(animal, matching_method,HOTA, MOTA, IDF1)

df_macaques %>%
  group_by(animal, matching_method) %>%
  summarise(HOTA = mean(HOTA), MOTA = mean(MOTA), IDF1= mean(IDF1)) %>%
  select(animal,matching_method,HOTA, MOTA, IDF1)

df_lemurs %>%
  group_by(animal, conf_thres) %>%
  summarise(HOTA = mean(HOTA), MOTA = mean(MOTA), IDF1= mean(IDF1)) %>%
  select(animal,conf_thres,HOTA, MOTA, IDF1) 

df_macaques %>%
  group_by(animal, conf_thres) %>%
  summarise(HOTA = mean(HOTA), MOTA = mean(MOTA), IDF1= mean(IDF1)) %>%
  select(animal, conf_thres, HOTA, MOTA, IDF1)


## fixed value
df_lemurs %>%
  filter(det_thres == 0.5, assoc_thres == 0.8, iou_prop == 0.8, conf_thres == 0.01) %>%
  select(matching_method, HOTA, MOTA, IDF1)

df_macaques %>%
  filter(det_thres == 0.5, assoc_thres == 0.8, iou_prop == 0.8, conf_thres == 0.04) %>%
  select(matching_method, HOTA, MOTA, IDF1)


df_lemurs %>%
  filter(det_thres == 0.5, assoc_thres == 0.8, iou_prop == 0.8, matching_method=="doublekalman") %>%
  select(conf_thres, HOTA, MOTA, IDF1)

df_macaques %>%
  filter(det_thres == 0.5, assoc_thres == 0.8, iou_prop == 0.8, matching_method=="singlekalman") %>%
  select(conf_thres, HOTA, MOTA, IDF1)


