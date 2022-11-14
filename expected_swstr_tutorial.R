library(tidyverse)
library(readr)
library(baseballr)
library(ggplot2)
library(xgboost)
library(Ckmeans.1d.dp)
library(pROC)
library(gt)
library(gtExtras)
library(mlbplotR)


#WHIFF MODEL EDA
#read in data, all of 2021 regular season
SavantData21 <- read_csv("savantData21.csv")

#add a count variable
SavantData21 <- SavantData21 %>% 
  mutate(count = paste(balls, strikes, 
                       sep = "-"))

#check weird cases
badCounts <- c("1-3", "4-1", "4-2")

#remove odd cases
pitches <- SavantData21 %>% 
  filter(!(count %in% badCounts))

#change date format
SavantData21$game_date <- as.Date(SavantData21$game_date, "%Y/%m/%d")

#get player info
mlbplayerids <- baseballr::get_chadwick_lu()

#add names to savant data
savantIDS <- mlbplayerids %>% 
  select(key_mlbam, name_first, name_last) %>% 
  mutate(name = paste(name_first, name_last, sep = " ")) %>% 
  filter(!is.na(key_mlbam))

#batter names
pitches <- left_join(pitches, savantIDS, by = c("batter" = "key_mlbam")) %>% 
  select(-name_first, -name_last) %>% 
  rename(batter_name = name)

#pitcher names
pitches <- left_join(pitches, savantIDS, by = c("pitcher" = "key_mlbam")) %>% 
  select(-name_first, -name_last) %>% 
  rename(pitcher_name = name)

#number of swstr
swstr <- c("swinging_strike", "swinging_strike_blocked")

#number of swinging strikes
num_swstr <- pitches %>% 
  filter(description %in% swstr) %>% 
  count() %>% 
  as.vector()

#number of pitches
nrow(pitches)

#average swstr rate
num_swstr/nrow(pitches) * 100

#add in binary swinging strike variable // going to be the target variable
pitches <- pitches %>% 
  mutate(swstr = if_else(description %in% swstr, 1, 0))

#only doing fastballs for demonstration purposes
pitches <- pitches %>% 
  filter(pitch_type == "FF")

#histogram of fastball velo
ggplot(pitches, aes(x = release_speed)) +
  geom_histogram() +
  labs(x = "Velocity",
       y = "Count", 
       title = "Distribution of Fastball Velocity") +
  #theme built in R
  theme_minimal() 

#standardizing movement numbers
pitches2 <- pitches %>% 
  mutate(
    release_pos_x_adj = ifelse(p_throws == "R", -release_pos_x, release_pos_x),
    pfx_x_adj = ifelse(p_throws == "R", -pfx_x, pfx_x)
  )

#drop unnecessary columns
pitches3 <- pitches2 %>% 
  #drop deprecated columns
  select(-spin_dir, -spin_rate_deprecated, -break_angle_deprecated, -break_length_deprecated) %>% 
  ungroup()

#remove NAs
pitches4 <- pitches3 %>% 
  filter(!is.na(release_spin_rate), !is.na(release_extension), !is.na(release_speed), !is.na(pfx_x),
         !is.na(pfx_z), !is.na(swstr))

#add a pitch id for joining later after model is run
pitches4 <- pitches4 %>% 
  mutate(playid = row_number())

#model inputs
#selecting the features we want to use
data_model <- pitches4 %>% 
  select(swstr, pfx_x_adj, pfx_z, release_pos_x_adj, release_pos_z, plate_x, plate_z
         , release_speed, release_spin_rate, release_extension)

View(data_model)

#get training data
#making 80/20 split
sample <- sample(c(TRUE, FALSE), nrow(data_model), replace=TRUE, prob=c(0.8,0.2))

#make them matrices for xgboost input
train <- as.matrix(data_model[sample, ])
test <- as.matrix(data_model[!sample, ])

#running the actual model
swstr_model <-
  xgboost(
    data = train[, 2:10],
    label = train[, 1],
    nrounds = 500,
    #we want predictions of a whiff or not
    objective = "binary:logistic",
    early_stopping_rounds = 3,
    max_depth = 5,
    eta = .3, 
  )  

#predict values
predict <- predict(swstr_model, test[, 2:10])

#join to test data
model_results <- predict %>% 
  as_tibble() %>% 
  bind_cols(test)

View(model_results)

#evaluating model performance
#pretty low r^2, needs future improvement
cor(model_results$value, model_results$swstr)^2

#variable importance plot
importance <- xgboost::xgb.importance(
  feature_names = colnames(swstr_model),
  model = swstr_model
)
xgboost::xgb.ggplot.importance(importance_matrix = importance)

#convert dataset to matrix
data_model_matrix <- as.matrix(data_model)

#for whole dataset now
swstr_model <-
  xgboost(
    data = data_model_matrix[, 2:10],
    label = data_model_matrix[, 1],
    nrounds = 500,
    #we want predictions of a whiff or not
    objective = "binary:logistic",
    early_stopping_rounds = 3,
    max_depth = 5,
    eta = .3, 
  )  

#predict on whole dataset now
predict_whole <- predict(swstr_model, data_model_matrix[, 2:10])

#final results, back to data frame
model_results_whole <- predict_whole %>% 
  as_tibble() %>% 
  bind_cols(data_model) %>% 
  as.data.frame()

#average swstr probabilities by actual result, validating result
model_results_whole %>% 
  group_by(swstr) %>% 
  summarise(x_swstr = mean(value))

#rename value to actual name
model_results_whole <- model_results_whole %>% 
  rename(x_swstr = value)

#joining back to original dataset
swstr_final <- left_join(pitches4, model_results_whole, by = c("swstr", "pfx_x_adj", "pfx_z", "plate_z",
                                                       "plate_x", "release_pos_z", "release_pos_x_adj",
                                                       "release_speed", "release_spin_rate", "release_extension"))

View(swstr_final)

#2021 xSWSTR leaders on fastballs
pitchers <- swstr_final %>% 
  group_by(pitcher_name, pitcher) %>% 
  summarise(x_swstr = mean(x_swstr), swstr = mean(swstr), pitches = n()) %>% 
  arrange(desc(x_swstr)) %>% 
  filter(pitches >= 100) 

#table of top ten
pitchers[1:10, ] %>% 
  select(pitcher_name, x_swstr, swstr, pitches) %>% 
  ungroup() %>% 
  gt() %>% 
  gt_hulk_col_numeric(c("x_swstr", "swstr", "pitches")) %>% 
  cols_align(align = "center",
             columns = everything()) %>% 
  cols_label(x_swstr = "Expected swSTR%",
             swstr = "swSTR%",
             pitches = "Pitches", 
             pitcher_name = "Pitcher") %>% 
  tab_header(title = "2021 Expected swSTR% Leaders",
             subtitle = "Minimum 100 Pitches | Fastballs Only") %>% 
  fmt_percent(columns = c("x_swstr", "swstr"))
  


#under/overperformers by swinging strike rate
pitchers2 <- pitchers %>% 
  filter(pitches >= 1000) %>% 
  rename(player_id = pitcher)

View(pitchers2)

#actual plot
ggplot(pitchers2, aes(x = x_swstr, y = swstr)) +
  geom_abline(intercept = 0, slope = 1, alpha = 0.6, color = "orange", linetype = "dashed") +
  geom_hline(yintercept = mean(pitchers$swstr), alpha = 0.6, linetype = "dashed", color = "red") +
  geom_vline(xintercept = mean(pitchers$x_swstr), alpha = 0.6, linetype = "dashed", color = "red") +
  labs(
    x = "Expected swSTR%",
    y = "swSTR%",
    title = "2021 Fastball Pitchers by Swinging Strike Rates",
    subtitle = "Minimum 100 Pitches"
  ) +
  theme_minimal() +
  mlbplotR::geom_mlb_headshots(aes(player_id = player_id), width = 0.15) +


