# Load required libraries #

if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")


###################################
# Create edx set and validation set
###################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind = "Rounding") # if using R 3.6.0: set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#########################
#Preprocessing Data
#########################

# Copy the original edx dataset in case we wish to reset
edxCopy<-edx

# Convert the Unix Time stamps to Date Time Format so that "lubridate"
# package can be used on it

edx<- edx %>% mutate(dateTime=as.Date(as.POSIXct(timestamp, origin="1970-01-01")))
head(edx)

#########################
#Data Inferences
#########################

# Plot frequency distribution of movies
edx %>% count(movieId) %>% ggplot(aes(n)) + 
  geom_histogram(bins = 50, color = "black") + scale_x_log10() + 
  ggtitle("Count of Movies")

# Find correlation between the movie rating and number of times it is rated
edx %>% filter(year(dateTime) >= 1993) %>% group_by(movieId) %>% 
  summarize(n = n(), years = 2017 - first(year(dateTime)), title = title[1], 
            rating = mean(rating)) %>% mutate(rate = n/years) %>% ggplot(aes(rate, rating)) + geom_point() + geom_smooth()

# Plot frequency distribution of users
edx %>% count(userId) %>% ggplot(aes(n)) + 
  geom_histogram(bins = 50, color = "black") + scale_x_log10() + 
  ggtitle("Count of Users")

# Plot correlation between avg rating given by user and number of users 
edx %>% filter(year(dateTime) >= 1993) %>% group_by(userId) %>%
  summarize(n = n(), years = 2017 - first(year(dateTime)), title = title[1],
            rating = mean(rating)) %>% mutate(rate = n/years) %>% 
  ggplot(aes(rate, rating)) + geom_point() + geom_smooth()

# Plot correlation between time and week of release
edx %>% mutate(date = round_date(dateTime, unit = "week")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth()

# Find whether some genres are rates much lower or higher than other genres
edx %>% group_by(genres) %>% 
  summarize(n = n(), avg = mean(rating)) %>% 
  filter(n >= 100000) %>% 
  mutate(genres = reorder(genres, avg)) %>% 
  ggplot(aes(x = genres, y = avg)) + 
  geom_point() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#########################
#Model Evaluations
#########################

# Create an RMSE function to repeatedly check the RMSE values for our various models

RMSE <- function(actual, predicted){
  sqrt(mean((actual - predicted)^2))
}

#########################
# Model 1: Average Rating
# All movies have same rating which is equal to mean of training set
#########################

rmse_avg<- RMSE(validation$rating,mean(edx$rating))
paste("RMSE - Average =",rmse_avg,sep = " ")

# Add the output to a results table. We will use this table 
# to compare various models

rmse_table <- data_frame(method = "Training Set average", 
                         RMSE = rmse_avg)

#########################
# Model 2: Factor Movie Bias
# Some movies are rated higher than other movies. This can be
# factored by a movie bias variable b_i which can then be added
# to our prediction model
#########################


# Find Movie Bias factor b_i
movie_bi <- edx %>% group_by(movieId) %>% 
  summarize(b_i = mean(rating - mean(edx$rating)))

# Compute RMSE value for the model based on its equation
predicted_ratings <- mean(edx$rating) + validation %>% 
  left_join(movie_bi, by='movieId') %>% pull(b_i)

rmse_bi <- RMSE(validation$rating, predicted_ratings)

# Add the output to a results table
rmse_table <- bind_rows(rmse_table, data_frame(method="Movie Bias Model",  
                                   RMSE = rmse_bi))

#########################
# Model 3: Factor User Bias
# Some users rate give higher rating to movies than other users. 
# This can be factored by a user bias variable b_u which can then be added
# to our prediction model
#########################

# Find User Bias factor b_u
user_bu <- edx %>% left_join(movie_bi, by='movieId') %>%
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mean(edx$rating) - b_i))

# Compute RMSE value for the model based on its equation
predicted_ratings <- validation %>% 
  left_join(movie_bi, by='movieId') %>%
  left_join(user_bu, by='userId') %>%
  mutate(pred = mean(edx$rating) + b_i + b_u) %>%
  pull(pred)

rmse_bu <- RMSE(validation$rating, predicted_ratings)

# Add the output to a results table
rmse_table <- bind_rows(rmse_table, data_frame(method="Movie + User Bias Model",  
                                               RMSE = rmse_bu))

#########################
# Model 4: Penalize movies with very few ratings
# Some movies are only rated very few times, and this adds
# noise to our model. We reduce this noise by adding penalty
# terms to RMSE for small number of ratings and high variation
# of b_i
#########################

# We find the top 20 movies as per Model 3 and find number
# of ratings given to them to prove our hypothesis

titles <- edx %>% select(movieId, title) %>% distinct()
edx %>% count(movieId) %>% left_join(movie_bi) %>%
  left_join(titles, by="movieId") %>% arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% slice(1:20) 

# We now try to find the RMSE values for our model
# We do not know optimal lambda. Hence we create a map with
# multiple lambda values

lambdas <- seq(1, 10, 1)

rmses <- sapply(lambdas, function(l){
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mean(edx$rating))/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mean(edx$rating))/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mean(edx$rating) + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})

# Add the output to a results table
rmse_table <- bind_rows(rmse_table, data_frame(method="Regularized Movie + User Bias Model",  
                                               RMSE = min(rmses)))
  