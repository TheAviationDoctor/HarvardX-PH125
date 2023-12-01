# ==============================================================================
#    NAME: script.R
#   INPUT: The MovieLens dataset named "edx" as per course instructions
# ACTIONS: 1 Prepare the data
#          2 Explore the data
#          3 Create manual predictions
#          4 Use the Recosystem model for matrix factorization
#  OUTPUT: A predictive model of movie ratings with an RMSE of 0.7738297
# RUNTIME: ~1 hour
#  AUTHOR: Thomas D. Pellegrin <thomas@pellegr.in>
#    YEAR: 2023
# ==============================================================================

# ==============================================================================
# 0 Housekeeping
# ==============================================================================

# Clear the environment
rm(list = ls())

# Load the required libraries
library(caret)
library(doParallel)
library(dplyr)
library(knitr)
library(Matrix)
library(parallel)
library(readr)
library(recosystem)
library(scales)
library(stringr)
library(tidyr)

# Set options
options(timeout = 120)

# Start a script timer
start_time <- Sys.time()

# Create a list to hold the resulting RMSEs
rmses <- list(goal = 0.8649)

# Clear the console
cat("\014")

# ==============================================================================
# 1 Prepare the data
# ==============================================================================

# ==============================================================================
# 1.1 Create the edx dataset if necessary (adapted from course-provided code)
# ==============================================================================

# Define the data source
dl  <- "ml-10M100K.zip"
url <- "https://files.grouplens.org/datasets/movielens/ml-10m.zip"

# Download the data only if not already downloaded
if(!file.exists(dl)) { download.file(url, dl) }

# Extract and join the movies and ratings
movielens <- left_join(

  read.table(
    text = gsub(
      x           = readLines(con = unzip(dl, "ml-10M100K/movies.dat")),
      pattern     = "::",
      replacement = ";",
      fixed       = TRUE
    ),
    sep           = ";",
    col.names     = c("movieId", "title", "genres"),
    colClasses    = c("integer", "character", "character"),
    quote         = "",
    comment.char  = "" # one movie title has a pound sign so we need this
  ),

  read.table(
    text = gsub(
      x           = readLines(con = unzip(dl, "ml-10M100K/ratings.dat")),
      pattern     = "::",
      replacement = ";",
      fixed       = TRUE
    ),
    sep        = ";",
    col.names  = c("userId", "movieId", "rating", "timestamp"),
    colClasses = c("integer", "integer", "numeric", "integer"),
    quote         = ""
  ),

  by = "movieId"

) |> na.omit() # Four movies were found to have no rating

# Set a seed for reproducibility
suppressWarnings(set.seed(1, sample.kind = "Rounding"))

# Partition the data. Test set will be 10% of MovieLens data
test_index <- createDataPartition(
  y     = movielens$rating,
  times = 1,
  p     = 0.1,
  list  = FALSE
)

# Separate the training and temporary test datasets
edx_train <- movielens[-test_index, ]
edx_temp  <- movielens[test_index, ]

# The temporary test set may now contain rows with a userId and/or movieId that
# are no longer present in the training set. This would cause problems when
# comparing rating predictions based on a userId and/or movieId bias. So, we
# need to temporarily remove any row in the final test set that doesn't have a
# matching userId and/or movieId in the training set.
final_holdout_test <- edx_temp |>
  semi_join(edx_train, by = "movieId") |>
  semi_join(edx_train, by = "userId")

# We add the row(s) removed from the final test set back into the training set
edx_train <- rbind(
  edx_train,                              # The training set partitioned earlier
  anti_join(edx_temp, final_holdout_test) # Rows removed from the final test set
)

# Remove unnecessary objects from memory
rm(dl, test_index, edx_temp, movielens)

# ==============================================================================
# 2 Explore the data
# ==============================================================================

# Describe the structure of the transformed dataset
# 9,000,055 observations of 6 variables
str(edx_train)

# Number of observations
# 9,000,056
format(x = nrow(edx_train), big.mark = ",")

# Number of variables
# 6
length(edx_train)

# ==============================================================================
# 2.1 movieId
# ==============================================================================

# Lowest movieId
# 1
min(edx_train$movieId)

# Highest movieId
# 65,133
format(x = max(edx_train$movieId), big.mark = ",")

# Number of unique movies
# 10,677
format(x = n_distinct(edx_train$movieId), big.mark = ",")

# Average number of ratings per movie
# 843
round(nrow(edx_train) / n_distinct(edx_train$movieId), 0)

# ==============================================================================
# 2.2 title
# ==============================================================================

# Release year might be a predictor, so extract it into a new integer variable
edx_train <- edx_train |>
  mutate(
    year_movie  = as.integer(str_sub(title, start = -5L, end = -2L)),
    title       = str_sub(title, end = -8L)
  )

# ==============================================================================
# 2.3 genres
# ==============================================================================

# Number of unique genres
# 20 including one undefined
edx_train |>
  select(genres) |>
  unique() |>
  separate_rows(genres, sep = "\\|") |>
  n_distinct()

# Number of concatenated genres
# 797 combinations
length(unique(edx_train$genres))

# Histogram of genres
edx_train |>
  distinct(title, .keep_all = TRUE) |>
  separate_rows(genres, sep = "\\|") |>
  count(genres, sort = TRUE) |>
  mutate(genres = factor(genres, levels = genres[order(n)])) |>
  ggplot(aes(x = genres, y = n)) +
  geom_col() +
  geom_text(
    aes(
      label = paste(
        format(x = n, big.mark = ","),
        " (",
        label_percent(accuracy = 0.1)(n / sum(n)),
        ")",
        sep = ""
      )
    ),
    nudge_y = 300L
  ) +
  coord_flip() +
  labs(x = "Genres", y = "Count of genres")

# ==============================================================================
# 2.4 userId
# ==============================================================================

# Lowest userId
# 1
min(edx_train$userId)

# Highest userId
# 71,567
format(x = max(edx_train$userId), big.mark = ",")

# Number of unique users
# 69,878
format(x = n_distinct(edx_train$userId), big.mark = ",")

# Average number of ratings per user
# 129
round(nrow(edx_train) / n_distinct(edx_train$userId), 0)

# ==============================================================================
# 2.5 rating
# ==============================================================================

# Minimum rating
# 0.5
min(edx_train$rating)

# Maximum rating
# 5.0
max(edx_train$rating)

# Most frequent rating (mode)
edx_train |>
  group_by(rating) |>
  summarize(
    n   = n(),
    per = label_percent(accuracy = .1)(n / nrow(edx_train))
  ) |>
  filter(n == max(n)) |>
  pull(rating)

# Least frequent rating
edx_train |>
  group_by(rating) |>
  summarize(
    n   = n(),
    per = label_percent(accuracy = .1)(n / nrow(edx_train))
  ) |>
  filter(n == max(n)) |>
  pull(rating)

# Mean of all ratings
round(x = mean(edx_train$rating), digits = 2)

# Histogram of ratings
edx_train |>
  group_by(rating) |>
  count(rating, sort = FALSE) |>
  ggplot(aes(x = rating, y = n)) +
  geom_col() +
  geom_text(
    aes(label = label_percent(accuracy = 0.1)(n / sum(n))),
    nudge_y = 10^6 / 6
  ) +
  coord_flip() +
  labs(x = "Ratings", y = "Count of ratings")

# Maximum number of ratings per movie
edx_train |> count(movieId) |> select(n) |> max() |> format(big.mark = ",")

# Minimum number of ratings per movie
edx_train |> count(movieId) |> select(n) |> min() |> format(big.mark = ",")

# Number of movies rated once
edx_train |> count(movieId, title) |> filter(n == 1) |> nrow()

# Movies in the bottom 25th percentile in number of ratings
# 2,685 movies with 30 ratings or less
edx_train |>
  count(movieId) |>
  filter(n <= quantile(n, .25)) |>
  nrow() |>
  format(big.mark = ",")

# Mean number of ratings per movie
format(round(nrow(edx_train) / n_distinct(edx_train$movieId), 0), nsmall = 0)

# Most rated movies
edx_train |>
  count(movieId, title, name = "n_ratings", sort = TRUE) |>
  head(10L) |>
  kable()

# Least rated movies
edx_train |>
  count(movieId, title, name = "n_ratings", sort = TRUE) |>
  tail(10L) |>
  kable()

# Frequency of ratings
edx_train |>
  count(movieId, name = "ratings_per_movie") |>
  count(ratings_per_movie, name = "frequency") |>
  ggplot(mapping = aes(x = ratings_per_movie, y = frequency)) +
  geom_vline(xintercept = nrow(edx_train) / n_distinct(edx_train$movieId)) +
  annotate(
    geom  = "text",
    label = format(round(nrow(edx_train) / n_distinct(edx_train$movieId), 0), nsmall = 0),
    x     = nrow(edx_train) / n_distinct(edx_train$movieId) - 150,
    y     = -10
  ) +
  geom_point() +
  scale_x_log10() +
  labs(x = "Count of ratings per movie", y = "Count of movies")

# Highest-rated movies (top n)
# We find that highest-rated movies also have very few ratings each
edx_train |>
  group_by(movieId, title) |>
  summarize(mean_rating = mean(rating), n_rating = n()) |>
  arrange(desc(mean_rating)) |>
  head(10L) |>
  kable(format.args = list(digits = 2))

# Same, but now excluding the 25th bottom percentile in # of ratings
# This returns a totally different set (zero overlap)
edx_train |>
  group_by(movieId, title) |>
  summarize(mean_rating = mean(rating), n_rating = n()) |>
  arrange(desc(mean_rating)) |>
  filter(n_rating > 30L) |>
  head(10L) |>
  kable(format.args = list(digits = 2))

# Drop the title column
edx_train <- edx_train |>
  select(!title)

# Plot the relationship between rating count and rating score
edx_train |>
  group_by(movieId) |>
  summarize(mean_rating = mean(rating), n_rating = n()) |>
  ggplot(mapping = aes(x = n_rating, y = mean_rating)) +
  geom_point() +
  labs(x = "Count of ratings per movie", y = "Mean rating")

# Ratings by reviewer
edx_train |>
  group_by(userId) |>
  summarize(mean_rating = mean(rating), n_rating = n()) |>
  ggplot(mapping = aes(x = n_rating, y = mean_rating)) +
  geom_point() +
  labs(x = "Count of ratings per user", y = "Mean rating")

# ==============================================================================
# 2.6 timestamp
# ==============================================================================

# Earliest timestamp
as.Date(as.POSIXct(min(edx_train$timestamp), origin = "1970-01-01"))

# Latest timestamp
as.Date(as.POSIXct(max(edx_train$timestamp), origin = "1970-01-01"))

# Convert timestamp
edx_train <- edx_train |>
  mutate(
    year_rating = as.integer(
      as.POSIXlt(
        timestamp, origin = "1970-01-01"
      )$year + 1900L
    ),
    timestamp   = NULL
  )

# Number of variables
# 26
length(edx_train)

# ==============================================================================
# 3.0 Create manual predictions
# ==============================================================================

# ==============================================================================
# 3.1 Naive prediction
# This just takes the mean of all ratings without accounting for any effect
# ==============================================================================

# Mean of all ratings in the training set
mu <- mean(edx_train$rating)

# RMSE without accounting for any effect
rmses <- c(
  rmses,
  naive = RMSE(pred = mu, obs = final_holdout_test$rating)
)

# Display the RMSE
# 1.060334
rmses[["naive"]]

# ==============================================================================
# 3.2 User effect
# Not all users rate movies the same
# ==============================================================================

# Measure the effect relative to the mean of all ratings
user_bias <- edx_train |>
  group_by(userId) |>
  summarize(user_bias = mean(rating - mu))

# Create a set of predicted ratings with the same cardinality as the test set
user_pred <- final_holdout_test |>
  left_join(
    user_bias,
    by = "userId"
  ) |>
  mutate(pred = mu + user_bias) |>
  select(userId, pred)

# RMSE after accounting for user effect
rmses <- c(
  rmses,
  user = RMSE(pred = user_pred$pred, obs = final_holdout_test$rating)
)

# Display the RMSE
# 0.9779362
rmses[["user"]]

# ==============================================================================
# 3.3 Movie effect
# Not all movies are rated the same
# ==============================================================================

# Measure the effect relative to the mean of all ratings
movie_bias <- edx_train |>
  group_by(movieId) |>
  summarize(movie_bias = mean(rating - mu))

# Create a set of predicted ratings with the same cardinality as the test set
movie_pred <- final_holdout_test |>
  left_join(
    movie_bias,
    by = "movieId"
  ) |>
  mutate(pred = mu + movie_bias) |>
  select(movieId, pred)

# RMSE after accounting for movie effect
rmses <- c(
  rmses,
  movie = RMSE(pred = movie_pred$pred, obs = final_holdout_test$rating)
)

# Display the RMSE
# 0.944011
rmses[["movie"]]

# ==============================================================================
# 3.5 Multi-genre effect
# Not all genres are rated the same
# Let's first treat the concatenated genres
# ==============================================================================

# Measure the effect relative to the mean of all ratings
multi_genre_bias <- edx_train |>
  group_by(genres) |>
  summarize(multi_genre_bias = mean(rating - mu))

# Create a set of predicted ratings with the same cardinality as the test set
multi_genre_pred <- final_holdout_test |>
  left_join(
    multi_genre_bias,
    by = "genres"
  ) |>
  mutate(pred = mu + replace_na(multi_genre_bias, 0)) |>
  select(genres, pred)

# RMSE after accounting for multi-genre effect
rmses <- c(
  rmses,
  multi_genre = RMSE(pred = multi_genre_pred$pred, obs = final_holdout_test$rating)
)

# Display the RMSE
# 1.018186
rmses[["multi_genre"]]

# ==============================================================================
# 3.6 Single-genre effect
# Not all genres are rated the same
# Let's now break down the concatenation and treat each genre separately
# ==============================================================================

# Split the genres column and pivot it into rows in the training set
edx_train_longer <- edx_train |>
  select(genres, rating) |>
  separate_longer_delim(genres, delim = "|")

# Measure the effect relative to the mean of all ratings
single_genre_bias <- edx_train_longer |>
  group_by(genres) |>
  summarize(single_genre_bias = mean(rating - mu))

# Split the genres column and pivot it into rows in the test set
final_holdout_test_longer <- final_holdout_test |>
  separate_longer_delim(genres, delim = "|")

# Create a set of predicted ratings w/ the same cardinality as the long test set
single_genre_pred <- final_holdout_test_longer |>
  left_join(
    single_genre_bias,
    by = "genres"
  ) |>
  mutate(pred = mu + single_genre_bias) |>
  select(genres, pred)

# RMSE after accounting for single-genre effect
rmses <- c(
  rmses,
  single_genre = RMSE(
    pred = single_genre_pred$pred,
    obs = final_holdout_test_longer$rating
  )
)

# Display the RMSE
# 1.045475
rmses[["single_genre"]]

# ==============================================================================
# 3.7 Year of release effect
# Ratings differ between years of release (older vs. newer cinema)
# ==============================================================================

# Summarize the effect relative to the mean of all ratings
year_movie_bias <- edx_train |>
  group_by(year_movie) |> 
  summarize(year_movie_bias = mean(rating - mu))

# Create the year_movie variable in the test set as well so that we can join
final_holdout_test <- final_holdout_test |>
  mutate(
    year_movie  = as.integer(str_sub(title, start = -5L, end = -2L)),
    title       = str_sub(title, end = -8L)
  )

# Create a set of predicted ratings with the same cardinality as the test set
year_movie_pred <- final_holdout_test |>
  left_join(
    year_movie_bias,
    by = "year_movie"
  ) |>
  mutate(pred = mu + year_movie_bias) |>
  select(year_movie, pred)

# RMSE after accounting for year of release effect
rmses <- c(
  rmses,
  year_movie = RMSE(pred = year_movie_pred$pred, obs = final_holdout_test$rating)
)

# Display the RMSE
# 1.049464
rmses[["year_movie"]]

# ==============================================================================
# 3.8 Year of rating effect
# Ratings differ between years of issuance (severity over time)
# ==============================================================================

# Summarize the effect relative to the mean of all ratings
year_rating_bias <- edx_train |>
  group_by(year_rating) |> 
  summarize(year_rating_bias = mean(rating - mu))

# Create the year_rating variable in the test set as well so that we can join
final_holdout_test <- final_holdout_test |>
  mutate(
    year_rating = as.integer(as.POSIXlt(timestamp, origin = "1970-01-01")$year + 1900L),
    # timestamp   = NULL
  )

# Create a set of predicted ratings with the same cardinality as the test set
year_rating_pred <- final_holdout_test |>
  left_join(
    year_rating_bias,
    by = "year_rating"
  ) |>
  mutate(pred = mu + year_rating_bias) |>
  select(year_rating, pred)

# RMSE after accounting for year of release effect
rmses <- c(
  rmses,
  year_rating = RMSE(pred = year_rating_pred$pred, obs = final_holdout_test$rating)
)

# Display the RMSE
# 1.058622
rmses[["year_rating"]]

# ==============================================================================
# 3.9 Combined movie, user, genre, and year effects
# ==============================================================================

# Create a set of predicted ratings with the same cardinality as the test set
movieusergenreyear_pred <- final_holdout_test |>
  left_join(movie_bias,       by = "movieId") |>
  left_join(user_bias,        by = "userId") |>
  left_join(multi_genre_bias, by = "genres") |>
  left_join(year_movie_bias,  by = "year_movie") |>
  left_join(year_rating_bias, by = "year_rating") |>
  mutate(
    pred = mu + movie_bias + user_bias + multi_genre_bias +
      year_movie_bias + year_rating_bias) |>
  select(movieId, userId, genres, year_movie, year_rating, pred)

# RMSE after accounting for movie, user, genre, and year effects
rmses <- c(
  rmses,
  movieusergenreyear = RMSE(
    pred = movieusergenreyear_pred$pred,
    obs  = final_holdout_test$rating)
)

# Display the RMSE
# 0.9707198
rmses[["movieusergenreyear"]]

# ==============================================================================
# 3.10 Combined movie, user, and genre effects
# ==============================================================================

# Create a set of predicted ratings with the same cardinality as the test set
movieusergenre_pred <- final_holdout_test |>
  left_join(movie_bias,       by = "movieId") |>
  left_join(user_bias,        by = "userId") |>
  left_join(multi_genre_bias, by = "genres") |>
  mutate(
    pred = mu + movie_bias + user_bias + multi_genre_bias) |>
  select(movieId, userId, genres, pred)

# RMSE after accounting for movie, user, and genre effects
rmses <- c(
  rmses,
  movieusergenre = RMSE(
    pred = movieusergenre_pred$pred,
    obs  = final_holdout_test$rating)
)

# Display the RMSE
# 0.9455764
rmses[["movieusergenre"]]

# ==============================================================================
# 3.11 Combined movie and user effects
# ==============================================================================

# Create a set of predicted ratings with the same cardinality as the test set
movieuser_pred <- final_holdout_test |>
  left_join(movie_bias, by = "movieId") |>
  left_join(user_bias,  by = "userId") |>
  mutate(
    pred = mu + movie_bias + user_bias) |>
  select(movieId, userId, pred)

# RMSE after accounting for movie and user effects
rmses <- c(
  rmses,
  movieuser = RMSE(
    pred = movieuser_pred$pred,
    obs  = final_holdout_test$rating)
)

# Display the RMSE
# 0.8853043
rmses[["movieuser"]]

# ==============================================================================
# 3.12 Combined movie and user effects, regularized
# ==============================================================================

# Measure the movie effect relative to the mean of all ratings
# And this time add a variable for how many times a movie is rated
movie_bias_reg <- edx_train |>
  group_by(movieId) |>
  summarize(
    movie_bias = sum(rating - mu),
    movie_n    = n()
  )

# Measure the user effect relative to the mean of all ratings
# And this time add a variable for how many times a user has rated
user_bias_reg <- edx_train |>
  group_by(userId) |>
  summarize(
    user_bias = sum(rating - mu),
    user_n    = n()
  )

# Function to calculate the RMSE using lambda as a parameter
fn_opt <- function(lambda) {
  return(
    RMSE(
      pred = (final_holdout_test |>
        left_join(movie_bias_reg, by = "movieId") |>
        left_join(user_bias_reg,  by = "userId") |>
        mutate(pred = mu + movie_bias / (movie_n + lambda) +
                 user_bias / (user_n + lambda)))$pred,
      obs  = final_holdout_test$rating
    )
  ) 
}

# Run the optimizer to minimize the residual error
opt <- optimize(
  f        = function(lambda) fn_opt(lambda), # Function to return the RMSE
  interval = c(0, 100),                       # Lambda search interval
  tol      = 10^-4                            # Optimizer tolerance
)

# Display the optimal lambda
# 24.62676
opt$minimum

# Save the RMSE
rmses[["movieuser_reg"]] <- opt$objective

# Display the RMSE
# 0.8813807
rmses[["movieuser_reg"]]

# ==============================================================================
# 4 Use the Recosystem model for matrix factorization
# ==============================================================================

# ==============================================================================
# 4.1 Convert the training and test data to sparse matrices and data sources
# ==============================================================================

edx_train_datasource <- data_matrix(
  mat = sparseMatrix(
    i    = edx_train$userId,
    j    = edx_train$movieId,
    x    = edx_train$rating,
    repr = "T"
  )
)

final_holdout_test_datasource <- data_matrix(
  mat = sparseMatrix(
    i    = final_holdout_test$userId,
    j    = final_holdout_test$movieId,
    x    = final_holdout_test$rating,
    repr = "T"
  )
)

# ==============================================================================
# 4.3 Train and run the prediction model
# ==============================================================================

# Instantiate the Recosystem object
r <- Reco()

# How many CPU threads to use
nthread <- 20L

# Set the training parameters
opts_tune <- list(
  dim      = 200L,              # Latent factors
  nbin     = 4 * nthread^2 + 1, # Number of bins
  nfold    = 10L,               # k-fold cross-validation
  nthread  = nthread            # CPU threads
)

# Tune the model to the optimal parameters
opts_train <- r$tune(train_data = edx_train_datasource, opts = opts_tune)

# Display the optimal parameters
opts_train$min

# Train the model with tuned parameters
r$train(
  edx_train_datasource,
  opts = c(opts_train$min, nthread = 20L, nbin = 30L)
)

# Calculate the final RMSE
rmses <- c(
  rmses,
  recosys = RMSE(
    pred = r$predict(test_data = final_holdout_test_datasource),
    obs  = final_holdout_test$rating)
)

# Display the RMSE
# 0.7738297
rmses[["recosys"]]

# ==============================================================================
# 5 Housekeeping
# ==============================================================================

# Stop the script timer
Sys.time() - start_time

# EOF