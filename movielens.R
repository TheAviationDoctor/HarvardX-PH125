# ==============================================================================
#    NAME: script.R
#   INPUT: The MovieLens dataset named "edx" as per course instructions
# ACTIONS: 
#  OUTPUT: 
# RUNTIME: N/A
#  AUTHOR: Thomas D. Pellegrin <thomas@pellegr.in>
#    YEAR: 2023
# ==============================================================================

# ==============================================================================
# 0 Housekeeping
# ==============================================================================

# Load the required libraries
library(caret)
library(dplyr)
library(knitr)
library(readr)
library(stringr)

# Set options
options(timeout = 120)

# Start a script timer
start_time <- Sys.time()

# ==============================================================================
# 1 Prepare the data
# ==============================================================================

# ==============================================================================
# 1.1 Create the edx dataset if necessary (adapted from course-provided code)
# ==============================================================================

# Download the data
dl <- "ml-10M100K.zip"
if(!file.exists(dl)) {
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
}

# Extract and join the ratings and movies
movielens <- left_join(
  
  ratings <- read.table(
    text = gsub(
      x           = readLines(con = unzip(dl, "ml-10M100K/ratings.dat")),
      pattern     = "::",
      replacement = ";",
      fixed       = TRUE
    ),
    sep        = ";",
    col.names  = c("userId", "movieId", "rating", "timestamp"),
    colClasses = c("integer", "integer", "numeric", "integer")
  ),
  
  movies <- read.table(
    text = gsub(
      x           = readLines(con = unzip(dl, "ml-10M100K/movies.dat")),
      pattern     = "::",
      replacement = ";",
      fixed       = TRUE
    ),
    sep        = ";",
    col.names  = c("movieId", "title", "genres"),
    colClasses = c("integer", "character", "character")
  ),
  
  by = "movieId"
)

# Set a seed for reproducibility
set.seed(1, sample.kind = "Rounding") # if using R 3.6 or later

# Partition the data. Final hold-out test set will be 10% of MovieLens data
test_index <- createDataPartition(
  y     = movielens$rating,
  times = 1,
  p     = 0.1,
  list  = FALSE
)

# Separate the test set from the edx dataset
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

# Remove unnecessary objects from memory
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# ==============================================================================
# 1.2 Describe and transform the data
# ==============================================================================

# Describe the data
# 9,000,055 observations of 6 variables
str(edx)

# Year of the movie and of the rating, and their distance, might be predictors,
# so extract them, remove year from titles, & remove no longer needed timestamps
edx <- edx %>%
    mutate(
      year_movie  = as.integer(str_sub(title, start = -5L, end = -2L)),
      year_rating = as.integer(as.POSIXlt(timestamp, origin = "1970-01-01")$year + 1900L),
      year_dist   = as.integer(abs(year_movie - year_rating)),
      title       = str_sub(title, end = -8L),
      timestamp   = NULL
    )

# ==============================================================================
# 2 Exploratory analysis to identify what might be useful predictors of ratings
# ==============================================================================

# ==============================================================================
# 2.1 Initial data exploration
# ==============================================================================

# Describe the structure of the transformed dataset
# 9,000,055 observations of 8 variables
str(df)

# Number of unique movies
# 10,677
n_distinct(df$movieId)

# Number of unique reviewers
# 69,878
n_distinct(df$userId)

# Mean number of ratings per movie
# 842.9386
nrow(df) / n_distinct(df$movieId)

# Mean number of ratings per reviewer
# 128.7967
nrow(df) / n_distinct(df$userId)

# List unique genres
# 19 genres + one blank, drama is the most frequent
df %>%
  distinct(title, .keep_all = TRUE) %>%
  separate_rows(genres, sep = "\\|") %>%
  count(genres, sort = TRUE)

# ==============================================================================
# 2.2 Overall distribution of ratings
# ==============================================================================

# Ratings distribution table
# Most frequent is 4.0 (28.8% of all ratings), followed by 3.0 (23.6%)
# Least frequent is 0.5 (0.9% of all ratings), followed by 1.5 (1.18%)
# There is no rating of zero
df %>%
  group_by(rating) %>%
  summarize(n_ratings = n(), n_per = n() / nrow(df) * 100L)

# Ratings distribution histogram
df %>%
  ggplot(mapping = aes(x = rating)) +
  geom_histogram() +
  theme_light()

# ==============================================================================
# 2.3 Number of ratings per movie
# ==============================================================================

# Ratings per movie
# Min is 1 rating per movie, max is 31,362, mean is 842.9, median is 122
df %>%
  count(movieId, name = "n_ratings") %>%
  select(n_ratings) %>%
  summary()

# Mode (most frequent number of ratings per movie)
# The most frequent number is 4 ratings per movie (154 movies have that)
df %>%
  count(movieId, name = "n_ratings") %>%
  count(n_ratings, name = "frequency", sort = TRUE) %>%
  head(1L)

# Most rated movies (top n)
# Pulp Fiction followed by Forrest Gump are the movies with the most ratings
df %>%
  count(movieId, title, name = "n_ratings", sort = TRUE) %>%
  head(10L)

# Movies with only one rating
# 126 of them, too many to list
df %>%
  count(movieId, title) %>%
  filter(n == 1) %>%
  nrow()

# Movies in the bottom 25th percentile in number of ratings
# 2,688 movies with 30 ratings or less
df %>%
  count(movieId) %>%
  filter(n <= quantile(n, .25)) %>%
  nrow()

# Frequency plot of ratings per movie
# We find a strong relationship: movies with few ratings are far more present
df %>%
  count(movieId, name = "n_ratings") %>%
  count(n_ratings, name = "frequency") %>%
  ggplot(mapping = aes(x = n_ratings, y = frequency)) +
  geom_point() +
  scale_x_log10() +
  theme_light()

# Maybe do a Pareto here?

# ==============================================================================
# 2.4 Ratings score per movie
# ==============================================================================

# Highest-rated movies (top n)
# We find that highest-rated movies also have very few (1–4) ratings each
df %>%
  group_by(movieId, title) %>%
  summarize(mean_rating = mean(rating), n_rating = n()) %>%
  arrange(desc(mean_rating)) %>%
  head(10L)

# Same, but now excluding the 25th bottom percentile in # of ratings
# This returns a totally different set (zero overlap)
df %>%
  group_by(movieId, title) %>%
  summarize(mean_rating = mean(rating), n_rating = n()) %>%
  arrange(desc(mean_rating)) %>%
  filter(n_rating > 30L) %>%
  head(10L)

# Lowest-rated movies (bottom n)
# We find that lowest-rated movies also have very few (1–199) ratings each
df %>%
  group_by(movieId, title) %>%
  summarize(mean_rating = mean(rating), n_rating = n()) %>%
  arrange(desc(mean_rating)) %>%
  tail(10L)

# Same, but now excluding the 25th bottom percentile in # of ratings
# This returns a fairly different set (only three movies overlap)
df %>%
  group_by(movieId, title) %>%
  summarize(mean_rating = mean(rating), n_rating = n()) %>%
  arrange(desc(mean_rating)) %>%
  filter(n_rating > 30L) %>%
  tail(10L)

# Plot the relationship between rating count and rating score
# We find that movies with few ratings have a much wider rating spread
df %>%
  group_by(movieId, title) %>%
  summarize(mean_rating = mean(rating), n_rating = n()) %>%
  ggplot(mapping = aes(x = n_rating, y = mean_rating)) +
  geom_point() +
  theme_light()

# ==============================================================================
# 2.5 Ratings by year of movie release
# ==============================================================================

# Ratings by year of movie release (table)
# We find that ratings are more severe for movies released after 1977
# Median rating is 4.0 for pre-1977 movies, and 3.5 for post-1977 movies
# Is there some nostalgia at play, whereby older movies are given more leeway?
# Or are older movies objectively better than more modern ones?df %>%
df %>%
  group_by(year_movie) %>%
  summarize(
    mean_rating   = mean(rating),
    median_rating = median(rating),
    n_rating      = n()
  )

# Ratings by year of movie release (plot)
df %>%
  group_by(year_movie) %>%
  summarize(
    mean   = mean(rating),
    median = median(rating),
    low    = quantile(rating, .25), # 25th percentile of the ratings for that year
    high   = quantile(rating, .75)  # 75th percentile of the ratings for that year
  ) %>%
  ggplot(aes(x = year_movie)) +
  geom_ribbon(
    mapping = aes(y = mean, ymin = low, ymax = high),
    fill    = "grey70"
  ) +
  geom_line(mapping = aes(y = mean)) +
  geom_point(mapping = aes(y = median)) +
  theme_light() +
  xlab("Year of movie release") +
  ylab("Ratings")

# ==============================================================================
# 2.6 Ratings by year of movie release
# ==============================================================================

# Ratings by year of rating (table)
# We find that more recent reviews (> 2022) are more severe than older reviews
df %>%
  group_by(year_rating) %>%
  summarize(
    mean   = mean(rating),
    median = median(rating),
    n_rating = n()
  )

# Ratings by year of rating (plot)
df %>%
  group_by(year_rating) %>%
  summarize(
    mean   = mean(rating),
    median = median(rating),
    low    = quantile(rating, .25), # 25th percentile of the ratings for that year
    high   = quantile(rating, .75)  # 75th percentile of the ratings for that year
  ) %>%
  ggplot(aes(x = year_rating)) +
  geom_ribbon(
    mapping = aes(y = mean, ymin = low, ymax = high),
    fill    = "grey70"
  ) +
  geom_line(mapping = aes(y = mean)) +
  geom_point(mapping = aes(y = median)) +
  theme_light() +
  xlab("Year of rating") +
  ylab("Ratings")

# ==============================================================================
# 2.7 Ratings by years of distance between movie release and rating
# ==============================================================================

# Ratings by years of distance between movie release and rating (table)
# Movies reviewed <19 or >88 years after their release are judged more severely
df %>%
  group_by(year_dist) %>%
  summarize(
    mean   = mean(rating),
    median = median(rating),
    n_rating = n()
  )

# Ratings by years of distance between movie release and rating (plot)
df %>%
  group_by(year_dist) %>%
  summarize(
    mean   = mean(rating),
    median = median(rating),
    low    = quantile(rating, .25), # 25th percentile of the ratings for that year
    high   = quantile(rating, .75)  # 75th percentile of the ratings for that year
  ) %>%
  ggplot(aes(x = year_dist)) +
  geom_ribbon(
    mapping = aes(y = mean, ymin = low, ymax = high),
    fill    = "grey70"
  ) +
  geom_line(mapping = aes(y = mean)) +
  geom_point(mapping = aes(y = median)) +
  theme_light() +
  xlab("Years of distance between movie release and rating") +
  ylab("Ratings")

# ==============================================================================
# 2.8 Ratings by genre
# ==============================================================================

# Ratings by genre
# Film-noir is the highest-rated (mean = 3.9), horror the worst-rated (2.85)
# Is it a bias in reviewer's preferences, or are many horror movies just bad?
df %>%
  distinct(title, .keep_all = TRUE) %>%
  separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(
    mean   = mean(rating),
    median = median(rating),
    n_rating = n()
  ) %>%
  arrange(desc(mean))

# ==============================================================================
# 2.9 Ratings by reviewer
# ==============================================================================

# Ratings by reviewer (table)
# We find that the more ratings a reviewer has given, the less spread out
df %>%
  group_by(userId) %>%
  summarize(
    mean_rating   = mean(rating),
    median_rating = median(rating),
    n_rating      = n()
  )

# Ratings by reviewer (plot)
df %>%
  group_by(userId) %>%
  summarize(
    mean_rating   = mean(rating),
    median_rating = median(rating),
    n_rating      = n()
  ) %>%
  ggplot(mapping = aes(x = n_rating, y = mean_rating)) +
  geom_point() +
  theme_light()

# ==============================================================================
# 3 Predictions
# ==============================================================================

# ==============================================================================
# 3.1 Basic prediction
# This just takes the mean of all ratings without accounting for any correlation
# ==============================================================================

# Calculate the root mean square error of predicting ratings with just the mean
# We use the standard RMSE function from the caret package
# RMSE = 1.0612018
RMSE(mean(df$rating), final_holdout_test$rating)




# ==============================================================================
# X Housekeeping
# ==============================================================================

# Stop the script timer
Sys.time() - start_time

# EOF