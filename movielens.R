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

# Clear the environment
# rm(list = ls())

# Install the required libraries if they are missing
repo <- "http://cran.us.r-project.org"
if(!require(tidyverse)) { install.packages("tidyverse", repos = repo) }
if(!require(caret))     { install.packages("caret",     repos = repo) }

# Load the required libraries
library(caret)
library(tidyverse)

# Set a 2-minute timeout for attempting to download datasets
options(timeout = 120L)

# Start a script timer
start_time <- Sys.time()

# Clear the console
cat("\014")

# ==============================================================================
# 1 Create the edx dataset if necessary (adapted from course-provided code)
# ==============================================================================

# Run only if the edx dataset is not yet loaded into the R environment
if(!exists("edx")) {
  
  # Download the zipped MovieLens dataset (had to downgrade to HTTP to make it work on Windows)
  dl <- "ml-10M100K.zip"
  if(!file.exists(dl)) { download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl) }
  
  # Create the data folder if missing (necessary on Windows)
  dir.create("ml-10M100K", showWarnings = FALSE)
  
  # Unzip the ratings file
  ratings_file <- "ml-10M100K/ratings.dat"
  if(!file.exists(ratings_file)) { unzip(dl, ratings_file) }
  
  # Unzip the movies file
  movies_file <- "ml-10M100K/movies.dat"
  if(!file.exists(movies_file)) { unzip(dl, movies_file) }
  
  # Load ratings into a data frame
  ratings <- as.data.frame(
    str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
    stringsAsFactors = FALSE
  )
  
  # Set the ratings column names
  colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
  
  # Change ratings column types
  ratings <- ratings %>%
    mutate(
      userId = as.integer(userId),
      movieId = as.integer(movieId),
      rating = as.numeric(rating),
      timestamp = as.integer(timestamp)
    )
  
  # Load movies into a data frame
  movies <- as.data.frame(
    str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
    stringsAsFactors = FALSE
  )
  
  # Set the movies column names
  colnames(movies) <- c("movieId", "title", "genres")
  
  # Change movies column types
  movies <- movies %>%
    mutate(movieId = as.integer(movieId))
  
  # Join the movies and ratings datasets
  movielens <- left_join(ratings, movies, by = "movieId")
  
  # Set a seed for reproducibility
  set.seed(1, sample.kind = "Rounding") # if using R 3.6 or later
  
  # Partition the data. Final hold-out test set will be 10% of MovieLens data
  test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
  
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
  
}

# ==============================================================================
# 2 Exploratory analysis to identify what might be useful predictors of ratings
# ==============================================================================

# ==============================================================================
# 2.1 Transform the data
# ==============================================================================

# Display first 5 rows before transformations
head(edx, 10L)

# Describe the structure of the data
str(edx)

# Copy the edx dataset to a df dataset so we can transform the data at will.
# Year of the movie and of the rating, and their distance, might be predictors,
# so extract them, remove year from titles, & remove no longer needed timestamps
df <- edx %>%
  mutate(
    year_movie  = as.integer(str_sub(title, start = -5L, end = -2L)),
    year_rating = as.integer(as.POSIXlt(timestamp, origin = "1970-01-01")$year + 1900L),
    year_dist   = as.integer(abs(year_movie - year_rating)),
    title       = str_sub(title, end = -8L),
    timestamp   = NULL
  )

# ==============================================================================
# 2.1 Number of review per movie
# ==============================================================================

# Ratings may be skewed if there's a very low number of them for a given movie
# So let's determine the number of reviews per movie
df_reviews <- df %>%
  count(title) %>%
  arrange(desc(n))

# Display the statistics
summary(df_reviews)

# Top ten movies by number of reviews
df_reviews %>% head(10L)

# Bottom ten movies by number of reviews
df_reviews %>% tail(10L)

# ==============================================================================
# 2.2 Ratings per movie
# ==============================================================================

# Top ten movies by rating
# We only consider movies above the 25th percentile of ratings count

quantile(df_reviews$n, .25)

df %>%
  group_by(title) %>%
  summarize(mean_rating = mean(rating)) %>%
  arrange(desc(mean_rating)) %>%
  head(10L)





# List unique genres
df %>%
  distinct(title, .keep_all = TRUE) %>%
  separate_rows(genres, sep = "\\|") %>%
  count(genres)
  # distinct(genres)
  # group_by(genres) %>%
  # summarize(
  #   # genres = as.factor(genres),
  #   count  = n()
  # )
genres

# Display first 5 rows after transformations
# head(df, 5L)

# ==============================================================================
# 3 Visualize relationships in the data
# ==============================================================================

# # Ratings by year of movie release
# df %>%
#   group_by(year_movie) %>%
#   summarize(
#     mean   = mean(rating),
#     median = median(rating),
#     low    = quantile(rating, .25), # 25th percentile of the ratings for that year
#     high   = quantile(rating, .75)  # 75th percentile of the ratings for that year
#   ) %>%
#   ggplot(aes(x = year_movie)) +
#     geom_ribbon(
#       mapping = aes(y = mean, ymin = low, ymax = high),
#       fill    = "grey70"
#     ) +
#     geom_line(mapping = aes(y = mean)) +
#     geom_point(mapping = aes(y = median)) +
#     theme_light() +
#     xlab("Year of release") +
#     ylab("Ratings")

# # Ratings by year of rating
# df %>%
#   group_by(year_rating) %>%
#   summarize(
#     mean   = mean(rating),
#     median = median(rating),
#     low    = quantile(rating, .25), # 25th percentile of the ratings for that year
#     high   = quantile(rating, .75)  # 75th percentile of the ratings for that year
#   ) %>%
#   ggplot(aes(x = year_rating)) +
#   geom_ribbon(
#     mapping = aes(y = mean, ymin = low, ymax = high),
#     fill    = "grey70"
#   ) +
#   geom_line(mapping = aes(y = mean)) +
#   geom_point(mapping = aes(y = median)) +
#   theme_light() +
#   xlab("Year of rating") +
#   ylab("Ratings")

# # Ratings by years of distance between movie release and rating
# df %>%
#   group_by(year_dist) %>%
#   summarize(
#     mean   = mean(rating),
#     median = median(rating),
#     low    = quantile(rating, .25), # 25th percentile of the ratings for that year
#     high   = quantile(rating, .75)  # 75th percentile of the ratings for that year
#   ) %>%
#   ggplot(aes(x = year_dist)) +
#   geom_ribbon(
#     mapping = aes(y = mean, ymin = low, ymax = high),
#     fill    = "grey70"
#   ) +
#   geom_line(mapping = aes(y = mean)) +
#   geom_point(mapping = aes(y = median)) +
#   theme_light() +
#   xlab("Years of distance between movie release and rating") +
#   ylab("Ratings")

# Ratings by genre

# ==============================================================================
# X Housekeeping
# ==============================================================================

# Stop the script timer
Sys.time() - start_time

# EOF