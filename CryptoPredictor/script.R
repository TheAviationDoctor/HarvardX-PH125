# ==============================================================================
#    NAME: script.R
#   INPUT: Ethereum historical prices
# ACTIONS: 1 Prepare the data
#          2 Explore the data
#          3 Create prediction model
#  OUTPUT: A predictive model of Ethereum prices
# RUNTIME: ~1 hour
#  AUTHOR: Thomas D. Pellegrin <thomas@pellegr.in>
#    YEAR: 2023
# ==============================================================================

# ==============================================================================
# 0 Housekeeping
# ==============================================================================

# Clear the environment
rm(list = ls())

# # Load the required libraries
library(caret)
library(dplyr)
library(gbm)
library(ggplot2)
library(randomForest)
library(readr)
library(TTR)

# Start a script timer
start_time <- Sys.time()

# Create a list to hold the resulting RMSEs
rmses <- list()

# Clear the console
cat("\014")

# ==============================================================================
# 1 Prepare the data
# ==============================================================================

# Load the data
# Downloaded from https://coinmarketcap.com/currencies/ethereum/historical-data/
eth <- read_delim(
    file       = "eth.csv",
    delim      = ";",
    col_types  = c("TTTTnnnnnnT"),
    col_select = c(1, 8:9)
  ) |>
  rename(price = close) |>
  mutate(
    timeOpen  = as.Date(timeOpen),
    SMA7  = SMA(price, n =  7),  # Simple Moving Average
    EMA7  = EMA(price, n =  7),  # Exponential Moving Average
    RSI7  = RSI(price, n =  4),  # Relative Strength Index
    SMA14 = SMA(price, n = 14),  # Simple Moving Average
    EMA14 = EMA(price, n = 14),  # Exponential Moving Average
    RSI14 = RSI(price, n = 14),  # Relative Strength Index
    SMA21 = SMA(price, n = 21),  # Simple Moving Average
    EMA21 = EMA(price, n = 21),  # Exponential Moving Average
    RSI21 = RSI(price, n = 21)   # Relative Strength Index
  ) |>
  na.omit(eth) |> # Remove NAs
  mutate(index = as.numeric(timeOpen - min(timeOpen)), timeOpen = NULL) # Index

# ==============================================================================
# 2 Partition the data
# ==============================================================================

# Set a seed for reproducibility
suppressWarnings(set.seed(1, sample.kind = "Rounding"))

# Partition the data. Test set will be 10% of data
index <- createDataPartition(
  y     = eth$index,
  times = 1,
  p     = 0.1,
  list  = FALSE
)

# Build the training and test data sets
eth_train <- eth[-index, ]
eth_test  <- eth[ index, ]

# ==============================================================================
# 3 Build the models
# ==============================================================================

# Training control
trControl <- trainControl(
  method        = "cv",
  number        = 10,
  p             = .5,
  search        = "grid",
  allowParallel = TRUE,
  verboseIter   = TRUE
)

# Tuning grid for the glmnet model
tuneGrid_glmnet <- expand.grid(
  .alpha  = seq(0, 1,     length.out = 100),
  .lambda = seq(0.001, 5, length.out = 100)
)

# Tuning grid for the gbm model
tuneGrid_gbm <- expand.grid(
  interaction.depth = c(1, 3, 5),        # Max depth of trees
  n.trees           = c(50, 100, 150),   # Number of trees
  shrinkage         = c(0.01, 0.1, 0.3), # Learning rate
  n.minobsinnode    = c(10, 20)          # Minimum number of observations in the trees' terminal nodes
)

# Train the models
train <- list(
  gbm    = train(price ~ ., data = eth_train, method = "gbm",    trControl = trControl, tuneGrid = tuneGrid_gbm),
  glmnet = train(price ~ ., data = eth_train, method = "glmnet", trControl = trControl, tuneGrid = tuneGrid_glmnet),
  lm     = train(price ~ ., data = eth_train, method = "lm",     trControl = trControl),
  rf     = train(price ~ ., data = eth_train, method = "rf",     trControl = trControl, tuneLength = 50L)
)

# ==============================================================================
# 4 Create the predictions
# ==============================================================================

# Predict prices based on test data predictors
eth_pred <- lapply(
  X   = names(train),
  FUN = function(x) predict(train[x], newdata = eth_test)
) |>
  as.data.frame() |>
  # In case a model predicts negative prices, set the floor to one cent
  mutate_all(list(~ replace(., . < 0, 0.01))) |>
  # Rowwise mean of all models
  mutate(mean = rowMeans(across(everything()))) |>
  # Add the index for plotting
  mutate(index = eth_test$index)

# Display the results
res <- lapply(X = eth_pred |> select(-index), FUN = function(x) postResample(obs = x, eth_test$price))

# Name of the model with the lowest RMSE
win <- res |>
  as.data.frame() |>
  t() |>
  as.data.frame() |>
  filter(RMSE == min(RMSE)) |>
  rownames()

# Plot the results
ggplot() +
  geom_line( data = eth_train, aes(x = index, y = price),    color = "gray") +
  geom_point(data = eth_test,  aes(x = index, y = price),    color = "red",  alpha = .25) +
  geom_point(data = eth_pred,  aes(x = index, y = get(win)), color = "blue", alpha = .25) +
  labs(x = "Time", y = "Price in USD")

# ==============================================================================
# 5 Housekeeping
# ==============================================================================

# Stop the script timer
Sys.time() - start_time

# EOF