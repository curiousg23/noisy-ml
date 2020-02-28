# Explore a simple noisy case: univariate linear relationship with a large
# amount of Gaussian noise. Examine the performance of an OLS vs. the oracle
# function. The OLS should not perform better than the oracle function on
# average.

library(tidyverse)
library(ggplot2)

set.seed(0)

N <- 500

# Univariate toy model: f(X) = 5x. Add Gaussian noise so the R^2 is roughly 0.5%.
X <- runif(N, min = -100, max = 100)
fX <- 5 * X
mat <- cbind(X, fX)
colnames(mat) <- c("X", "fX")
tbl <- as_tibble(mat)

# For R^2 of roughly 0.5%.
e_sd <- sqrt((0.995 / 0.005) * var(tbl$fX))
tbl <- mutate(tbl, e = e_sd * rnorm(N), Y = fX + e)
tbl <- mutate(tbl, e_cauchy = e_sd * rcauchy(N), Y_cauchy = fX + e_cauchy)

model <- lm(Y ~ X, data = tbl)
model_cauchy <- lm(Y_cauchy ~ X, data = tbl)

ggplot(data = tbl) + geom_point(mapping = aes(x = X, y = Y)) +
  geom_line(mapping = aes(x = X, y = predict(model, tbl)), color = "blue") +
  geom_line(mapping = aes(x = X, y = predict(model_cauchy, tbl)), color = "green") +
  geom_line(mapping = aes(x = X, y = fX, color = "red"))

# Generate a random sample, hold 20% out for testing, and fit the model on the
# other 80%. We want to see that our oracle (f(X) = 5x) has the smallest error.
# We can replicate this multiple times to build a distribution of what the error
# looks like.
simulate_ols <- function() {
  idx <- seq(1, 500)
  X <- runif(N, min = -100, max = 100)
  fX <- 5 * X
  mat <- cbind(idx, X, fX)
  colnames(mat) <- c("idx", "X", "fX")
  tbl <- as_tibble(mat)
  tbl$idx <- as.integer(tbl$idx)
  
  # Start with normal Gaussian noise, with R^2 of 0.5%. We should not be seeing
  # results better than the oracle, on average.
  e_sd <- sqrt((0.995 / 0.005) * var(tbl$fX))
  tbl <- mutate(tbl, e = rnorm(N, mean = 0, sd = e_sd), Y = fX + e)
  
  train <- filter(tbl, idx <= 400)
  test <- filter(tbl, idx >= 401)
  
  model <- lm(Y ~ X, data = train)
  
  model_loss <- mean((predict(model, test) - test$Y)^2)
  oracle_loss <- mean((5 * test$X - test$Y)^2)
  rob_model_loss <- mean(abs(predict(model, test) - test$Y))
  rob_oracle_loss <- mean(abs(5 * test$X - test$Y))
  mat <- cbind(oracle_loss, model_loss, rob_oracle_loss, rob_model_loss)
  colnames(mat) <- c("oracle_mse", "model_mse", "oracle_mae", "model_mae")
  mse_tbl <- as_tibble(mat)
  return(mse_tbl)
}

sim_results <- replicate(1000, simulate_ols(), simplify = FALSE)
sim_results_tbl <- bind_rows(sim_results)
sim_results_long <- melt(sim_results_tbl) %>% as_tibble()
colnames(sim_results_long) <- c("type", "mse")

means <- sim_results_long %>% group_by(type) %>% summarise(mu = mean(mse))

# In the plot, we should see that the mean MSE for the oracle is less than that
# for the model. In general this plot shows that--the mean model MSE is close
# to, but still larger than the oracle_mse. The estimated density shows that the
# oracle density has slightly more mass towards the left than the model.
ggplot(data = filter(sim_results_long, type == "oracle_mse" | type == "model_mse")) +
  geom_histogram(mapping = aes(x = mse, y = ..density.., color = type, fill = type, alpha = 0.5)) +
  geom_vline(data = filter(means, type == "oracle_mse" | type == "model_mse"), mapping = aes(xintercept = mu, color = type)) +
  geom_density(mapping = aes(x = mse, color = type))

ggplot(data = filter(sim_results_long, type == "oracle_mae" | type == "model_mae")) +
  geom_histogram(mapping = aes(x = mse, y = ..density.., color = type, fill = type, alpha = 0.5)) +
  geom_vline(data = filter(means, type == "oracle_mae" | type == "model_mae"), mapping = aes(xintercept = mu, color = type)) +
  geom_density(mapping = aes(x = mse, color = type))
