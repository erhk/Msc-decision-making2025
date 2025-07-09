### Main
### Decision Making - Explotation vs. exploration in IGT ###

pacman::p_load(tidyverse, cmdstanr, posterior, bayesplot, ggplot2, tidyr, dplyr,purrr)

# Load pvl model
model <- cmdstan_model("PVL_model.stan")

# Load empirical data
igt_all_with_wins <- read_csv("data/Final_IGT_Dataset_with_Wins_and_Running_Total.csv", sep = ",")

# subset to condiion 95
df_95 <- 

stan_data <- list(
  T = nrow(agent_data),
  Choice = agent_data$Choice,
  Win = agent_data$Win,
  Loss = agent_data$Loss
)

# Fit model
fit <- model$sample(
  data = stan_data,
  chains = 2,
  iter_warmup = 500,
  iter_sampling = 500,
  refresh = 0,
  parallel_chains = 2,
  max_treedepth = 20,
  adapt_delta= 0.99
)