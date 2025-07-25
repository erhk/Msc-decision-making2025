### Main
### Decision Making - Explotation vs. exploration in IGT ###

pacman::p_load(tidyverse, cmdstanr, posterior, bayesplot, ggplot2, tidyr, dplyr,purrr)

# Load pvl model
model <- cmdstan_model("models/PVL_hierach.stan", cpp_options = list(stan_threads = TRUE))

# Load empirical data
igt_all_with_wins <- read_csv("data/Final_IGT_Dataset_with_Wins_and_Running_Total.csv")

# subset to different conditions (studies)
igt_95  <- igt_all_with_wins %>% filter(Condition == "IGT_95")
#igt_100 <- igt_all_with_wins %>% filter(Condition == "IGT_100")
igt_150 <- igt_all_with_wins %>% filter(Condition == "IGT_150")

# Prepare and fit data to model - function
prepare_and_fit <- function(data, model, seed = 1990) {
  # Assign subject index
  data <- data %>%
    mutate(SubjectCondID = paste0("S", SubjectID, "_", Condition)) %>%
    group_by(SubjectCondID) %>%
    mutate(subj_index = cur_group_id()) %>%
    ungroup()
  
  # Create required components
  N <- n_distinct(data$subj_index)
  T_per_subj <- data %>%
    count(subj_index) %>%
    arrange(subj_index) %>%
    pull(n)
  
  stan_data <- list(
    N = N,
    T = T_per_subj,
    T_total = nrow(data),
    subj = data$subj_index,
    Choice = data$Choice,
    Win = data$Win,
    Loss = data$Loss
  )
  
  # Fit model
  fit <- model$sample(
    data = stan_data,
    chains = 4,
    parallel_chains = 4,
    threads_per_chain = 2,
    iter_warmup = 1000,
    iter_sampling = 1000,
    seed = seed,
    refresh = 10,
    max_treedepth = 20,
    adapt_delta = 0.99
    
  )
  
  return(list(fit = fit, lookup = data %>% distinct(subj_index, SubjectCondID)))
}

# Fit model
fit_95_new  <- prepare_and_fit(igt_95, model)

# Save the full fit
#fit_150$fit$save_object("fit_150_cmdstanr.stanfit")
#fit_95$fit$save_object("fit_95_cmdstanr.stanfit")

# Save only RDS for local macine
#saveRDS(fit_95, file = "fit_95.rds")
#saveRDS(fit_150, file = "fit_150.rds")







