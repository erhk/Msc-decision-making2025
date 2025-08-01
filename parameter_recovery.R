# Simulate choice data from known parameters
set.seed(1990)
n_sim_subj <- 100

# Simulate parameter sets
sim_params <- tibble(
  SubjectID = 1:n_sim_subj,
  alpha = runif(n_sim_subj, 0.01, 0.99),
  w     = runif(n_sim_subj, 0.1, 2.0),
  A     = runif(n_sim_subj, 0.1, 1.0),
  c     = runif(n_sim_subj, 0.1, 10.0)
)

# Simulate choice data from those parameters
sim_data <- sim_params %>%
  pmap_dfr(function(SubjectID, alpha, w, A, c) {
    wins   <- rnorm(95, mean = 30, sd = 15)
    losses <- rnorm(95, mean = -20, sd = 15)
    choices <- simulate_pvl_choices(alpha, w, A, c, wins, losses, 95)
    
    tibble(
      SubjectID = SubjectID,
      Trial = 1:95,
      Choice = choices,
      Win = wins,
      Loss = losses
    )
  })

# Save simulated data to csv
write_csv(sim_data, "data/simulated_IGT95_dataset.csv")

# Load simulated data (optional, if reloading from file)
sim_data <- read_csv("data/simulated_IGT95_dataset.csv")

# Create the required Stan data structure
stan_data_sim <- list(
  N = length(unique(sim_data$SubjectID)),
  T = sim_data %>% group_by(SubjectID) %>% summarise(n = n()) %>% pull(n),
  T_total = nrow(sim_data),
  subj = as.integer(factor(sim_data$SubjectID)),
  Choice = sim_data$Choice,
  Win = sim_data$Win,
  Loss = sim_data$Loss
)

# Load pvl model
model_sim <- cmdstan_model("models/fixed_PVL Hierarchical model.stan", cpp_options = list(stan_threads = TRUE))


fit_sim <- model_sim$sample(
  data = stan_data_sim,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2,
  iter_warmup = 1000,
  iter_sampling = 1000,
  seed = 1990,
  refresh = 10,
  max_treedepth = 20,
  adapt_delta = 0.99
)



# Assuming you get `draws_recovery` after fitting

true_vs_estimated <- tibble(
  SubjectID = 1:n_sim_subj,
  alpha_true = sim_params$alpha,
  alpha_est  = sapply(1:n_sim_subj, function(i) mean(draws_recovery[[paste0("alpha[", i, "]")]])),
  w_true = sim_params$w,
  w_est  = sapply(1:n_sim_subj, function(i) mean(draws_recovery[[paste0("w[", i, "]")]])),
  A_true = sim_params$A,
  A_est  = sapply(1:n_sim_subj, function(i) mean(draws_recovery[[paste0("A[", i, "]")]])),
  c_true = sim_params$c,
  c_est  = sapply(1:n_sim_subj, function(i) mean(draws_recovery[[paste0("c[", i, "]")]]))
)

# Plot recovery for each parameter
true_vs_estimated %>%
  pivot_longer(-SubjectID) %>%
  separate(name, into = c("param", "type")) %>%
  pivot_wider(names_from = type, values_from = value) %>%
  ggplot(aes(x = true, y = est)) +
  geom_point(alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  facet_wrap(~param, scales = "free") +
  labs(title = "Parameter Recovery", x = "True Value", y = "Estimated Value") +
  theme_minimal()
