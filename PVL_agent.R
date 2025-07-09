### Decision Making - Explotation vs. exploration in IGT ###

pacman::p_load(tidyverse, cmdstanr, posterior, bayesplot, ggplot2, tidyr, dplyr,purr)

# Load pvl model
model <- cmdstan_model("PVL_model.stan")

# Load empirical data
igt_all_with_wins <- read_csv("data/Final_IGT_Dataset_with_Wins_and_Running_Total.csv", sep = ",")

#### Create PVL agents for simulated data for IGT ####

# STEP 1: Prepare outcome pools by deck
prepare_deck_outcomes <- function(df) {
  df %>%
    select(Deck = Choice, Win, Loss) %>%
    group_by(Deck) %>%
    group_split()
}

# STEP 2: Define PVL-Delta agent simulator
simulate_pvl_agents <- function(n_agents = 100, n_trials = 95, deck_outcomes) {
  
  all_trials <- list()
  true_params <- list()
  
  for (agent_id in 1:n_agents) {
    
    # Sample true agent parameters
    alpha <- runif(1, 0.05, 0.95) # learning rate
    w <- runif(1, 0.1, 2.0) # loss aversion
    A <- runif(1, 0.1, 1) # utility curvature (small for numerical stability)
    c <- runif(1, 0.5, 5.0) # choice sensitivity
    
    # Initialize value estimates
    V <- rep(0, 4)
    
    agent_trials <- tibble(
      AgentID = agent_id,
      Trial = integer(n_trials),
      Choice = integer(n_trials),
      Win = numeric(n_trials),
      Loss = numeric(n_trials),
      alpha_true = alpha,
      w_true = w,
      A_true = A,
      c_true = c
    )
    
    for (t in 1:n_trials) {
      # Softmax decision
      maxV <- max(c * V)
      probs <- exp(c * V - maxV) / sum(exp(c * V - maxV))
      choice <- sample(1:4, size = 1, prob = probs)
      
      # Sample empirical outcome for chosen deck
      outcome_pool <- deck_outcomes[[choice]]
      sampled <- outcome_pool[sample(nrow(outcome_pool), 1), ]
      
      # Subjective utility
      reward <- sampled$Win + sampled$Loss
      abs_reward <- max(abs(reward), 1e-6)
      u <- ifelse(reward < 0,
                  -w * abs_reward^A,
                  abs_reward^A)
      
      # Update value
      V[choice] <- V[choice] + alpha * (u - V[choice])
      
      # Store trial
      agent_trials$Trial[t] <- t
      agent_trials$Choice[t] <- choice
      agent_trials$Win[t] <- sampled$Win
      agent_trials$Loss[t] <- sampled$Loss
    }
    
    all_trials[[agent_id]] <- agent_trials
    
    true_params[[agent_id]] <- tibble(
      AgentID = agent_id,
      alpha_true = alpha,
      w_true = w,
      A_true = A,
      c_true = c
    )
  }
  
  sim_data <- bind_rows(all_trials)
  true_param_df <- bind_rows(true_params)
  
  return(list(trial_data = sim_data, true_params = true_param_df))
}


#write.csv(sim_data, "data/sim_data.csv", row.names = F)
sim_result <- simulate_pvl_agents(n_agents = 50, n_trials = 150, deck_outcomes = deck_outcomes)

sim_data <- sim_result$trial_data # trial-level data for Stan fitting
true_params <- sim_result$true_params


# Get list of agent IDs
agent_ids <- unique(sim_data$AgentID)

# Store estimated parameter summaries
fit_results <- vector("list", length(agent_ids))

for (i in seq_along(agent_ids)) {
  agent_id <- agent_ids[i]
  agent_data <- sim_data %>%
    filter(AgentID == agent_id) %>%
    arrange(Trial)
  
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
    parallel_chains = 2
  )
  
  # Extract posterior means
  summary <- fit$summary(variables = c("alpha", "w", "A", "c")) %>%
    select(variable, mean) %>%
    pivot_wider(names_from = variable, values_from = mean) %>%
    mutate(AgentID = agent_id)
  
  fit_results[[i]] <- summary
}

estimates_df <- bind_rows(fit_results)
recovery_df <- left_join(estimates_df, true_params, by = "AgentID")


# Estimated values (one per parameter, per agent)
est_long <- recovery_df %>%
  select(AgentID, alpha, w, A, c) %>%
  pivot_longer(cols = -AgentID, names_to = "param", values_to = "estimated")

# True values, matching names to above
true_long <- recovery_df %>%
  select(AgentID, alpha_true, w_true, A_true, c_true) %>%
  rename_with(~ gsub("_true", "", .x)) %>%
  pivot_longer(cols = -AgentID, names_to = "param", values_to = "true")

# Join estimated + true on AgentID and param
recovery_long <- left_join(est_long, true_long, by = c("AgentID", "param"))

# Plot
ggplot(recovery_long, aes(x = true, y = estimated)) +
  geom_point(alpha = 0.8) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  facet_wrap(~ param, scales = "free") +
  theme_minimal() +
  labs(
    title = "Parameter Recovery: Estimated vs. True",
    x = "True Value",
    y = "Estimated Value"
  )

recovery_df %>%
  summarise(
    across(c(alpha, w, A, c), list(min = min, max = max, mean = mean, sd = sd))
  )

cor(recovery_df$w, recovery_df$w_true)
cor(recovery_df$A, recovery_df$A_true)
# All 4 parameters together
draws <- as_draws_df(fit$draws())

mcmc_dens(draws, pars = c("alpha", "w", "A", "c"))

true_w <- unique(agent_data$w_true)

ggplot(draws, aes(x = w)) +
  geom_density(fill = "steelblue", alpha = 0.6) +
  geom_vline(xintercept = true_w, color = "red", linetype = "dashed") +
  labs(title = "Posterior for w (vs. true value)")

posterior_widths <- recovery_df %>%
  mutate(w_error = abs(w - w_true))

ggplot(posterior_widths, aes(x = w_true, y = w_error)) +
  geom_point() +
  geom_smooth() +
  labs(title = "Absolute Error of w Estimates", y = "|Estimated - True|", x = "True w")


