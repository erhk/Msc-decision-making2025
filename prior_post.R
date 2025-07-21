# PRIOR PREDICTIVE CHECK: Simulate behavioral metrics from priors
#### Load everything ####
#Load library
pacman::p_load(tidyverse, cmdstanr, posterior, bayesplot, ggplot2, tidyr, dplyr,purrr, GGally)

# Load empirical data
igt_all_with_wins <- read_csv("data/Final_IGT_Dataset.csv")

# subset to different conditions (studies). Not using igt_100, fitting the model takes too long
igt_95  <- igt_all_with_wins %>% filter(Condition == "IGT_95")
igt_150 <- igt_all_with_wins %>% filter(Condition == "IGT_150")

# Loads fits
fit_95 <- readRDS("fits/fit_95_cmdstanr.stanfit")
fit_150 <- readRDS("fits/fit_150_cmdstanr.stanfit")

# Extract draws
draws_95 <- as_draws_df(fit_95$draws())
draws_150 <- as_draws_df(fit_150$draws())


# PRIOR PREDICTIVE CHECK: Simulate behavioral metrics from priors

# Number of simulated participants
n_subj <- 100
n_trials <- 95  # Use 150 for IGT 150 if needed

set.seed(1990)

simulate_from_prior <- function() {
  # Sample group-level parameters from their priors
  mu_alpha <- rnorm(1, 0.5, 0.2)
  mu_w     <- rnorm(1, 1.0, 0.5)
  mu_A     <- rbeta(1, 2, 2)
  mu_c     <- rnorm(1, 2.0, 1.0)
  
    
  sigma_alpha <- rexp(1, 1)
  sigma_w     <- rexp(1, 1)
  sigma_A     <- rexp(1, 1)
  sigma_c     <- rexp(1, 1)
  
  
  metrics <- tibble(switch_rate = numeric(n_subj), entropy = numeric(n_subj), max_streak = numeric(n_subj))
  
  for (i in 1:n_subj) {
    alpha <- plogis(mu_alpha + sigma_alpha * rnorm(1)) * 0.98 + 0.01
    w     <- plogis(mu_w + sigma_w * rnorm(1)) * 1.9 + 0.1
    A     <- plogis(mu_A + sigma_A * rnorm(1)) * 0.9 + 0.1
    c     <- plogis(mu_c + sigma_c * rnorm(1)) * 9.9 + 0.1
    
    # Simulate rewards (realistic placeholder: NetWin ~ N(0, 50))
    wins   <- rnorm(n_trials, mean = 30, sd = 15)
    losses <- rnorm(n_trials, mean = -20, sd = 15)
    
    sim <- simulate_pvl_choices(alpha, w, A, c, wins, losses, n_trials)
    
    # Compute behavioral metrics
    n_switches <- sum(sim != dplyr::lag(sim, default = sim[1]))
    p_decks <- prop.table(table(factor(sim, levels = 1:4)))
    streaks <- rle(sim)$lengths
    
    metrics$switch_rate[i] <- n_switches / (n_trials - 1)
    metrics$entropy[i]     <- -sum(p_decks * log2(p_decks + 1e-8))
    metrics$max_streak[i]  <- max(streaks)
  }
  
  return(metrics)
}

prior_metrics <- simulate_from_prior()

# Visualize prior predictive behavioral distributions
prior_metrics_long <- pivot_longer(prior_metrics, cols = everything(), names_to = "Metric", values_to = "Value")

ggplot(prior_metrics_long, aes(x = Value)) +
  geom_density(fill = "skyblue", alpha = 0.7) +
  facet_wrap(~ Metric, scales = "free") +
  labs(title = "Adjusted prior: Prior Predictive Distributions of Behavioral Metrics") +
  theme_minimal()


# POSTERIOR PREDICTIVE CHECK: Simulate from posterior draws and compare

# Ensure subjects_95 is defined
subjects_95 <- sort(unique(igt_95$SubjectID))

posterior_metrics <- list()
n_draws <- 100  # Adjust as needed
set.seed(4321)

draw_ids <- sample(1:nrow(draws_95), n_draws)

for (d in seq_len(n_draws)) {
  metrics_d <- tibble(switch_rate = numeric(length(subjects_95)), entropy = numeric(length(subjects_95)), max_streak = numeric(length(subjects_95)))
  
  for (i in seq_along(subjects_95)) {
    alpha <- draws_95[[paste0("alpha[", i, "]")]][draw_ids[d]]
    w     <- draws_95[[paste0("w[", i, "]")]][draw_ids[d]]
    A     <- draws_95[[paste0("A[", i, "]")]][draw_ids[d]]
    c     <- draws_95[[paste0("c[", i, "]")]][draw_ids[d]]
    
    subj_data <- igt_95 %>% filter(SubjectID == subjects_95[i]) %>% arrange(Trial)
    wins <- subj_data$Win
    losses <- subj_data$Loss
    n_trials <- nrow(subj_data)
    
    sim <- simulate_pvl_choices(alpha, w, A, c, wins, losses, n_trials)
    
    n_switches <- sum(sim != dplyr::lag(sim, default = sim[1]))
    p_decks <- prop.table(table(factor(sim, levels = 1:4)))
    streaks <- rle(sim)$lengths
    
    metrics_d$switch_rate[i] <- n_switches / (n_trials - 1)
    metrics_d$entropy[i]     <- -sum(p_decks * log2(p_decks + 1e-8))
    metrics_d$max_streak[i]  <- max(streaks)
  }
  
  posterior_metrics[[d]] <- metrics_d
}

posterior_df <- bind_rows(posterior_metrics, .id = "draw") %>%
  pivot_longer(cols = -c(draw), names_to = "Metric", values_to = "Value")

# Create real behavior metrics for igt_95
behavior_metrics <- igt_95 %>%
  arrange(SubjectID, Trial) %>%
  group_by(SubjectID) %>%
  summarise(
    switch_rate = sum(Choice != lag(Choice, default = first(Choice))) / (n() - 1),
    entropy = {
      p <- prop.table(table(Choice))
      -sum(p * log2(p))
    },
    max_streak = max(rle(Choice)$lengths),
    .groups = "drop"
  )

# Visualize posterior predictive distributions
real_metrics <- behavior_metrics %>%
  select(SubjectID, switch_rate, entropy, max_streak) %>%
  pivot_longer(cols = -SubjectID, names_to = "Metric", values_to = "Value") %>%
  mutate(Source = "Observed")

posterior_df_summary <- posterior_df %>%
  group_by(Metric) %>%
  mutate(Source = "Posterior")

combined_ppc <- bind_rows(posterior_df_summary, real_metrics)

ggplot(combined_ppc, aes(x = Value, fill = Source)) +
  geom_density(alpha = 0.6) +
  facet_wrap(~ Metric, scales = "free") +
  labs(title = "Posterior Predictive Checks: Behavioral Metrics", x = "Metric Value") +
  theme_minimal()

