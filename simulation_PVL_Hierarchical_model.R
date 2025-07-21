#### ANALYSIS ####


#### Load everything ####
#Load library
pacman::p_load(tidyverse, cmdstanr, posterior, bayesplot, ggplot2, tidyr, dplyr,purrr, GGally)

# Load empirical data
igt_all_with_wins <- read_csv("data/Final_IGT_Dataset_with_Wins_and_Running_Total.csv")

# subset to different conditions (studies). Not using igt_100, fitting the model takes too long
igt_95  <- igt_all_with_wins %>% filter(Condition == "IGT_95")
igt_150 <- igt_all_with_wins %>% filter(Condition == "IGT_150")

# Loads fits
fit_95 <- readRDS("fits/fit_95_cmdstanr.stanfit")
fit_150 <- readRDS("fits/fit_150_cmdstanr.stanfit")

# Extract draws
draws_95 <- as_draws_df(fit_95$draws())
draws_150 <- as_draws_df(fit_150$draws())





### Diagnostics of fits ####
# Summarise
summary_95 <- summarise_draws(draws_95)
summary_150 <- summarise_draws(draws_150)

# Convert to draws_array for plotting
draws_95_arr <- as_draws_array(fit_95$draws())
draws_150_arr <- as_draws_array(fit_150$draws())

# Trace plots for group-level parameters
#Well mixed
mcmc_trace(draws_95_arr, pars = c("mu_alpha", "mu_c", "mu_A", "mu_w"))
#Well mixed, though noticabily narrower ranges for mu_A, and mu_w compared to 95.
mcmc_trace(draws_150_arr, pars = c("mu_alpha", "mu_c", "mu_A", "mu_w"))

# Slightly below 0.8 for both, 0.60-70. No divergences or hitting treedepth
fit_95$diagnostic_summary()
fit_150$diagnostic_summary()





#### Simulate agents using empirical posterior parameter #### 


# Simulate IGT function:

# Parameters draws: alpha (learning rate), w (loss aversion), A (subjectuve utility/utility curve) 
# and c (inverse temp/softmax). 

simulate_pvl_choices <- function(alpha, w, A, c, wins, losses, n_trials) {
  V <- rep(0, 4) # Initial deck values
  choices <- integer(n_trials) # store choices
  
  # iterate over ecah trial
  for (t in seq_len(n_trials)) {
    probs <- softmax(c * V) # apply softmax rule to each deck
    choice <- sample(1:4, size = 1, prob = probs) # Samples one deck based on the softmax probability
    choices[t] <- choice # save each choice
    
    outcome <- wins[t] + losses[t] # Total outcome
    abs_outcome <- max(abs(outcome), 1e-6) # to avoid getting a case of pow(0, A)!
    # calculate subjective utility
    u <- if (outcome < 0) -w * abs_outcome^A else abs_outcome^A 
    
    # RW update rule. Updates the value of the chosen deck based on the prediction error 
    # between observed utility and expected value.
    V[choice] <- V[choice] + alpha * (u - V[choice])
  }

  return(choices)
}

# Helper function for pvl simulation
softmax <- function(x) {
  exp_x <- exp(x - max(x))
  exp_x / sum(exp_x)
}


# ---- Simulate IGT 95 Using Posterior Means ----------------------------------

# Identify subjects
subjects_95 <- sort(unique(igt_95$SubjectID)) 
n_subjects_95 <- length(subjects_95)

# Use posterior means for all subject-level parameters
alpha_means_95 <- sapply(1:n_subjects_95, function(i) mean(draws_95[[paste0("alpha[", i, "]")]]))
w_means_95     <- sapply(1:n_subjects_95, function(i) mean(draws_95[[paste0("w[", i, "]")]]))
A_means_95     <- sapply(1:n_subjects_95, function(i) mean(draws_95[[paste0("A[", i, "]")]]))
c_means_95     <- sapply(1:n_subjects_95, function(i) mean(draws_95[[paste0("c[", i, "]")]]))

# Placeholder for simulated choices
sim_choices_95 <- list()

# Simulate behavior for each subject
set.seed(1990)
for (i in seq_along(subjects_95)) {
  subj_id <- subjects_95[i]
  
  subj_data <- igt_95 %>%
    filter(SubjectID == subj_id) %>%
    arrange(Trial)
  
  n_trials <- nrow(subj_data)
  
  wins <- subj_data$Win
  losses <- subj_data$Loss
  
  alpha_i <- alpha_means_95[i]
  w_i     <- w_means_95[i]
  A_i     <- A_means_95[i]
  c_i     <- c_means_95[i]
  
  sim_choices_95[[i]] <- simulate_pvl_choices(alpha_i, w_i, A_i, c_i, wins, losses, n_trials)
}

# Create simulated data frame
sim_df_95 <- map2_df(sim_choices_95, subjects_95, function(sim, subj_id) {
  tibble(
    SubjectID = subj_id,
    Trial = seq_along(sim),
    Choice = sim,
    Source = "Simulated"
  )
})

# Prepare real data
real_df_95 <- igt_95 %>%
  select(SubjectID, Trial, Choice) %>%
  mutate(Source = "Real")

# Combine into one df
combined_df_95 <- bind_rows(real_df_95, sim_df_95)

# Bin trials into 10 bins
combined_df_95 <- combined_df_95 %>%
  mutate(TrialBin = ntile(Trial, 10))

# Compute proportions by deck and bin
prop_df_95 <- combined_df_95 %>%
  group_by(Source, TrialBin, Choice) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(Source, TrialBin) %>%
  mutate(prop = n / sum(n))

# Plot PPC
ggplot(prop_df_95, aes(x = TrialBin, y = prop, color = factor(Choice), group = Choice)) +
  geom_line(linewidth = 1.2) +
  facet_wrap(~ Source) +
  labs(
    x = "Trial Bin (1–10)",
    y = "Proportion of Deck Choices",
    color = "Deck",
    title = "IGT 95: Real vs. Simulated Deck Choice Proportions"
  ) +
  theme_minimal()




# ---- Posterior Distribution of c (IGT 95) ------------------------------------

summary(draws_95[, grep("c\\[", colnames(draws_95))])

c_params_95 <- draws_95 %>%
  select(starts_with("c[")) %>%
  pivot_longer(cols = everything(), names_to = "param", values_to = "value")

ggplot(c_params_95, aes(x = value)) +
  geom_density(fill = "steelblue", alpha = 0.6) +
  labs(title = "IGT 95: Distribution of Subject-Level c", x = "c", y = "Density")

# ---- Compute Real Behavioral Metrics and Compare with Parameters ------------

# Behavioral summaries
behavior_metrics_95 <- igt_95 %>%
  arrange(SubjectID, Trial) %>%
  group_by(SubjectID) %>%
  summarise(
    n_trials = n(),
    n_switches = sum(Choice != lag(Choice, default = first(Choice))),
    switch_rate = n_switches / (n_trials - 1),
    entropy = {
      p <- prop.table(table(Choice))
      -sum(p * log2(p))
    },
    max_streak = max(rle(Choice)$lengths),
    final_deck = Choice[n()],
    .groups = "drop"
  )

# Subject-level parameter summary
param_df_95 <- tibble(
  SubjectID = sort(unique(igt_95$SubjectID)),
  alpha = alpha_means_95,
  w     = w_means_95,
  A     = A_means_95,
  c     = c_means_95
)

# Merge behavior and parameters
behavior_with_params_95 <- behavior_metrics_95 %>%
  inner_join(param_df_95, by = "SubjectID")

ggpairs(
  behavior_with_params_95,
  columns = c("switch_rate", "entropy", "max_streak", "alpha", "w", "A", "c"),
  title = "IGT 95: Behavioral Metrics vs. Fitted Parameters"
)


# ---- Simulate IGT 150 Using Posterior Means ----------------------------------

# Identify subjects
subjects_150 <- sort(unique(igt_150$SubjectID)) 
n_subjects_150 <- length(subjects_150)

# Use posterior means for all subject-level parameters
alpha_means_150 <- sapply(1:n_subjects_150, function(i) mean(draws_150[[paste0("alpha[", i, "]")]]))
w_means_150     <- sapply(1:n_subjects_150, function(i) mean(draws_150[[paste0("w[", i, "]")]]))
A_means_150     <- sapply(1:n_subjects_150, function(i) mean(draws_150[[paste0("A[", i, "]")]]))
c_means_150     <- sapply(1:n_subjects_150, function(i) mean(draws_150[[paste0("c[", i, "]")]]))

# Placeholder for simulated choices
sim_choices_150 <- list()

# Simulate behavior for each subject
set.seed(1990)
for (i in seq_along(subjects_150)) {
  subj_id <- subjects_150[i]
  
  subj_data <- igt_150 %>%
    filter(SubjectID == subj_id) %>%
    arrange(Trial)
  
  n_trials <- nrow(subj_data)
  
  wins <- subj_data$Win
  losses <- subj_data$Loss
  
  alpha_i <- alpha_means_150[i]
  w_i     <- w_means_150[i]
  A_i     <- A_means_150[i]
  c_i     <- c_means_150[i]
  
  sim_choices_150[[i]] <- simulate_pvl_choices(alpha_i, w_i, A_i, c_i, wins, losses, n_trials)
}

# Create simulated data frame
sim_df_150 <- map2_df(sim_choices_150, subjects_150, function(sim, subj_id) {
  tibble(
    SubjectID = subj_id,
    Trial = seq_along(sim),
    Choice = sim,
    Source = "Simulated"
  )
})

# Prepare real data
real_df_150 <- igt_150 %>%
  select(SubjectID, Trial, Choice) %>%
  mutate(Source = "Real")

# Combine into one df
combined_df_150 <- bind_rows(real_df_150, sim_df_150)

# Bin trials into 10 bins
combined_df_150 <- combined_df_150 %>%
  mutate(TrialBin = ntile(Trial, 10))

# Compute proportions by deck and bin
prop_df_150 <- combined_df_150 %>%
  group_by(Source, TrialBin, Choice) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(Source, TrialBin) %>%
  mutate(prop = n / sum(n))

# Plot PPC
ggplot(prop_df_150, aes(x = TrialBin, y = prop, color = factor(Choice), group = Choice)) +
  geom_line(linewidth = 1.2) +
  facet_wrap(~ Source) +
  labs(
    x = "Trial Bin (1–10)",
    y = "Proportion of Deck Choices",
    color = "Deck",
    title = "IGT 150: Real vs. Simulated Deck Choice Proportions"
  ) +
  theme_minimal()

# ---- Posterior Distribution of c (IGT 150) ------------------------------------

summary(draws_150[, grep("c\\[", colnames(draws_150))])

c_params_150 <- draws_150 %>%
  select(starts_with("c[")) %>%
  pivot_longer(cols = everything(), names_to = "param", values_to = "value")

ggplot(c_params_150, aes(x = value)) +
  geom_density(fill = "steelblue", alpha = 0.6) +
  labs(title = "IGT 150: Distribution of Subject-Level c", x = "c", y = "Density")

# ---- Compute Real Behavioral Metrics and Compare with Parameters ------------

# Behavioral summaries
behavior_metrics_150 <- igt_150 %>%
  arrange(SubjectID, Trial) %>%
  group_by(SubjectID) %>%
  summarise(
    n_trials = n(),
    n_switches = sum(Choice != lag(Choice, default = first(Choice))),
    switch_rate = n_switches / (n_trials - 1),
    entropy = {
      p <- prop.table(table(Choice))
      -sum(p * log2(p))
    },
    max_streak = max(rle(Choice)$lengths),
    final_deck = Choice[n()],
    .groups = "drop"
  )

# Subject-level parameter summary
param_df_150 <- tibble(
  SubjectID = sort(unique(igt_150$SubjectID)),
  alpha = alpha_means_150,
  w     = w_means_150,
  A     = A_means_150,
  c     = c_means_150
)

# Merge behavior and parameters
behavior_with_params_150 <- behavior_metrics_150 %>%
  inner_join(param_df_150, by = "SubjectID")

ggpairs(
  behavior_with_params_150,
  columns = c("switch_rate", "entropy", "max_streak", "alpha", "w", "A", "c"),
  title = "IGT 150: Behavioral Metrics vs. Fitted Parameters"
)

