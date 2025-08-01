#### ANALYSIS ####


#### Load everything ####
#Load library
pacman::p_load(tidyverse, cmdstanr, posterior, bayesplot, ggplot2, tidyr, dplyr,purrr, GGally)

# Load empirical data
igt_all_with_wins <- read_csv("data/Final_IGT_Dataset.csv")

# subset to different conditions (studies). Not using igt_100, fitting the model takes too long
igt_95  <- igt_all_with_wins %>% filter(Condition == "IGT_95")
igt_150 <- igt_all_with_wins %>% filter(Condition == "IGT_150")

# Loads fits
fit_95 <- readRDS("fits/fit_95_fix_cmdstanr.stanfit")
fit_150 <- readRDS("fits/fit_150_fix_cmdstanr.stanfit")
fit_sim <- readRDS("fits/fit_sim_cmdstanr.stanfit")

# Extract draws
draws_95 <- as_draws_df(fit_95$draws())
draws_150 <- as_draws_df(fit_150$draws())
draws_sim <- as_draws_df(fit_sim$draws())




### Diagnostics of fits ####
# Summarise
summary_95 <- summarise_draws(draws_95)
summary_150 <- summarise_draws(draws_150)
summary_sim <- summarise_draws(draws_sim)

# Convert to draws_array for plotting
draws_95_arr <- as_draws_array(fit_95$draws())
draws_150_arr <- as_draws_array(fit_150$draws())
draws_sim_arr <- as_draws_array(fit_sim$draws())

# Trace plots for group-level parameters
#Well mixed
mcmc_trace(draws_95_arr, pars = c("mu_alpha", "mu_c", "mu_A", "mu_w"))
#Well mixed, though noticabily narrower ranges for mu_A, and mu_w compared to 95.
mcmc_trace(draws_150_arr, pars = c("mu_alpha", "mu_c", "mu_A", "mu_w"))

mcmc_trace(draws_sim_arr, pars = c("mu_alpha", "mu_c", "mu_A", "mu_w"))


# Slightly below 0.8 for both, 0.60-70. No divergences or hitting treedepth
fit_95$diagnostic_summary()
fit_150$diagnostic_summary()
fit_sim$diagnostic_summary()


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


#### IGT 95: Simulate IGT 95 using posterior means ----------------------------------

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
  w_i <- w_means_95[i]
  A_i <- A_means_95[i]
  c_i <- c_means_95[i]
  
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

# NO binning — keep Trial as-is
# Group by trial number and deck
# prop_df_95_trial <- combined_df_95 %>%
#   group_by(Source, Trial, Choice) %>%
#   summarise(n = n(), .groups = "drop") %>%
#   group_by(Source, Trial) %>%
#   mutate(prop = n / sum(n))


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

# ggplot(prop_df_95_trial, aes(x = Trial, y = prop, color = factor(Choice), group = Choice)) +
#   geom_line(linewidth = 0.5) +
#   facet_wrap(~ Source) +
#   labs(
#     x = "Trial",
#     y = "Proportion of Deck Choices",
#     color = "Deck",
#     title = "IGT 95: Real vs. Simulated Deck Choice Proportions (Per Trial)"
#   ) +
#   theme_minimal()





#### IGT 95: Posterior distribution of c  ------------------------------------

summary(draws_95[, grep("c\\[", colnames(draws_95))])

c_params_95 <- draws_95 %>%
  select(starts_with("c[")) %>%
  pivot_longer(cols = everything(), names_to = "param", values_to = "value")

ggplot(c_params_95, aes(x = value)) +
  geom_density(fill = "steelblue", alpha = 0.6) +
  labs(title = "IGT 95: Distribution of Subject-Level c", x = "c", y = "Density")

#### IGT 95: Compute real behavior metrics, compare with params ------------

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
  w = w_means_95,
  A = A_means_95,
  c = c_means_95
)

# Merge behavior and parameters
behavior_with_params_95 <- behavior_metrics_95 %>%
  inner_join(param_df_95, by = "SubjectID")

ggpairs(
  behavior_with_params_95,
  columns = c("switch_rate", "entropy", "max_streak", "alpha", "w", "A", "c"),
  title = "IGT 95: Behavioral Metrics vs. Fitted Parameters"
)

#### IGT 95: Parameter posterior plots ---------------------------------------------

# Tidy subject-level parameter means
param_means_df <- tibble(
  SubjectID = sort(unique(igt_95$SubjectID)),
  alpha = alpha_means_95,
  w     = w_means_95,
  A     = A_means_95,
  c     = c_means_95
)

# Merge behavioral metrics and parameters again if needed
param_behavior_df <- behavior_with_params_95 %>%
  select(SubjectID, switch_rate, entropy, max_streak) %>%
  left_join(param_means_df, by = "SubjectID") %>%
  pivot_longer(cols = c(alpha, w, A, c), names_to = "Parameter", values_to = "Value")

# Plot parameter distributions vs behavioral clusters
ggplot(param_behavior_df, aes(x = Value)) +
  geom_density(fill = "skyblue", alpha = 0.6) +
  facet_wrap(~ Parameter, scales = "free") +
  labs(title = "IGT 95: Posterior Predictive Distributions of Subject-Level Parameters",
       x = "Parameter Value", y = "Density") +
  theme_minimal()

# Correlation matrix between parameters and behavioral indices
cor_df <- behavior_with_params_95 %>%
  select(switch_rate, entropy, max_streak, alpha, w, A, c)

cor_matrix <- cor(cor_df)
round(cor_matrix, 2)

# Posterior distributions of group-level means (mu parameters)
bayesplot::mcmc_areas(
  draws_95 %>% select(mu_alpha, mu_w, mu_A, mu_c),
  prob = 0.8,
  ggtitle("Posterior Distributions of Group-Level Parameters (80% HDI)")
)

# Tidier version of abive plot
group_params <- draws_95 %>%
  select(mu_alpha, mu_w, mu_A, mu_c)

group_summary <- summarise_draws(group_params, 
                                 mean, median, sd, 
                                 ~quantile2(.x, probs = c(0.025, 0.975)))

print(group_summary)

color_scheme_set("blue")
mcmc_areas(group_params,
           pars = c("mu_alpha", "mu_w", "mu_A", "mu_c"),
           prob = 0.95) + 
  ggtitle("Posterior Distributions of Group-Level Parameters (95% HDI)")


# Extract posterior means for each subject-level parameter
param_names <- c("alpha", "w", "A", "c")

param_densities <- map_dfr(param_names, function(p) {
  subject_cols <- draws_95 %>% select(starts_with(paste0(p, "[")))
  subject_means <- colMeans(subject_cols)
  
  tibble(Parameter = p,
         Value = subject_means)
})





#### Simulate IGT 150 using posterior means ----------------------------------

# Identify subjects
subjects_150 <- sort(unique(igt_150$SubjectID)) 
n_subjects_150 <- length(subjects_150)

# Use posterior means for all subject-level parameters
alpha_means_150 <- sapply(1:n_subjects_150, function(i) mean(draws_150[[paste0("alpha[", i, "]")]]))
w_means_150 <- sapply(1:n_subjects_150, function(i) mean(draws_150[[paste0("w[", i, "]")]]))
A_means_150 <- sapply(1:n_subjects_150, function(i) mean(draws_150[[paste0("A[", i, "]")]]))
c_means_150 <- sapply(1:n_subjects_150, function(i) mean(draws_150[[paste0("c[", i, "]")]]))

# Placeholder for simulated choices
sim_choices_150 <- list()

# Simulate behavior for each subject
#set.seed(1990)
for (i in seq_along(subjects_150)) {
  subj_id <- subjects_150[i]
  
  subj_data <- igt_150 %>%
    filter(SubjectID == subj_id) %>%
    arrange(Trial)
  
  n_trials <- nrow(subj_data)
  
  wins <- subj_data$Win
  losses <- subj_data$Loss
  
  alpha_i <- alpha_means_150[i]
  w_i <- w_means_150[i]
  A_i <- A_means_150[i]
  c_i <- c_means_150[i]
  
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

# NO binning 
# Group by trial number and deck
# prop_df_150_trial <- combined_df_150 %>%
#   group_by(Source, Trial, Choice) %>%
#   summarise(n = n(), .groups = "drop") %>%
#   group_by(Source, Trial) %>%
#   mutate(prop = n / sum(n))


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

# ggplot(prop_df_150_trial, aes(x = Trial, y = prop, color = factor(Choice), group = Choice)) +
#   geom_line(linewidth = 0.1) +
#   facet_wrap(~ Source) +
#   labs(
#     x = "Trial",
#     y = "Proportion of Deck Choices",
#     color = "Deck",
#     title = "IGT 150: Real vs. Simulated Deck Choice Proportions (Per Trial)"
#   ) +
#   theme_minimal()

#### Posterior distribution of c (IGT 150) -------------------------------------

summary(draws_150[, grep("c\\[", colnames(draws_150))])

c_params_150 <- draws_150 %>%
  select(starts_with("c[")) %>%
  pivot_longer(cols = everything(), names_to = "param", values_to = "value")

ggplot(c_params_150, aes(x = value)) +
  geom_density(fill = "steelblue", alpha = 0.6) +
  labs(title = "IGT 150: Distribution of Subject-Level c", x = "c", y = "Density")

#### Compute real behavioral metrics and compare with parameters ------------

# Behavior summaries
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
  w = w_means_150,
  A = A_means_150,
  c = c_means_150
)

# Merge behavior and parameters
behavior_with_params_150 <- behavior_metrics_150 %>%
  inner_join(param_df_150, by = "SubjectID")

ggpairs(
  behavior_with_params_150,
  columns = c("switch_rate", "entropy", "max_streak", "alpha", "w", "A", "c"),
  title = "IGT 150: Behavioral Metrics vs. Fitted Parameters"
)

#### IGT 150: Parameter posterior plots ---------------------------------------------

# Tidy subject-level parameter means
param_means_df_150 <- tibble(
  SubjectID = sort(unique(igt_150$SubjectID)),
  alpha = alpha_means_150,
  w     = w_means_150,
  A     = A_means_150,
  c     = c_means_150
)

# Merge behavioral metrics and parameters again if needed
param_behavior_df_150 <- behavior_with_params_150 %>%
  select(SubjectID, switch_rate, entropy, max_streak) %>%
  left_join(param_means_df_150, by = "SubjectID") %>%
  pivot_longer(cols = c(alpha, w, A, c), names_to = "Parameter", values_to = "Value")

# Plot parameter distributions vs behavioral clusters
ggplot(param_behavior_df_150, aes(x = Value)) +
  geom_density(fill = "skyblue", alpha = 0.6) +
  facet_wrap(~ Parameter, scales = "free") +
  labs(title = "IGT 150: Posterior Predictive Distributions of Subject-Level Parameters",
       x = "Parameter Value", y = "Density") +
  theme_minimal()

# Correlation matrix between parameters and behavioral indices
cor_df <- behavior_with_params_150 %>%
  select(switch_rate, entropy, max_streak, alpha, w, A, c)

cor_matrix <- cor(cor_df)
round(cor_matrix, 2)

# Posterior distributions of group-level means (mu parameters)
bayesplot::mcmc_areas(
  draws_150 %>% select(mu_alpha, mu_w, mu_A, mu_c),
  prob = 0.8
)


#### Combined 95 and 150
# Parameters of interest
param_names <- c("alpha", "w", "A", "c")

# Extract and reshape subject-level draws for IGT 95
draws_95_long <- draws_95 %>%
  select(matches("^alpha\\[|^w\\[|^A\\[|^c\\[")) %>%
  pivot_longer(everything(), names_to = "param", values_to = "value") %>%
  mutate(
    param = str_extract(param, "^[a-zA-Z]+"),
    Task = "IGT 95"
  )

# Extract and reshape subject-level draws for IGT 150
draws_150_long <- draws_150 %>%
  select(matches("^alpha\\[|^w\\[|^A\\[|^c\\[")) %>%
  pivot_longer(everything(), names_to = "param", values_to = "value") %>%
  mutate(
    param = str_extract(param, "^[a-zA-Z]+"),
    Task = "IGT 150"
  )

# Combine into one dataframe
combined_param_draws <- bind_rows(draws_95_long, draws_150_long)

ggplot(combined_param_draws, aes(x = value, fill = Task)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~ param, scales = "free", ncol = 2) +
  scale_fill_manual(values = c("IGT 95" = "tomato", "IGT 150" = "steelblue")) +
  labs(
    title = "Comparison of Subject-Level Posterior Distributions",
    x = "Parameter Value",
    y = "Density"
  ) +
  theme_minimal()


combined_param_draws %>%
  group_by(Task, param) %>%
  summarise(
    Mean = mean(value),
    Median = median(value),
    SD = sd(value),
    `2.5%` = quantile(value, 0.025),
    `97.5%` = quantile(value, 0.975),
    .groups = "drop"
  )

