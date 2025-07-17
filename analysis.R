#### ANALYSIS ####


#### Load everything ####
#Load library
pacman::p_load(tidyverse, cmdstanr, posterior, bayesplot, ggplot2, tidyr, dplyr,purrr)

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


# Simulate!

# Load posterior draws from whichever dataset used to simulate from, using 95 here
draws <- as_draws_df(fit_95$draws()) 

# Identify subjects
subjects <- sort(unique(igt_95$SubjectID)) 
n_subjects <- length(subjects)

# Placeholder for simulated choices
sim_choices <- list()

# Use posterior means for all subject-level parameters
alpha_means <- sapply(1:n_subjects, function(i) mean(draws[[paste0("alpha[", i, "]")]]))
w_means <- sapply(1:n_subjects, function(i) mean(draws[[paste0("w[", i, "]")]]))
A_means <- sapply(1:n_subjects, function(i) mean(draws[[paste0("A[", i, "]")]]))
c_means <- sapply(1:n_subjects, function(i) mean(draws[[paste0("c[", i, "]")]]))


# Loop over subjects
for (i in seq_along(subjects)) {
  subj_id <- subjects[i]
  
  # Filter rows for this subject, sorted by trial just in case
  subj_data <- igt_95 %>% 
    filter(SubjectID == subj_id) %>% 
    arrange(Trial)
  
  n_trials <- nrow(subj_data)
  
  alpha_i <- alpha_means[i]
  w_i <- w_means[i]
  A_i <- A_means[i]
  c_i <- c_means[i]
  
  # Real outcomes for simulation
  wins <- subj_data$Win
  losses <- subj_data$Loss
  
  # Simulate choice sequence
  sim_choices[[i]] <- simulate_pvl_choices(alpha_i, w_i, A_i, c_i, wins, losses, n_trials)
}


# Combine simulated data into a df
sim_df <- map2_df(sim_choices, subjects, function(sim, subj_id) {
  tibble(
    SubjectID = subj_id,
    Trial = seq_along(sim),
    Choice = sim,
    Source = "Simulated"
  )
})

# Prepare real data
real_df <- igt_95 %>%
  select(SubjectID, Trial, Choice) %>%
  mutate(Source = "Real")

# Combine into one df
combined_df <- bind_rows(real_df, sim_df)

# Bin trials into sets of 10 - otherwise it's hard to read the plot
combined_df <- combined_df %>%
  mutate(TrialBin = ntile(Trial, 10))

# Count choices per deck per trial bin per source
prop_df <- combined_df %>%
  group_by(Source, TrialBin, Choice) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(Source, TrialBin) %>%
  mutate(prop = n / sum(n))


#### Explore comparison

ggplot(prop_df, aes(x = TrialBin, y = prop, color = factor(Choice), group = Choice)) +
  geom_line(linewidth = 1.2) +
  facet_wrap(~ Source) +
  labs(
    x = "Trial Bin (1–10)",
    y = "Proportion of Deck Choices",
    color = "Deck",
    title = "Real vs. Simulated Deck Choice Proportions"
  ) +
  theme_minimal()

# Check the parameters draws for c, 
summary(draws[, grep("c\\[", colnames(draws))])

c_params <- draws %>% 
  select(starts_with("c[")) %>% 
  pivot_longer(cols = everything(), names_to = "param", values_to = "value")

ggplot(c_params, aes(x = value)) +
  geom_density(fill = "steelblue", alpha = 0.6) +
  labs(title = "Distribution of Subject-Level c (Inverse Temperature)", x = "c", y = "Density")

