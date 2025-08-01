library(dplyr)
library(tidyr)
library(posterior)
library(ggplot2)

#----------------------------------------------------------
# 1. Make trial-level lookup from dataset
#----------------------------------------------------------
make_trial_lookup <- function(data) {
  data %>%
    mutate(SubjectCondID = paste0("S", SubjectID, "_", Condition)) %>%
    group_by(SubjectCondID) %>%
    mutate(
      subj_index   = cur_group_id(),
      trial_within = row_number()
    ) %>%
    ungroup() %>%
    select(subj_index, trial_within, SubjectCondID, SubjectID, Condition, Choice)
}

lookup_95  <- make_trial_lookup(igt_95)
lookup_150 <- make_trial_lookup(igt_150)

#----------------------------------------------------------
# 2. Compute observed deck proportions over trial bins
#----------------------------------------------------------
compute_observed_props <- function(lookup_df, n_bins = 10) {
  lookup_df %>%
    rename(deck = Choice) %>%
    group_by(subj_index) %>%
    mutate(bin = cut(trial_within,
                     breaks = seq(0, max(trial_within), length.out = n_bins + 1),
                     labels = FALSE)) %>%
    ungroup() %>%
    group_by(bin, deck) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(bin) %>%
    mutate(prop = n / sum(n)) %>%
    rename(Deck = deck)
}

#----------------------------------------------------------
# 3. Extract simulated deck proportions from Choice_sim
#----------------------------------------------------------
compute_simulated_props <- function(fit, lookup_df, n_bins = 10) {
  sim_draws <- fit$draws(variables = "Choice_sim", format = "draws_df")
  
  # Extract trial index mapping
  trial_cols <- grep("^Choice_sim\\[", names(sim_draws), value = TRUE)
  trial_index <- as.integer(gsub("Choice_sim\\[|\\]", "", trial_cols))
  
  # Reshape into long format
  sim_long <- sim_draws %>%
    select(all_of(trial_cols)) %>%
    pivot_longer(cols = everything(), names_to = "trial_name", values_to = "deck") %>%
    mutate(
      trial_index  = trial_index[match(trial_name, trial_cols)],
      deck         = as.integer(deck),
      subj_index   = lookup_df$subj_index[trial_index],
      trial_within = lookup_df$trial_within[trial_index]
    )
  
  # Bin trials
  sim_long <- sim_long %>%
    mutate(bin = cut(trial_within,
                     breaks = seq(0, max(trial_within), length.out = n_bins + 1),
                     labels = FALSE))
  
  # Correct proportion calc: proportion of times deck==k in all draws
  sim_props <- sim_long %>%
    group_by(bin, Deck = deck) %>%
    summarise(prop = n() / nrow(sim_long %>% filter(bin == unique(bin))),
              .groups = "drop")
  
  return(sim_props)
}

#----------------------------------------------------------
# 4. Plotting function
#----------------------------------------------------------
plot_ppc <- function(obs_props, sim_props, title = "") {
  ggplot() +
    geom_line(data = obs_props,
              aes(x = bin, y = prop, color = factor(Deck), group = Deck),
              size = 1) +
    geom_line(data = sim_props,
              aes(x = bin, y = prop, color = factor(Deck), group = Deck),
              linetype = "dashed", size = 1) +
    scale_color_brewer(palette = "Set1", name = "Deck") +
    labs(x = "Trial Bin", y = "Proportion Chosen", title = title) +
    theme_minimal()
}

#----------------------------------------------------------
# 5. Run for IGT 95
#----------------------------------------------------------
obs_props_95 <- compute_observed_props(lookup_95, n_bins = 10)
sim_props_95 <- compute_simulated_props(fit_95, lookup_95, n_bins = 10)

plot_ppc(obs_props_95, sim_props_95, title = "IGT 95: Observed vs Simulated")

#----------------------------------------------------------
# 6. Run for IGT 150
#----------------------------------------------------------
obs_props_150 <- compute_observed_props(lookup_150, n_bins = 10)
sim_props_150 <- compute_simulated_props(fit_150, lookup_150, n_bins = 10)

plot_ppc(obs_props_150, sim_props_150, title = "IGT 150: Observed vs Simulated")



#### Behavioural metrics

compute_behavior_metrics <- function(data) {
  data %>%
    arrange(SubjectID, Trial) %>% # make sure trials are in correct order for each subject
    group_by(SubjectID) %>% # calculate metrics separately per subject
    summarise(
      n_trials   = n(), # total trials for the subject
      n_switches = sum(Choice != lag(Choice,# count where choice changes from previous trial
                                     default = first(Choice))),
      switch_rate = n_switches / (n_trials - 1), # proportion of trials that are switches
      entropy = { # Shannon entropy of deck choice distribution
        # relative frequency of each deck. 
        # Counts how many times each deck was chosen, then onverts these counts into proportions (pi)
        p <- prop.table(table(Choice)) 
        -sum(p * log2(p)) # The minus sign makes it positive, because pi log2pi is negative for 0<pi<1.
      },
      max_streak = max(rle(Choice)$lengths), # longest run of consecutive identical deck choices
      final_deck = Choice[n()], # the last deck chosen in the task
      .groups = "drop" # don't keep group structure after summarise
    )
}

behavior_metrics_95  <- compute_behavior_metrics(igt_95)
behavior_metrics_150 <- compute_behavior_metrics(igt_150)

extract_param_means <- function(fit, param_name) {
  # Extract draws for the parameter
  draws <- fit$draws(variables = param_name, format = "draws_matrix")
  # Posterior mean per subject
  colMeans(draws)
}

# IGT 95
alpha_means_95 <- extract_param_means(fit_95, "alpha")
w_means_95     <- extract_param_means(fit_95, "w")
A_means_95     <- extract_param_means(fit_95, "A")
c_means_95     <- extract_param_means(fit_95, "c")

param_df_95 <- tibble(
  SubjectID = sort(unique(igt_95$SubjectID)), # keep the SubjectID ordering consistent
  alpha = alpha_means_95,
  w = w_means_95,
  A = A_means_95,
  c = c_means_95
)

# IGT 150
alpha_means_150 <- extract_param_means(fit_150, "alpha")
w_means_150     <- extract_param_means(fit_150, "w")
A_means_150     <- extract_param_means(fit_150, "A")
c_means_150     <- extract_param_means(fit_150, "c")

param_df_150 <- tibble(
  SubjectID = sort(unique(igt_150$SubjectID)),
  alpha = alpha_means_150,
  w = w_means_150,
  A = A_means_150,
  c = c_means_150
)

# Merge behaviour metrics - observed vs. fitted
behavior_with_params_95 <- behavior_metrics_95 %>%
  inner_join(param_df_95, by = "SubjectID")

behavior_with_params_150 <- behavior_metrics_150 %>%
  inner_join(param_df_150, by = "SubjectID")


library(GGally)
# Plot
ggpairs(
  behavior_with_params_95,
  columns = c("switch_rate", "entropy", "max_streak", "alpha", "w", "A", "c"),
  title = "IGT 95: Behavioral Metrics vs. Fitted Parameters"
)

ggpairs(
  behavior_with_params_150,
  columns = c("switch_rate", "entropy", "max_streak", "alpha", "w", "A", "c"),
  title = "IGT 150: Behavioral Metrics vs. Fitted Parameters"
)



