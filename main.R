### Decision Making - Explotation vs. exploration in IGT ###

pacman::p_load(tidyverse)

# Load data
igt_all_with_wins <- read_csv("data/Final_IGT_Dataset_with_Wins_and_Running_Total.csv", sep = ",")


# STEP 1: Prepare outcome pools by deck
prepare_deck_outcomes <- function(df) {
  df %>%
    select(Deck = Choice, Win, Loss) %>%
    group_by(Deck) %>%
    group_split()
}

# STEP 2: Define PVL-Delta agent simulator
simulate_pvl_agents <- function(n_agents = 100, n_trials = 95, deck_outcomes) {
  
  # Storage
  results <- list()
  
  for (agent_id in 1:n_agents) {
    
    # Sample agent parameters
    alpha  <- runif(1, 0.1, 0.9)         # learning rate
    lambda <- runif(1, 0.5, 2.0)         # loss aversion
    c      <- runif(1, 0.5, 5.0)         # inverse temperature
    
    # Initialize value estimates
    V <- rep(0, 4)
    
    # Storage for this agent
    agent_data <- tibble(
      AgentID = agent_id,
      Trial = integer(n_trials),
      Choice = integer(n_trials),
      Win = numeric(n_trials),
      Loss = numeric(n_trials)
    #  Value_A = numeric(n_trials),
    #  Value_B = numeric(n_trials),
    #  Value_C = numeric(n_trials),
    #  Value_D = numeric(n_trials)
    )
    
    for (t in 1:n_trials) {
      # Softmax choice
      maxV <- max(c * V)
      probs <- exp(c * V - maxV) / sum(exp(c * V - maxV))
      choice <- sample(1:4, size = 1, prob = probs)
      
      # Sample outcome from real data pool
      outcome_pool <- deck_outcomes[[choice]]
      sampled <- outcome_pool[sample(nrow(outcome_pool), 1), ]
      
      # Calculate subjective utility
      u_win  <- sampled$Win^lambda
      u_loss <- -lambda * (-sampled$Loss)^lambda * as.numeric(sampled$Loss < 0)
      u <- u_win + u_loss
      
      
      # Update values using delta rule
      V[choice] <- V[choice] + alpha * (u - V[choice])
      
      # Record trial
      agent_data$Trial[t] <- t
      agent_data$Choice[t] <- choice
      agent_data$Win[t] <- sampled$Win
      agent_data$Loss[t] <- sampled$Loss
      #agent_data$Value_A[t] <- V[1]
      #agent_data$Value_B[t] <- V[2]
      #agent_data$Value_C[t] <- V[3]
      #agent_data$Value_D[t] <- V[4]
      
    }
    
    results[[agent_id]] <- agent_data
  }
  
  bind_rows(results)
}


deck_outcomes <- prepare_deck_outcomes(igt_all_with_wins)
sim_data <- simulate_pvl_agents(n_agents = 100, n_trials = 95, deck_outcomes = deck_outcomes)

# Preview result
head(sim_data)
