// PVL model

// 
data {
  int<lower=1> T; // num of trials
  array[T] int<lower=1, upper=4> Choice; // choice of deck per trial
  // To calculate the subjective utility 
  array[T]real Win; // Win per trial
  array[T]real Loss; // Loss per trial
}

// 
parameters {
  real<lower=0.01, upper=0.99> alpha;
  real<lower=0.1, upper=2.0> w; // Upper bound, based on kahneman/tversky original loss aversion in pospect theory
  real<lower=0.1, upper=1.0> A; // upper bound set for numerical stability
  real<lower=0.1, upper=10> c; // also set for stability purposes, and not too deterministic, but should prevent overftting
  
  
}

// 
model {
  
  // priors
  alpha ~ beta(2, 2); // Learning rate
  w ~ normal(1, 0.5); // Loss aversion
  A ~ normal(0.5, 0.3); // Utility curvature
  c ~ normal(2, 1); // Choice sensitivity
  
  
  vector[4] V = rep_vector(0.0, 4); //Start the initial decks subjective value at 0. Copy 0.0 into each V with rep_vector
  real epsilon = 1e-6; // needed to avoid issues with pow(0,A), since stan needs derivatives, we have to ensure we never try to raise 0 to a power
  
  // Loop over trials, start at 1, and loop through until t <= T
  for (t in 1:T) { 
    // Subjective utility
    real outcome = Win[t] + Loss[t];
    real abs_outcome = fmax(abs(outcome), epsilon); // Using epislon prevent pow(0,A) issues
    // Conditional statement (ternanry operator)
    real u = (outcome < 0 // // If outcome < 0, apply loss aversion (-w) and curvature (A)
                ? -w * pow(abs_outcome, A) 
                : pow(abs_outcome, A)); // Else just apply A 
    
    // Softmax
    // Calculate log probability of choosing a deck with softmax function
    vector[4] logp = c * V; // Mulitply deck values V with inverse temperature c
    
    // log_softmax = returns log-probabilies
    // build incremental likelihood, by adding log-probability of observed choice t to the log likelihood
    target += log_softmax(logp)[Choice[t]];  
    
    // Learning rule - RW (resorcla wagner)
    int d = Choice[t]; // d = deck chosen at trial t
    V[d] += alpha * (u-V[d]); // RW update, updating value with learning rate, u-V[d] prediction error
    

  }
  
}

