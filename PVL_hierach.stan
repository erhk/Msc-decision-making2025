// PVL Hierarchical model 

data {
  int<lower=1> N; // number of subjects
  array[N] int<lower=1> T; // number of trials per subject
  int<lower=1> T_total; // total number of trials
  array[T_total] int<lower=1> subj; // subject index per trial, which participant made t-th choice
  array[T_total] int<lower=1, upper=4> Choice; // deck choice
  array[T_total] real Win; // win gained
  array[T_total] real Loss; // loss gained
}

parameters {
  // Group-level means
  real<lower=0.01, upper=0.99> mu_alpha; // learning rate
  real<lower=0.1, upper=2.0> mu_w; // loss aversion
  real<lower=0.1, upper=1.0> mu_A; // subjective utility
  real<lower=0.1, upper=10> mu_c; // inverse temp

  // Group-level standard deviations - how much indivudals are allowed to vary
  real<lower=0.001> sigma_alpha;
  real<lower=0.001> sigma_w;
  real<lower=0.001> sigma_A;
  real<lower=0.001> sigma_c;

  // Subject-specific parameters
  vector[N] alpha_raw;
  vector[N] w_raw;
  vector[N] A_raw;
  vector[N] c_raw;
  //vector<lower=0.01, upper=0.99>[N] alpha;
  //vector<lower=0.1, upper=2.0>[N] w;
  //vector<lower=0.1, upper=1.0>[N] A;
  //vector<lower=0.1, upper=10>[N] c;
}

// Define variables that depend on parameters, but aren't sampled directy
transformed parameters {
  

  vector[N] alpha;
  vector[N] w;
  vector[N] A;
  vector[N] c;

  for (i in 1:N) {
    alpha[i] = inv_logit(mu_alpha + sigma_alpha * alpha_raw[i]) * 0.98 + 0.01;  // → [0.01, 0.99]
    w[i]     = inv_logit(mu_w     + sigma_w     * w_raw[i])     * 1.9  + 0.1;   // → [0.1, 2.0]
    A[i]     = inv_logit(mu_A     + sigma_A     * A_raw[i])     * 0.9  + 0.1;   // → [0.1, 1.0]
    c[i]     = inv_logit(mu_c     + sigma_c     * c_raw[i])     * 9.9  + 0.1;   // → [0.1, 10.0]
  }
}

  // compute the actual (bounded) parameters from the standardised raw ones.
  // They aren't sampled directly
  //vector<lower=0.01, upper=0.99>[N] alpha;
  //vector<lower=0.1, upper=2.0>[N] w;
  //vector<lower=0.1, upper=1.0>[N] A;
  //vector<lower=0.1, upper=10>[N] c;
  
  // Transform the raw paramters. Each line in the loop takes a normal draw
  // and transforms it. Centers the parameter around the group mean, and spreads 
  // it based on group variance 
  //for (i in 1:N) {
    //alpha[i] = mu_alpha + sigma_alpha * alpha_raw[i];
    //w[i] = mu_w + sigma_w * w_raw[i];
    //A[i] = mu_A + sigma_A * A_raw[i];
    //c[i] = mu_c + sigma_c * c_raw[i];
  //}
//}


model {
  // Hyperpriors 
  // Group-level means. 
  //mu_alpha ~ beta(2, 2);
  mu_alpha ~ normal(0.5, 0.2);
  mu_w ~ normal(1, 0.5); 
  mu_A ~ beta(2, 2);
  mu_c ~ normal(2, 1);
  // Group-level variance - standard deviations, exponential is weakly informative
  // should pull towards small variance, but allows for large ones too
  sigma_alpha ~ exponential(1);
  sigma_w ~ exponential(1);
  sigma_A ~ exponential(1);
  sigma_c ~ exponential(1);

  // Individual-level priors
  // latent aramters come from a normal distirbutions, and are later transformed
  // using group-level parameters
  alpha_raw ~ normal(0,1);
  w_raw ~ normal(0,1);
  A_raw ~ normal(0,1);
  c_raw ~ normal(0,1);

  // Likelihood
  array[N] vector[4] V; // deck values per subject
  for (i in 1:N)
    V[i] = rep_vector(0.0, 4); // initial value at 0 for each subject 1

  for (t in 1:T_total) {
    int i = subj[t];  // subject i for trial t
    real outcome = Win[t] + Loss[t]; // combine reward and loss
    real abs_outcome = fmax(abs(outcome), 1e-6); // avoid pow(0,A) using small constant
    
    // If else - calculating subjective utlity 
    real u = outcome < 0 // check if outcome is a loss
    // if loss, take absolute value of the loss and raise it to power of curve A[i]
    // multiply by loss aversion w[i]. 
    // u(x) = (subjetive utlity), x = net outcome of trial 
      ? -w[i] * pow(abs_outcome, A[i]) // u(x) =−w⋅(−x)^A
    // If it's a gain we don't weigh it by w, but only apply A[i]   
      : pow(abs_outcome, A[i]); // u(x) = x^A[i]
    
    // softmax choice probability. Calculate log-probability of choices, then
    // update the likelihood based on the probability of the observed choice
    vector[4] logp = c[i] * V[i];
    target += log_softmax(logp)[Choice[t]];
    
    // Update the chosen deck value with RW rule
    int d = Choice[t];
    V[i, d] += alpha[i] * (u - V[i, d]);
  }
}
