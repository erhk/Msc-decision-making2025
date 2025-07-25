// PVL Hierarchical model (cleaned)

data {
  int<lower=1> N;                     // number of subjects
  array[N] int<lower=1> T;            // number of trials per subject
  int<lower=1> T_total;               // total number of trials
  array[T_total] int<lower=1> subj;   // subject index per trial
  array[T_total] int<lower=1, upper=4> Choice;  // deck choice
  array[T_total] real Win;            // win gained
  array[T_total] real Loss;           // loss gained
}

parameters {
  // Group-level means (unbounded, transformed later)
  real mu_alpha;  // learning rate
  real mu_w;      // loss aversion
  real mu_A;      // subjective utility
  real mu_c;      // inverse temperature

  // Group-level standard deviations
  real<lower=0.001> sigma_alpha;
  real<lower=0.001> sigma_w;
  real<lower=0.001> sigma_A;
  real<lower=0.001> sigma_c;

  // Subject-specific raw parameters
  vector[N] alpha_raw;
  vector[N] w_raw;
  vector[N] A_raw;
  vector[N] c_raw;
}

transformed parameters {
  vector[N] alpha; // learning rate [0.01, 0.99]
  vector[N] w;     // loss aversion [0.1, 2.0]
  vector[N] A;     // subjective utility [0.1, 1.0]
  vector[N] c;     // inverse temperature [0.1, 10.0]

  for (i in 1:N) {
    alpha[i] = inv_logit(mu_alpha + sigma_alpha * alpha_raw[i]) * 0.98 + 0.01;
    w[i]     = inv_logit(mu_w     + sigma_w     * w_raw[i])     * 1.9  + 0.1;
    A[i]     = inv_logit(mu_A     + sigma_A     * A_raw[i])     * 0.9  + 0.1;
    c[i]     = inv_logit(mu_c     + sigma_c     * c_raw[i])     * 9.9  + 0.1;
  }
}

model {
  // Hyperpriors for group-level means
  mu_alpha ~ normal(0.5, 0.2);
  mu_w     ~ normal(1, 0.5);
  mu_A     ~ normal(0.5, 0.2);
  mu_c     ~ normal(2, 1);

  // Hyperpriors for group-level SDs
  sigma_alpha ~ exponential(1);
  sigma_w     ~ exponential(1);
  sigma_A     ~ exponential(1);
  sigma_c     ~ exponential(1);

  // Priors for subject-specific raw parameters
  alpha_raw ~ normal(0, 1);
  w_raw     ~ normal(0, 1);
  A_raw     ~ normal(0, 1);
  c_raw     ~ normal(0, 1);

  // Likelihood
  array[N] vector[4] V;  // deck values per subject
  for (i in 1:N)
    V[i] = rep_vector(0.0, 4);  // initialize deck values at 0

  for (t in 1:T_total) {
    int i = subj[t];  
    int d = Choice[t];
    real outcome = Win[t] + Loss[t];
    real abs_outcome = fmax(abs(outcome), 1e-6); // avoid pow(0, A[i])

    // Subjective utility u(x)
    real u = outcome < 0
               ? -w[i] * pow(abs_outcome, A[i])
               : pow(abs_outcome, A[i]);

    // Softmax choice probability
    vector[4] logp = c[i] * V[i];
    target += log_softmax(logp)[d];

    // Rescorla-Wagner update for chosen deck
    V[i, d] += alpha[i] * (u - V[i, d]);
  }
}
