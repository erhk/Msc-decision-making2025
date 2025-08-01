// PVL Hierarchical model (corrected with PPC and beta prior for mu_alpha)

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
  // Group-level means (on (0,1) scale for beta priors)
  real<lower=0, upper=1> mu_alpha_raw;      // learning rate (raw)
  real<lower=0.1, upper=2.0> mu_w;          // loss aversion
  real<lower=0, upper=1> mu_A_raw;          // subjective utility (raw)
  real<lower=0.1, upper=10> mu_c;           // inverse temp

  // Group-level standard deviations
  real<lower=0.001> sigma_alpha;
  real<lower=0.001> sigma_w;
  real<lower=0.001> sigma_A;
  real<lower=0.001> sigma_c;

  // Subject-specific standardized deviations
  vector[N] alpha_raw;
  vector[N] w_raw;
  vector[N] A_raw;
  vector[N] c_raw;
}

transformed parameters {
  real mu_alpha = mu_alpha_raw * 0.98 + 0.01;  // scale to [0.01, 0.99]
  real mu_A     = mu_A_raw     * 0.9  + 0.1;   // scale to [0.1, 1.0]

  vector[N] alpha;
  vector[N] w;
  vector[N] A;
  vector[N] c;

  for (i in 1:N) {
    alpha[i] = inv_logit(mu_alpha + sigma_alpha * alpha_raw[i]) * 0.98 + 0.01; // [0.01, 0.99]
    w[i]     = inv_logit(mu_w     + sigma_w     * w_raw[i])     * 1.9  + 0.1;  // [0.1, 2.0]
    A[i]     = inv_logit(mu_A     + sigma_A     * A_raw[i])     * 0.9  + 0.1;  // [0.1, 1.0]
    c[i]     = inv_logit(mu_c     + sigma_c     * c_raw[i])     * 9.9  + 0.1;  // [0.1, 10.0]
  }
}

model {
  // Hyperpriors
  mu_alpha_raw ~ beta(2, 2);              // prior on (0,1)
  mu_w         ~ normal(1, 0.5);
  mu_A_raw     ~ beta(2, 2);              // prior on (0,1)
  mu_c         ~ normal(2, 1);

  sigma_alpha ~ exponential(1);
  sigma_w     ~ exponential(1);
  sigma_A     ~ exponential(1);
  sigma_c     ~ exponential(1);

  // Subject-level priors (standard normal for latent variables)
  alpha_raw ~ normal(0, 1);
  w_raw     ~ normal(0, 1);
  A_raw     ~ normal(0, 1);
  c_raw     ~ normal(0, 1);

  // Likelihood
  array[N] vector[4] V;  // deck values per subject
  for (i in 1:N)
    V[i] = rep_vector(0.0, 4);  // initialize deck values

  for (t in 1:T_total) {
    int i = subj[t];
    int d = Choice[t];
    real outcome = Win[t] + Loss[t];
    real abs_outcome = fmax(abs(outcome), 1e-6);  // avoid pow(0, A[i])

    // Subjective utility function
    real u = outcome < 0 ? -w[i] * pow(abs_outcome, A[i])
                         :       pow(abs_outcome, A[i]);

    // Choice probability via softmax
    vector[4] logp = c[i] * V[i];
    logp -= max(logp);  // numerical stability
    target += log_softmax(logp)[Choice[t]];

    
    // Rescorla-Wagner update
    V[i, d] += alpha[i] * (u - V[i, d]);
  }
}

generated quantities {
  array[T_total] int Choice_sim;

  {
    array[N] vector[4] V_sim;
    for (i in 1:N)
      V_sim[i] = rep_vector(0.0, 4);

    for (t in 1:T_total) {
      int i = subj[t];
      real outcome = Win[t] + Loss[t];
      real abs_outcome = fmax(abs(outcome), 1e-6);

      real u = outcome < 0 ? -w[i] * pow(abs_outcome, A[i])
                           :       pow(abs_outcome, A[i]);



      vector[4] logp = c[i] * V_sim[i];
      logp -= max(logp);  // ✅ stabilize softmax input
      vector[4] p = softmax(logp);

      Choice_sim[t] = categorical_rng(p);

      int d = Choice[t];  // use actual choice for learning
      V_sim[i, d] += alpha[i] * (u - V_sim[i, d]);
    }
  }
}
