### PVL Hierarchical model 

data {
  int<lower=1> N; // number of subjects
  int<lower=1> T[N]; // number of trials per subject
  int<lower=1> T_total; // total number of trials
  array[T_total] int<lower=1> subj; // subject index per trial
  array[T_total] int<lower=1, upper=4> Choice;
  array[T_total] real Win;
  array[T_total] real Loss;
}

parameters {
  // Group-level means
  real<lower=0.01, upper=0.99> mu_alpha;
  real<lower=0.1, upper=2.0> mu_w;
  real<lower=0.1, upper=1.0> mu_A;
  real<lower=0.1, upper=10> mu_c;

  // Group-level standard deviations
  real<lower=0.001> sigma_alpha;
  real<lower=0.001> sigma_w;
  real<lower=0.001> sigma_A;
  real<lower=0.001> sigma_c;

  // Subject-specific parameters
  vector<lower=0.01, upper=0.99>[N] alpha;
  vector<lower=0.1, upper=2.0>[N] w;
  vector<lower=0.1, upper=1.0>[N] A;
  vector<lower=0.1, upper=10>[N] c;
}

model {
  // Hyperpriors
  mu_alpha ~ beta(2, 2);
  mu_w ~ normal(1, 0.5);
  mu_A ~ beta(2, 2);
  mu_c ~ normal(2, 1);

  sigma_alpha ~ exponential(1);
  sigma_w ~ exponential(1);
  sigma_A ~ exponential(1);
  sigma_c ~ exponential(1);

  // Individual-level priors
  alpha ~ normal(mu_alpha, sigma_alpha);
  w ~ normal(mu_w, sigma_w);
  A ~ normal(mu_A, sigma_A);
  c ~ normal(mu_c, sigma_c);

  // Likelihood
  vector[4] V[N]; // deck values per subject
  for (i in 1:N)
    V[i] = rep_vector(0.0, 4);

  for (t in 1:T_total) {
    int i = subj[t];  // subject index
    real outcome = Win[t] + Loss[t];
    real abs_outcome = fmax(fabs(outcome), 1e-6);

    real u = outcome < 0
      ? -w[i] * pow(abs_outcome, A[i])
      : pow(abs_outcome, A[i]);

    vector[4] logp = c[i] * V[i];
    target += log_softmax(logp)[Choice[t]];

    int d = Choice[t];
    V[i, d] += alpha[i] * (u - V[i, d]);
  }
}
