const methodContent = {
  linear: {
    badge: "Simple",
    title: "Linear calibration before AIPW",
    summary:
      "Fits a straight-line recalibration map from prediction score to outcome, then runs AIPW on the recalibrated scores.",
    good: [
      "You want the simplest affine recalibration.",
      "Predictions seem useful but shifted or stretched.",
      "Your labeled calibration sample is modest rather than huge."
    ],
    tradeoffs: [
      "Easy to explain and debug.",
      "Usually more stable than flexible nonlinear calibration.",
      "Cannot capture strongly nonlinear calibration error."
    ],
    recommendation:
      "Use when you want the simplest affine recalibration and a very easy-to-explain baseline."
  },
  prognostic_linear: {
    badge: "Score + X",
    title: "Prognostic linear adjustment",
    summary:
      "Uses the prediction score plus optional covariates X in a semisupervised linear adjustment before AIPW. The intercept and score stay unpenalized; extra covariates are ridge-tuned on the labeled sample.",
    good: [
      "You want the score to stay in the model no matter what.",
      "You have extra covariates X that may explain residual outcome variation.",
      "You want a linear adjustment rather than a nonlinear calibration curve."
    ],
    tradeoffs: [
      "Clear, regression-style interpretation.",
      "More flexible than score-only linear recalibration when X matters.",
      "Needs X and X_unlabeled to realize its full benefit."
    ],
    recommendation:
      "Use when you want a simple regression-style adjustment built around the score, with optional extra covariates."
  },
  aipw: {
    badge: "Baseline",
    title: "Raw-score AIPW",
    summary:
      "Uses the original prediction scores directly and applies AIPW with no calibration layer. Add `efficiency_maximization=True` if you want the package to rescale the predictor to `lambda m(X)` using empirical efficiency maximization.",
    good: [
      "You trust the original score scale.",
      "You want a direct uncalibrated baseline.",
      "You want to separate calibration effects from AIPW itself."
    ],
    tradeoffs: [
      "Simple and stable.",
      "Best when the original model is already well calibrated.",
      "Can miss efficiency gains when the score is systematically mis-scaled unless you turn on efficiency maximization."
    ],
    recommendation:
      "Use as a baseline and sanity check. Turn on efficiency maximization when you want the package to rescale the raw score for lower variance."
  },
  sigmoid: {
    badge: "Bounded scores",
    title: "Sigmoid calibration before AIPW",
    summary:
      "Fits a sigmoid-shaped calibration map. For nonbinary outcomes, the package rescales outcomes into the observed labeled range, fits the sigmoid map there, and rescales back.",
    good: [
      "Predictions are probability-like or naturally bounded.",
      "A smooth monotone recalibration is plausible.",
      "You suspect overconfidence or underconfidence has an S-shaped pattern."
    ],
    tradeoffs: [
      "More structured than isotonic calibration.",
      "Often stable with limited labeled data.",
      "Can underfit if the true calibration curve is not close to sigmoid-shaped."
    ],
    recommendation:
      "Good robustness check for bounded scores, especially classification-style workflows."
  },
  monotone_spline: {
    badge: "Default",
    title: "Smooth monotone spline before AIPW",
    summary:
      "Fits a smooth monotone spline calibration curve, then plugs the calibrated predictions into the AIPW estimator. This is the package's smooth monotone alternative to a stepwise isotonic fit.",
    good: [
      "You want a strong smooth monotone default.",
      "You expect monotone nonlinear miscalibration.",
      "You want something smoother than isotonic calibration.",
      "You want a middle ground between linear and isotonic recalibration."
    ],
    tradeoffs: [
      "More flexible than linear calibration.",
      "Smoother and often easier to read than isotonic calibration.",
      "Still more complex than linear calibration when the labeled sample is very small."
    ],
    recommendation:
      "Start here unless you have a specific reason not to. This is the package’s recommended default."
  },
  isotonic: {
    badge: "Flexible",
    title: "Isotonic calibration before AIPW",
    summary:
      "Fits a monotone isotonic calibration curve, then plugs the calibrated predictions into the AIPW estimator. The default backend is a one-round monotone XGBoost calibrator with `min_child_weight=10`; switch to `isocal_backend=\"sklearn\"` if you want scikit-learn isotonic regression instead.",
    good: [
      "You expect monotone but nonlinear miscalibration.",
      "Ordering is trustworthy but the numeric scale is not.",
      "You have enough labeled data to fit a stable flexible calibrator."
    ],
    tradeoffs: [
      "Most flexible option in the package.",
      "Can capture nonlinear score distortions that linear or sigmoid calibration miss.",
      "Less stable than simpler methods when the labeled sample is very small."
    ],
    recommendation:
      "Use when you want a nonlinear upgrade and the labeled sample is large enough to support it."
  }
};

const inferenceContent = {
  wald: {
    title: "Wald intervals",
    summary:
      "Fast analytic intervals and the right place to start for routine package use.",
    best: "Default production workflow",
    cost: "Low compute",
    when: "You want speed, clarity, and a good first-pass interval"
  },
  bootstrap: {
    title: "Bootstrap intervals",
    summary:
      "Resampling-based intervals that treat the prediction model as fixed, resample rows, and refit the calibration step inside each replicate.",
    best: "Robustness checks and final reporting",
    cost: "Higher compute",
    when: "You want a more empirical uncertainty check and can afford the extra runtime"
  }
};

function updateMethod(method) {
  const content = methodContent[method];
  if (!content) return;

  document.querySelector("[data-method-badge]").textContent = content.badge;
  document.querySelector("[data-method-title]").textContent = content.title;
  document.querySelector("[data-method-summary]").textContent = content.summary;
  document.querySelector("[data-method-recommendation]").textContent = content.recommendation;

  const goodList = document.querySelector("[data-method-good]");
  const tradeoffsList = document.querySelector("[data-method-tradeoffs]");
  goodList.innerHTML = content.good.map((item) => `<li>${item}</li>`).join("");
  tradeoffsList.innerHTML = content.tradeoffs.map((item) => `<li>${item}</li>`).join("");
}

function updateInference(inference) {
  const content = inferenceContent[inference];
  if (!content) return;

  document.querySelector("[data-inference-title]").textContent = content.title;
  document.querySelector("[data-inference-summary]").textContent = content.summary;
  document.querySelector("[data-inference-best]").textContent = content.best;
  document.querySelector("[data-inference-cost]").textContent = content.cost;
  document.querySelector("[data-inference-when]").textContent = content.when;
}

document.querySelectorAll("[data-module='method-explorer'] .tab-button").forEach((button) => {
  button.addEventListener("click", () => {
    document
      .querySelectorAll("[data-module='method-explorer'] .tab-button")
      .forEach((tab) => tab.classList.remove("is-active"));
    button.classList.add("is-active");
    updateMethod(button.dataset.method);
  });
});

document.querySelectorAll("[data-module='inference-toggle'] .tab-button").forEach((button) => {
  button.addEventListener("click", () => {
    document
      .querySelectorAll("[data-module='inference-toggle'] .tab-button")
      .forEach((tab) => tab.classList.remove("is-active"));
    button.classList.add("is-active");
    updateInference(button.dataset.inference);
  });
});

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-revealed");
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.15 }
);

document
  .querySelectorAll(".section, .panel-card, .timeline-card, .history-callout")
  .forEach((node) => observer.observe(node));

updateMethod("monotone_spline");
updateInference("wald");
