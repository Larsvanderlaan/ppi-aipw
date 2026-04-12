const methodContent = {
  linear: {
    badge: "Simple",
    title: "Linear calibration before AIPW",
    summary:
      "Fits a straight-line recalibration map from prediction score to outcome, then runs AIPW on the recalibrated scores.",
    good: [
      "You want the simplest affine recalibration.",
      "Predictions seem useful but shifted or stretched.",
      "Your labeled sample is modest rather than large."
    ],
    tradeoffs: [
      "Simple and fast.",
      "Usually more stable than flexible nonlinear calibration.",
      "Cannot capture strongly nonlinear calibration error."
    ],
    recommendation:
      "Use when affine recalibration is adequate and you want the simplest calibrated estimator.",
    visualNote:
      "A single affine map tilts and shifts the original score before the final AIPW aggregation."
  },
  prognostic_linear: {
    badge: "Score + X",
    title: "Prognostic linear adjustment",
    summary:
      "Uses the prediction score together with optional covariates X in a semisupervised linear adjustment before AIPW. The intercept and score are unpenalized; extra covariates are ridge-regularized with tuning on the labeled sample.",
    good: [
      "You have extra covariates X that may explain residual outcome variation after conditioning on the score.",
      "You want to perform linear calibration with linear covariate adjustment.",
      "You have enough labeled data to support adjustment for X."
    ],
    tradeoffs: [
      "Clear, regression-style interpretation.",
      "Can improve on score-only linear adjustment when X carries additional prognostic information.",
      "Adjusting for many covariates can require a larger labeled sample; ridge regularization helps stabilize the fit."
    ],
    recommendation:
      "Use when the score is useful but you also have covariates worth adding in a linear adjustment.",
    visualNote:
      "Shown as the score-only affine part of the fit. The actual method can also add linear covariate adjustment through X."
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
      "Use when you want uncalibrated AIPW on the original score, optionally with empirical efficiency rescaling.",
    visualNote:
      "Raw-score AIPW leaves the score on the identity line. Efficiency maximization adds a global rescaling by lambda."
  },
  sigmoid: {
    badge: "Bounded scores",
    title: "Sigmoid calibration before AIPW",
    summary:
      "Fits a sigmoid-shaped calibration map before AIPW. For nonbinary outcomes, the package rescales outcomes into the observed labeled range, fits the sigmoid map there, and rescales back.",
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
      "Use mainly for bounded or probability-like scores when a smooth sigmoid recalibration is plausible.",
    visualNote:
      "A smooth S-shaped map compresses the extremes and expands the middle when the score behaves like a probability."
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
  "More flexible than linear calibration and sigmoid calibration.",
  "Produces smoother calibration curves than isotonic calibration.",
  "Retains much of the flexibility of isotonic calibration while often being more stable when the labeled sample is very small."
    ],
    recommendation:
      "Use as the default monotone calibrator when you want a smooth nonlinear adjustment.",
    visualNote:
      "A smooth monotone curve bends where needed, but avoids the hard plateaus of a stepwise isotonic fit."
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
      "Most flexible monotone option in the package.",
      "Can capture nonlinear score distortions that linear or sigmoid calibration miss.",
      "Less stable than simpler methods when the labeled sample is very small."
    ],
    recommendation:
      "Use when you want a nonlinear upgrade and the labeled sample is large enough to support it.",
    visualNote:
      "The fitted map is monotone and piecewise constant, so the calibration curve moves in visible steps."
  }
};

const methodCurves = {
  linear: {
    curves: [
      {
        label: "Affine calibration",
        color: "var(--accent)",
        width: 4,
        points: [
          [0.08, 0.14],
          [0.92, 0.84]
        ]
      }
    ]
  },
  prognostic_linear: {
    curves: [
      {
        label: "Score-only affine part",
        color: "var(--accent)",
        width: 4,
        points: [
          [0.08, 0.14],
          [0.92, 0.84]
        ]
      }
    ]
  },
  aipw: {
    curves: [
      {
        label: "Raw AIPW: identity",
        color: "var(--ink)",
        width: 4,
        points: [
          [0.08, 0.08],
          [0.92, 0.92]
        ]
      },
      {
        label: "Efficiency-maximized: lambda m(X)",
        color: "var(--warm)",
        width: 4,
        dasharray: "10 8",
        points: [
          [0.08, 0.06],
          [0.92, 0.72]
        ]
      }
    ]
  },
  sigmoid: {
    curves: [
      {
        label: "Sigmoid calibration",
        color: "var(--accent)",
        width: 4,
        points: Array.from({ length: 60 }, (_, index) => {
          const x = 0.06 + (0.88 * index) / 59;
          const z = (x - 0.5) * 8.5;
          const y = 0.12 + 0.76 * (1 / (1 + Math.exp(-z)));
          return [x, y];
        })
      }
    ]
  },
  monotone_spline: {
    curves: [
      {
        label: "Smooth monotone spline",
        color: "var(--accent)",
        width: 4,
        points: [
          [0.06, 0.12],
          [0.12, 0.15],
          [0.19, 0.20],
          [0.27, 0.28],
          [0.36, 0.40],
          [0.46, 0.53],
          [0.57, 0.65],
          [0.68, 0.74],
          [0.79, 0.80],
          [0.88, 0.84],
          [0.94, 0.87]
        ]
      }
    ]
  },
  isotonic: {
    curves: [
      {
        label: "Isotonic step fit",
        color: "var(--accent)",
        width: 4,
        points: [
          [0.06, 0.14],
          [0.18, 0.14],
          [0.18, 0.23],
          [0.33, 0.23],
          [0.33, 0.37],
          [0.49, 0.37],
          [0.49, 0.56],
          [0.68, 0.56],
          [0.68, 0.73],
          [0.83, 0.73],
          [0.83, 0.88],
          [0.94, 0.88]
        ]
      }
    ]
  }
};

const inferenceContent = {
  wald: {
    title: "Wald intervals",
    summary:
      "Fast analytic intervals for routine use.",
    best: "Routine analyses",
    cost: "Low compute",
    when: "You want a fast interval based on the Wald approximation"
  },
  jackknife: {
    title: "Jackknife intervals",
    summary:
      "V-fold delete-a-group intervals that refit calibration after dropping one labeled and one unlabeled fold at a time, then use the jackknife SE in a normal approximation.",
    best: "Recommended resampling check",
    cost: "Moderate compute",
    when: "You want the package's main resampling-style uncertainty check without running a large bootstrap"
  },
  bootstrap: {
    title: "Bootstrap intervals",
    summary:
      "Percentile bootstrap intervals that treat the prediction model as fixed, resample rows, and refit the calibration step inside each replicate.",
    best: "Percentile interval checks",
    cost: "Higher compute",
    when: "You specifically want percentile bootstrap intervals and can afford the extra runtime"
  }
};

function updateMethod(method) {
  const content = methodContent[method];
  if (!content) return;

  document.querySelector("[data-method-badge]").textContent = content.badge;
  document.querySelector("[data-method-title]").textContent = content.title;
  document.querySelector("[data-method-summary]").textContent = content.summary;
  document.querySelector("[data-method-recommendation]").textContent = content.recommendation;
  document.querySelector("[data-method-visual-note]").textContent = content.visualNote;

  const goodList = document.querySelector("[data-method-good]");
  const tradeoffsList = document.querySelector("[data-method-tradeoffs]");
  goodList.innerHTML = content.good.map((item) => `<li>${item}</li>`).join("");
  tradeoffsList.innerHTML = content.tradeoffs.map((item) => `<li>${item}</li>`).join("");
  renderMethodCurve(method);
}

function normalizePoint([x, y]) {
  const left = 54;
  const right = 388;
  const bottom = 228;
  const top = 24;
  const svgX = left + x * (right - left);
  const svgY = bottom - y * (bottom - top);
  return [svgX, svgY];
}

function pointsToPath(points) {
  return points
    .map((point, index) => {
      const [x, y] = normalizePoint(point);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function renderMethodCurve(method) {
  const svg = document.querySelector("[data-method-curve]");
  const legend = document.querySelector("[data-method-legend]");
  const spec = methodCurves[method];
  if (!svg || !legend || !spec) return;

  const gridXs = [54, 137.5, 221, 304.5, 388];
  const gridYs = [24, 75, 126, 177, 228];
  const gridLines = [
    ...gridXs.map(
      (x) =>
        `<line class="curve-grid-line" x1="${x}" y1="24" x2="${x}" y2="228"></line>`
    ),
    ...gridYs.map(
      (y) =>
        `<line class="curve-grid-line" x1="54" y1="${y}" x2="388" y2="${y}"></line>`
    )
  ].join("");

  const baselinePath = pointsToPath([
    [0.06, 0.06],
    [0.94, 0.94]
  ]);
  const showReference = method !== "aipw";

  const renderedCurves = spec.curves
    .map(
      (curve) => `
        <path
          d="${pointsToPath(curve.points)}"
          fill="none"
          stroke="${curve.color}"
          stroke-width="${curve.width || 4}"
          stroke-linecap="round"
          stroke-linejoin="round"
          ${curve.dasharray ? `stroke-dasharray="${curve.dasharray}"` : ""}
        ></path>
      `
    )
    .join("");

  svg.innerHTML = `
    <rect class="curve-backdrop" x="0" y="0" width="420" height="280" rx="26"></rect>
    ${gridLines}
    <line class="curve-axis" x1="54" y1="228" x2="388" y2="228"></line>
    <line class="curve-axis" x1="54" y1="228" x2="54" y2="24"></line>
    ${showReference ? `<path class="curve-reference" d="${baselinePath}"></path>` : ""}
    ${renderedCurves}
  `;

  legend.innerHTML = spec.curves
    .map(
      (curve) => `
        <span class="curve-legend-item">
          <span
            class="curve-legend-swatch"
            style="--swatch-color: ${curve.color};"
            ${curve.dasharray ? 'data-dashed="true"' : ""}
          ></span>
          <span>${curve.label}</span>
        </span>
      `
    )
    .join("");
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
  .querySelectorAll(".section, .panel-card")
  .forEach((node) => observer.observe(node));

updateMethod("monotone_spline");
updateInference("wald");
