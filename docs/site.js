const methodContent = {
  linear: {
    badge: "Simple",
    title: "Linear calibration before AIPW",
    summary:
      "Fits a straight-line calibration map from prediction score to outcome, then runs AIPW on the calibrated scores.",
    good: [
      "A linear calibration map is adequate.",
      "Predictions seem useful but shifted or stretched.",
      "The labeled sample is small and more flexible methods may overfit."
    ],
    tradeoffs: [
      "Simple and fast.",
      "Usually more stable than flexible nonlinear calibration.",
      "Cannot capture strongly nonlinear calibration error."
    ],
    recommendation:
      "Appropriate when a linear calibration map is adequate and a simple calibrated estimator is sufficient.",
    visualNote:
      "A single affine map tilts and shifts the original score before the final AIPW aggregation."
  },
  prognostic_linear: {
    badge: "Score + X",
    title: "Prognostic linear adjustment",
    summary:
      "Uses the prediction score together with optional covariates X in a semisupervised linear adjustment before AIPW. The intercept and score are unpenalized; extra covariates are ridge-regularized with tuning on the labeled sample.",
    good: [
      "Extra covariates X may explain residual outcome variation after conditioning on the score.",
      "Linear calibration with linear covariate adjustment is appropriate.",
      "The labeled sample is large enough to support adjustment for X."
    ],
    tradeoffs: [
      "Clear, regression-style interpretation.",
      "Can improve on score-only linear adjustment when X carries additional prognostic information.",
      "Adjusting for many covariates can require a larger labeled sample; ridge regularization helps stabilize the fit."
    ],
    recommendation:
      "Appropriate when the score is useful and extra covariates are worth adding in a linear adjustment.",
    visualNote:
      "Shown as the score-only affine part of the fit. The actual method can also add linear covariate adjustment through X."
  },
  aipw: {
    badge: "Baseline",
    title: "Raw-score AIPW",
    summary:
      "Uses the original prediction scores directly and applies AIPW with no calibration layer. Add `efficiency_maximization=True` if you want the package to rescale the predictor to `lambda m(X)` using empirical efficiency maximization.",
    good: [
      "The original score scale is trusted.",
      "A direct uncalibrated baseline is useful.",
      "Calibration effects need to be separated from AIPW itself."
    ],
    tradeoffs: [
      "Simple and stable.",
      "Best when the original model is already well calibrated.",
      "Can miss efficiency gains when the score is systematically mis-scaled unless you turn on efficiency maximization."
    ],
    recommendation:
      "Appropriate for uncalibrated AIPW on the original score, optionally with empirical efficiency rescaling.",
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
      "A smooth monotone calibration map is plausible.",
      "Overconfidence or underconfidence appears to have an S-shaped pattern."
    ],
    tradeoffs: [
      "More structured than isotonic calibration.",
      "Often stable with limited labeled data.",
      "Can underfit if the true calibration curve is not close to sigmoid-shaped."
    ],
    recommendation:
      "Primarily appropriate for bounded or probability-like scores when a smooth sigmoid calibration map is plausible.",
    visualNote:
      "A smooth S-shaped map can correct systematic nonlinear distortion by moderating the extremes and expanding the middle of the score range."
  },
  monotone_spline: {
    badge: "Default",
    title: "Smooth monotone calibration map before AIPW",
    summary:
      "Fits a smooth monotone spline calibration curve, then plugs the calibrated predictions into the AIPW estimator. This is the package's smooth monotone alternative to a stepwise isotonic fit.",
    good: [
      "Smooth monotone calibration is appropriate as a default.",
      "Monotone nonlinear miscalibration is expected.",
      "Smoother behavior than isotonic calibration is preferred.",
      "A middle ground between linear and isotonic calibration is useful."
    ],
    tradeoffs: [
  "More flexible than linear calibration and sigmoid calibration.",
  "Produces smoother calibration curves than isotonic calibration.",
  "Retains much of the flexibility of isotonic calibration while often being more stable when the labeled sample is small."
    ],
    recommendation:
      "Default smooth monotone calibrator for nonlinear adjustment.",
    visualNote:
      "A smooth monotone curve bends where needed, but avoids the hard plateaus of a stepwise isotonic fit."
  },
  isotonic: {
    badge: "Flexible",
    title: "Isotonic calibration before AIPW",
    summary:
      "Fits a monotone isotonic calibration curve, then plugs the calibrated predictions into the AIPW estimator. The default backend is a one-round monotone XGBoost calibrator with `min_child_weight=10`; switch to `isocal_backend=\"sklearn\"` if you want scikit-learn isotonic regression instead.",
    good: [
      "Monotone but nonlinear miscalibration is expected.",
      "Ordering is trustworthy but the numeric scale is not.",
      "The labeled sample is large enough to fit a stable flexible calibrator."
    ],
    tradeoffs: [
      "Most flexible monotone option in the package.",
      "Can capture nonlinear score distortions that linear or sigmoid calibration miss.",
      "Less stable than simpler methods when the labeled sample is very small."
    ],
    recommendation:
      "Appropriate when a more flexible nonlinear calibrator is needed and the labeled sample is large enough to support it.",
    visualNote:
      "The fitted map is monotone and piecewise constant, so the calibration curve moves in visible steps."
  },
  auto: {
    badge: "Adaptive",
    title: "Automatic method selection",
    summary:
      "Compares a candidate shortlist by cross-validated influence-function variance, then refits the selected method on the full labeled sample before the final estimate. By default the shortlist is `(\"aipw\", \"linear\", \"monotone_spline\", \"isotonic\")`.",
    good: [
      "A small interpretable shortlist is preferred over a fixed method.",
      "Extra compute is acceptable to avoid hand-picking a method.",
      "A data-adaptive selector is preferred over a single calibration family."
    ],
    tradeoffs: [
      "More compute than fitting one method directly.",
      "Only chooses among the candidate methods you provide.",
      "If `\"aipw\"` is in the shortlist, the selector also compares an efficiency-maximized AIPW candidate."
    ],
    recommendation:
      "Appropriate for data-adaptive selection across a small candidate set rather than committing to one method up front.",
    visualNote:
      "Shown as a candidate comparison rather than a single calibration map: the selector scores the shortlisted methods and then refits the winner."
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
  },
  auto: {
    curves: [
      {
        label: "Original score",
        color: "var(--ink)",
        width: 3,
        dasharray: "8 7",
        points: [
          [0.08, 0.08],
          [0.92, 0.92]
        ]
      },
      {
        label: "Linear candidate",
        color: "var(--warm)",
        width: 4,
        points: [
          [0.08, 0.14],
          [0.92, 0.84]
        ]
      },
      {
        label: "Monotone spline candidate",
        color: "var(--muted)",
        width: 4,
        points: [
          [0.06, 0.11],
          [0.18, 0.16],
          [0.31, 0.25],
          [0.46, 0.39],
          [0.61, 0.56],
          [0.76, 0.72],
          [0.92, 0.87]
        ]
      },
      {
        label: "Isotonic candidate",
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
      "Fast analytic intervals.",
    best: "Standard analyses",
    cost: "Low compute",
    when: "Fast interval based on the Wald approximation."
  },
  jackknife: {
    title: "Jackknife intervals",
    summary:
      "V-fold delete-a-group intervals that refit calibration after dropping one labeled and one unlabeled fold at a time, then use the jackknife SE in a normal approximation.",
    best: "Finite-sample checks",
    cost: "Moderate compute",
    when: "Finite-sample check; it may work better than bootstrap in practice."
  },
  bootstrap: {
    title: "Bootstrap intervals",
    summary:
      "Percentile bootstrap intervals that treat the prediction model as fixed, resample rows, and refit the calibration step inside each replicate.",
    best: "Classical resampling",
    cost: "Higher compute",
    when: "Classical resampling check with percentile intervals."
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
  const visualCard = document.querySelector(".method-visual-card");
  if (visualCard) {
    visualCard.classList.toggle("is-hidden", method === "auto");
  }

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
