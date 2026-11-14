const methodContent = {
  linear: {
    badge: "Affine fit",
    title: "Linear calibration",
    summary:
      "Fits an intercept and slope to the labeled outcomes, then uses the calibrated predictions in AIPW.",
    good: [
      "The predictions need a shift or a change of scale.",
      "The labeled sample is small."
    ],
    tradeoffs: [
      "Estimates only an intercept and slope.",
      "Does not fit nonlinear calibration curves."
    ],
    visualNote:
      "An affine transformation shifts and rescales the score. The plot is schematic."
  },
  prognostic_linear: {
    badge: "Score + covariates",
    title: "Prognostic linear adjustment",
    summary:
      "Regresses the outcome on the prediction score and optional covariates X before AIPW. The intercept and score are unpenalized; coefficients for X use ridge regularization.",
    good: [
      "Covariates X may explain outcome variation beyond the score.",
      "The labeled sample supports estimating the additional coefficients."
    ],
    tradeoffs: [
      "Can improve precision when X adds predictive information.",
      "Requires estimating more coefficients than score-only calibration."
    ],
    visualNote:
      "The plot shows the affine transformation of the score. The fitted regression can also include covariates X."
  },
  aipw: {
    badge: "Uncalibrated",
    title: "AIPW with the original score",
    summary:
      "Uses the original predictions directly in AIPW, without fitting a calibration curve.",
    good: [
      "The predictions are already on the outcome scale.",
      "You want a baseline for comparison with calibrated estimates."
    ],
    tradeoffs: [
      "No calibration fit is needed.",
      "Miscalibration can increase the variance of the estimate."
    ],
    visualNote:
      "The score is unchanged. Optional efficiency maximization rescales it by an estimated coefficient."
  },
  sigmoid: {
    badge: "Sigmoid fit",
    title: "Sigmoid calibration",
    summary:
      "Fits a sigmoid curve before AIPW. For nonbinary outcomes, the package rescales to the labeled outcome range for fitting, then transforms back.",
    good: [
      "A sigmoid curve describes the relationship between scores and outcomes.",
      "The scores are probabilities or other bounded predictions."
    ],
    tradeoffs: [
      "Fits a smooth curve with few parameters.",
      "Can underfit calibration curves with a different shape."
    ],
    visualNote:
      "A sigmoid transformation changes the middle and extremes of the score range by different amounts. The plot is schematic."
  },
  monotone_spline: {
    badge: "Default",
    title: "Monotone spline calibration",
    summary:
      "Fits a smooth nondecreasing curve to the labeled outcomes, then uses the calibrated predictions in AIPW. This is the default method.",
    good: [
      "The relationship between scores and outcomes is nonlinear and nondecreasing.",
      "You prefer a smooth curve to a step function."
    ],
    tradeoffs: [
      "Allows more curvature than linear calibration.",
      "Smoothing can miss abrupt changes in the calibration curve."
    ],
    visualNote:
      "A smooth nondecreasing curve adjusts the score while preserving its ordering. The plot is schematic."
  },
  isotonic: {
    badge: "Step function",
    title: "Isotonic calibration",
    summary:
      'Fits a nondecreasing step function before AIPW. The default uses a one-round monotone XGBoost fit; isocal_backend="sklearn" uses scikit-learn isotonic regression.',
    good: [
      "The relationship between scores and outcomes is nonlinear and nondecreasing.",
      "The labeled sample is large enough to estimate a step function reliably."
    ],
    tradeoffs: [
      "Does not impose a smooth or parametric curve.",
      "Can be more variable in small labeled samples."
    ],
    visualNote:
      "The fitted score is constant within each step and nondecreasing across steps. The plot is schematic."
  },
  auto: {
    badge: "Cross-validation",
    title: "Automatic method selection",
    summary:
      "Compares methods by cross-validated variance estimates, then refits the selected method on all labeled observations. The default candidates are AIPW, linear, monotone spline, and isotonic calibration.",
    good: [
      "You want to compare several calibration methods using the labeled data.",
      "You can afford repeated fits for cross-validation."
    ],
    tradeoffs: [
      "Selects only among the methods in candidate_methods.",
      "Including AIPW also adds an efficiency-rescaled AIPW candidate."
    ],
    visualNote: ""
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
      "Uses an analytic standard error and a normal approximation.",
    best: "Analytic standard error",
    cost: "Low",
    when: "No resampling"
  },
  jackknife: {
    title: "Jackknife intervals",
    summary:
      "Deletes one paired fold at a time and refits the estimator, including automatic selection. Intervals use Student t critical values with V-1 degrees of freedom; the default is V=10.",
    best: "Pseudo-value standard error; Student t interval",
    cost: "10 refits by default",
    when: "Calibration is refit after each deletion"
  },
  bootstrap: {
    title: "Bootstrap intervals",
    summary:
      "Resamples observations with replacement and uses percentiles of the resulting mean estimates.",
    best: "Percentiles of bootstrap estimates",
    cost: "One refit per replicate",
    when: "Calibration is refit in each replicate"
  }
};

function updateMethod(method) {
  const content = methodContent[method];
  if (!content) return;

  document.querySelector("[data-method-badge]").textContent = content.badge;
  document.querySelector("[data-method-title]").textContent = content.title;
  document.querySelector("[data-method-summary]").textContent = content.summary;
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
      .forEach((tab) => {
        tab.classList.remove("is-active");
        tab.setAttribute("aria-pressed", "false");
      });
    button.classList.add("is-active");
    button.setAttribute("aria-pressed", "true");
    updateMethod(button.dataset.method);
  });
});

document.querySelectorAll("[data-module='inference-toggle'] .tab-button").forEach((button) => {
  button.addEventListener("click", () => {
    document
      .querySelectorAll("[data-module='inference-toggle'] .tab-button")
      .forEach((tab) => {
        tab.classList.remove("is-active");
        tab.setAttribute("aria-pressed", "false");
      });
    button.classList.add("is-active");
    button.setAttribute("aria-pressed", "true");
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
