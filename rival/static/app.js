const form = document.querySelector("#study-form");
const runButton = document.querySelector("#run-button");
const emptyState = document.querySelector("#empty-state");
const loadingState = document.querySelector("#loading-state");
const resultState = document.querySelector("#result-state");
const progressBar = document.querySelector("#progress-bar");
const loadingLabel = document.querySelector("#loading-label");
const studyView = document.querySelector("#study-view");
const validationView = document.querySelector("#validation-view");

const stages = [
  ["Compiling calibrated population…", "18%"],
  ["Routing behavior model…", "42%"],
  ["Sampling human anchor…", "67%"],
  ["Correcting estimates and scoring confidence…", "88%"],
];

const pct = value => `${(Number(value) * 100).toFixed(1)}%`;
const number = value => Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 });

function setText(selector, value) {
  document.querySelector(selector).textContent = value;
}

function createBar(choice, hybrid, synthetic, interval) {
  const row = document.createElement("div");
  row.className = "bar-row";
  const label = document.createElement("div");
  label.className = "bar-label";
  const strong = document.createElement("strong");
  strong.textContent = choice.label;
  const small = document.createElement("small");
  small.textContent = choice.description;
  label.append(strong, small);

  const track = document.createElement("div");
  track.className = "bar-track";
  const prior = document.createElement("div");
  prior.className = "bar-synthetic";
  prior.style.width = `${Math.max(1, synthetic * 100)}%`;
  const corrected = document.createElement("div");
  corrected.className = "bar-hybrid";
  corrected.style.width = `${Math.max(1, hybrid * 100)}%`;
  track.append(prior, corrected);

  const value = document.createElement("div");
  value.className = "bar-value";
  value.textContent = pct(hybrid);
  const ci = document.createElement("small");
  ci.textContent = `${pct(interval.lower)}–${pct(interval.upper)}`;
  value.append(ci);
  row.append(label, track, value);
  return row;
}

function render(data) {
  const simulation = data.simulation;
  const hybrid = data.hybrid;
  const choices = simulation.scenario.choices;
  const winner = choices.reduce((best, choice) =>
    hybrid.corrected_distribution[choice.choice_id] > hybrid.corrected_distribution[best.choice_id] ? choice : best
  );
  setText("#result-title", simulation.scenario.name);
  setText("#winner", winner.label);
  setText("#winner-share", `${pct(hybrid.corrected_distribution[winner.choice_id])} corrected preference`);
  setText("#expected-error", simulation.confidence ? simulation.confidence.expected_tvd.toFixed(3) : "—");
  const anchorRate = hybrid.human_sample_size / simulation.scenario.sample_size;
  setText("#anchor-rate", pct(Math.max(0, anchorRate)));

  const bars = document.querySelector("#bars");
  bars.replaceChildren(...choices.map(choice => createBar(
    choice,
    hybrid.corrected_distribution[choice.choice_id],
    simulation.distribution[choice.choice_id],
    hybrid.intervals[choice.choice_id]
  )));

  const confidence = simulation.confidence;
  setText("#confidence-label", confidence ? `${confidence.label[0].toUpperCase()}${confidence.label.slice(1)} confidence` : "Not assessed");
  setText("#confidence-reason", confidence ? confidence.reason : "No confidence model result.");
  const score = confidence ? Math.max(0, 1 - confidence.expected_tvd) : 0;
  setText("#confidence-score", pct(score));
  document.querySelector("#confidence-gauge").style.background = `conic-gradient(var(--lime-deep) 0deg, var(--lime-deep) ${score * 360}deg, #e4e6df ${score * 360}deg)`;

  const syntheticTvd = data.synthetic_evaluation.metrics.tvd;
  const hybridTvd = data.hybrid_evaluation.metrics.tvd;
  setText("#synthetic-tvd", syntheticTvd.toFixed(4));
  setText("#hybrid-tvd", hybridTvd.toFixed(4));
  setText("#improvement", `${pct(Math.max(0, data.improvement.relative_tvd_reduction))} lower error after the human anchor`);

  const card = data.evidence_card;
  setText("#lineage", card.lineage_hash);
  setText("#eligible-records", number(card.population.eligible_seed_records));
  setText("#effective-draws", number(card.population.effective_sample_size));
  setText("#human-observations", number(hybrid.human_sample_size));
  setText("#population-error", card.population.control_error === null ? "n/a" : card.population.control_error.toExponential(1));
  const warnings = document.querySelector("#warnings");
  warnings.replaceChildren(...card.warnings.map(message => {
    const node = document.createElement("div");
    node.className = "warning";
    node.textContent = message;
    return node;
  }));
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  runButton.disabled = true;
  emptyState.classList.add("hidden");
  resultState.classList.add("hidden");
  loadingState.classList.remove("hidden");
  let stage = 0;
  const timer = setInterval(() => {
    const [label, width] = stages[Math.min(stage, stages.length - 1)];
    loadingLabel.textContent = label;
    progressBar.style.width = width;
    stage += 1;
  }, 420);
  try {
    const response = await fetch("/api/demo/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sample_size: Number(document.querySelector("#sample-size").value),
        human_anchor_size: Number(document.querySelector("#anchor-size").value),
        scenario: {
          name: document.querySelector("#study-name").value,
          question: document.querySelector("#question").value,
          context: document.querySelector("#context").value,
        },
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Study failed");
    render(payload);
    progressBar.style.width = "100%";
    await new Promise(resolve => setTimeout(resolve, 260));
    loadingState.classList.add("hidden");
    resultState.classList.remove("hidden");
  } catch (error) {
    loadingState.classList.add("hidden");
    emptyState.classList.remove("hidden");
    emptyState.querySelector("h3").textContent = "The study could not run";
    emptyState.querySelector("p").textContent = error.message;
  } finally {
    clearInterval(timer);
    runButton.disabled = false;
  }
});

fetch("/api/health").catch(() => {
  document.querySelector(".status").textContent = "Engine unavailable";
});

function showView(name) {
  const validation = name === "validation";
  studyView.classList.toggle("hidden", validation);
  validationView.classList.toggle("hidden", !validation);
  document.querySelectorAll(".nav-item[data-view]").forEach(button => {
    button.classList.toggle("active", button.dataset.view === name);
  });
}

document.querySelectorAll(".nav-item[data-view]").forEach(button => {
  button.addEventListener("click", () => showView(button.dataset.view));
});

async function loadQualification() {
  try {
    const response = await fetch("/api/qualification");
    if (!response.ok) return;
    const data = await response.json();
    const op = data.opinionqa;
    const twin = data.twin2k;
    if (!op || !twin) return;
    setText("#op-questions", number(op.questions));
    setText("#op-personas", number(op.personas));
    setText("#op-raw", Number(op.raw_mean_tvd).toFixed(3));
    setText("#op-global", Number(op.global_history_mean_tvd).toFixed(3));
    setText("#op-calibrated", Number(op.calibrated_mean_tvd).toFixed(3));
    setText("#op-reduction", pct(op.relative_mean_tvd_reduction));
    setText("#op-win", `${pct(op.question_win_rate)} of questions improved`);
    setText("#op-scope", `${pct(op.question_win_rate_vs_global_history)} of questions also beat the global-history baseline. Scope: five-choice Pew opinion distributions; not individual fidelity.`);
    setText("#tw-twins", number(twin.twins));
    setText("#tw-questions", number(twin.questions));
    setText("#tw-direct", pct(twin.direct_llm_categorical_accuracy));
    setText("#tw-human", pct(twin.human_test_retest_categorical_accuracy));
    setText("#tw-population", pct(twin.population_mode_categorical_accuracy));
    setText("#tw-transfer", pct(twin.transfer_categorical_accuracy));
    setText("#hybrid-raw", Number(twin.raw_mean_tvd).toFixed(3));
    setText("#hybrid-corrected", Number(twin.hybrid_mean_tvd).toFixed(3));
  } catch (_) {
    // The study workflow remains usable if the static qualification artifact is absent.
  }
}

loadQualification();
