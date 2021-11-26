## Metrics

For a longer form description and motivation of metrics, read the competition whitepaper.
We highly recommend this, for ease of reading the LaTeX!

### Notation

Competitors are provided with a training dataset
$\mathcal{D}_\textup{train}=\{(\bm{x}_i, \bm{y}_i)\}_{i = 1}^{N}$
of time-profiled ground truth trajectories (i.e., plans) $\bm{y}$ paired with high-dimensional observations $\bm{x}$ of the corresponding scenes.
Note that we will use plans and trajectories interchangeably in the following.

$\bm{y} = (s_1, \dots, s_T)$ correspond to the trajectory of a given vehicle observed through the SDG perception stack.

Each of the $T$ states $s_t$ correspond to the x- and y-displacement of the vehicle, s.t. $\bm{y} \in \mathbb{R}^{T \times 2}$.

### Confidence Scores

- Per-Plan Confidence Scores (aka Per-Trajectory Confidence Scores)

   In our task, we expect a stochastic model to accompany its $D_i$ predicted plans on a given input $\bm{x}_i$ with scalar per-plan confidence scores $c_i^{(d)}, d \in \{1, \dots D_i\}$.
   These provide an ordering of the plausibility of the various plans predicted for a given input.
   They must be non-negative and sum to 1 (i.e., form a valid distribution).

- Per--Prediction Request Uncertainty Score

    We also expect models to produce a scalar uncertainty score U corresponding to each prediction request input $\bm{x}_i$ (this is an **uncertainty** rather than a confidence for consistency with other Shifts Challenge tasks).
    Often in the codebase we refer to a per--prediction request confidence score which is simply a negation of U; C = -U.

### Standard Metrics

- Average Displacement Error (ADE)

The average displacement error measures the quality of a predicted plan $\bm{y}$ with respect to the ground-truth plan $\bm{y}^*$ as

    \text{ADE}(\bm{y}) \coloneqq \frac{1}{T} \sum_{t = 1}^T \left\lVert s_t - s^*_t \right\rVert_2,

where $\bm{y} = (s_1, \dots, s_T)$.

- Aggregated ADE and Final Displacement Error (FDE)

Stochastic models define a predictive distribution $q(\bf{y}|\bm{x}; \bm{\theta})$, and can therefore be evaluated over the $D$ trajectories sampled for a given input $\bm{x}$.

For example, we can measure an aggregated ADE over $D$ samples with

    \text{aggADE}_D(q) \coloneqq \underset{\{\bm{y}\}_{d = 1}^{D} \sim q(\bf{y} \mid \bf{x})}{\oplus} \text{ADE}(\mathbf{y}^{d}),

where $\oplus$ is an aggregation operator, e.g., $\oplus = \min$ recovers the minimum ADE ($\text{minADE}_{D}$)

We consider minimum and mean aggregation (minADE, avgADE), as well as the final displacement error (FDE)

    \text{FDE}(\bf{y}) \coloneqq \left\lVert s_T - s^*_T \right\rVert,

as well as its aggregated variants minFDE and avgFDE.

### Per-Plan Confidence-Aware Metrics

A stochastic model used in practice for motion prediction ultimately must **decide** on a particular predicted plan for a given prediction request.
We may make this decision by selecting for evaluation the predicted plan with the **highest per-plan confidence score**.

In other words, given per-plan confidence scores $c^{(d)}, d \in \{1, \dots D\}$ we select the top plan $y^{(d^{\\*})}, d^* = \underset{d}{\arg \max}\ c^{(d)}$, and measure the decision quality using ``top1'' ADE and FDE metrics, e.g.,

    \text{top1ADE}_D(q) \coloneqq \text{ADE}(\mathbf{y}^{(d^*)}).

We may also wish to assess the quality of the relative weighting of the $D$ plans with their corresponding per-plan confidence scores $c^{(d)}$. This is accomplished with a weighted metric

    \text{weightedADE}_D(q) \coloneqq \sum_{d \in D} c^{(d)} \cdot \text{ADE}(\mathbf{y}^{(d)}).

top1FDE and weightedFDE follow analogously to the above.


### Per--Prediction Request Uncertainty-Aware Metrics

We evaluate the quality of uncertainty quantification using retention-based metrics, in which the per--prediction request uncertainty scores determine retention order.

Note that each retention curve is plotted with respect to a particular metric above (e.g., we consider AUC for retention on weightedADE).

See the Performance Metrics section of the whitepaper and the [Retention Task](baselines_training.md#retention-task) section for more details.