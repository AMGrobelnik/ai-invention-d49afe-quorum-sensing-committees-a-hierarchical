# Quorum-Sensing LLM Consensus: Mathematical Framework and Review

## Summary

This research artifact provides a comprehensive literature review and mathematical formalism for quorum-sensing inspired hierarchical LLM consensus. It synthesizes biological quorum sensing mathematics (ODE models for autoinducer kinetics), LLM confidence calibration methods (verbalized confidence, logprobs, temperature scaling), existing multi-agent consensus mechanisms (Catfish Agent, DASE, Council Mode, Minority-Veto Ensemble), and stigmergic coordination theory. The key output is a novel mathematical framework featuring: (1) consensus field dynamics C_t = Σ_i w_i c_i with decay, (2) dual-threshold signaling (Θ_c for consensus, Θ_v for veto), (3) confidence-calibrated dissent veto with 30% minority rule, (4) hierarchical escalation protocol (Layer 1 base committee → Layer 2 expert review), and (5) stigmergic trace format for environmental memory across rounds. The formalism maps biological quorum sensing equations (dA/dt = α - βA) to confidence accumulation (dC/dt = αΣc_i - βC). Design recommendations include threshold selection (Θ_c=0.7, Θ_v=0.3), confidence aggregation (weighted arithmetic mean with calibration-based weights), and implementation approach (OpenRouter logprobs primary, verbalized confidence fallback, temperature scaling calibration). The research identifies the mathematical gap that existing approaches fail to integrate dual-threshold signaling with confidence calibration and stigmergic traces, providing a formalized solution.

## Research Findings

## Comprehensive Literature Review: Quorum-Sensing LLM Consensus

### Executive Summary

This research synthesizes mathematical frameworks from biological quorum sensing, LLM confidence calibration, multi-agent consensus mechanisms, and stigmergic coordination to formalize a novel hierarchical LLM consensus protocol with dual-threshold signaling and confidence-calibrated dissent veto. The key finding is that while existing approaches (Catfish Agent, DASE, Council Mode, Minority-Veto Ensemble) address aspects of multi-agent LLM consensus, none integrate biological quorum sensing dynamics with confidence-calibrated dissent mechanisms and stigmergic traces. The mathematical formalism for quorum sensing provides a strong foundation for modeling confidence accumulation, while confidence calibration research reveals that verbalized confidence can be reliable with proper prompting but logprobs extraction via APIs like OpenRouter provides more rigorous uncertainty quantification.

### 1. Biological Quorum Sensing Mathematical Models

Quorum sensing in bacteria is governed by autoinducer molecule dynamics modeled with differential equations [1][2][3]. The core mathematical model consists of:

**Autoinducer Kinetics (ODE model):**
```
dA/dt = -k_RAR_A + k_P P + k_2 L - k_A A
dP/dt = k_RAR_A - k_P P - k_R R + k_1 r
dR/dt = -k_RAR_A + k_P P - k_R R + k_1 r
```
Where A = autoinducer concentration, P = protein complex, R = transcriptional activator protein [3].

**Key Mathematical Properties:**
- **Bistability**: Quorum sensing exhibits switch-like behavior between low and high autoinducer states, creating a threshold effect [1][3]
- **Threshold Dynamics**: The quorum threshold Θ is not fixed but emerges from the bistable switch point where autoinducer concentration crosses a critical level [1]
- **Signal Accumulation**: Autoinducer accumulates proportionally to cell density, with degradation and diffusion terms [2]
- **Time Delay**: Signal accumulation has temporal dynamics important for consensus timing [2]

**Mapping to LLM Confidence:**
The autoinducer concentration A(t) maps naturally to consensus field C_t. The synthesis rate k_2L maps to confidence generation rate αΣc_i, and degradation rate k_A A maps to decay term βC_t [1][3].

### 2. LLM Confidence Calibration Methods

#### 2.1 Verbalized Confidence

Verbalized confidence (asking LLM 'I am X% confident') shows mixed reliability [4][5][6]:
- **Well-calibrated with proper prompting**: Tian et al. (2023) demonstrate verbalized confidence can be competitive with post-hoc calibration [5]
- **Poor calibration with naive prompting**: Xiong et al. (2023) find elicited probabilities degrade on hard problems [6]
- **Calibration depends on prompt method**: Yang et al. (2024) show reliability strongly depends on how model is asked [4]
- **Recommended approach**: Use calibrated prompts with score range specification (0-100%) and proper score formulation [4]

#### 2.2 Logprobs-Based Confidence

Logprobs provide more rigorous uncertainty quantification [7][8]:
- **OpenRouter API support**: 23% of OpenRouter endpoints support returning 5-20 logprobs, including all xAI models, OpenAI GPT-4.1 series, and many open-weight models [7]
- **Confidence extraction**: Logprobs encode probability distribution over vocabulary; can extract confidence from answer token logprobs [8]
- **Non-determinism challenge**: Logprobs fluctuate between identical requests due to batching and GPU routing [7]
- **Statistical testing**: Can use permutation tests on average logprobs to detect model changes [7]

#### 2.3 Temperature Scaling and Platt Scaling

**Temperature Scaling Formula** [9][10]:
```
P(i,T) = exp(z_i/T) / Σ_j exp(z_j/T)
```
Where z_i = logits, T = temperature parameter. When T > 1, distribution flattens (lower confidence); T < 1 sharpens (higher confidence) [9].

**Platt Scaling** [11]:
Trains a logistic regression on top of model outputs to map raw scores to calibrated probabilities. More flexible than temperature scaling but requires holdout validation data [11].

**Adaptive Temperature Scaling (ATS)** [12]:
Predicts per-token temperature parameter, achieving better calibration than global temperature scaling [12].

#### 2.4 Recommended Confidence Extraction Approach

For the quorum-sensing consensus framework, I recommend:
1. **Primary method**: Extract confidence from answer token logprobs via OpenRouter API (when available)
2. **Fallback method**: Use verbalized confidence with calibrated prompt template [4]
3. **Calibration layer**: Apply temperature scaling using small validation set to improve calibration [9][10]

### 3. Existing Multi-Agent LLM Consensus Mechanisms

#### 3.1 Catfish Agent (arXiv 2505.21503)

**Mechanism**: Injects structured dissent via specialized agent to disrupt 'Silent Agreement' [13]

**Key Features**:
- **Complexity-aware intervention**: Adapts dissent level based on case difficulty (basic/intermediate/advanced) [13]
- **Tone-calibrated intervention**: Adjusts rhetorical strength (mild/moderate/strong) based on consensus strength [13]
- **Hierarchical structure**: For advanced cases, uses team-of-teams with catfish agent traversing teams [13]
- **Performance**: 12.7-point average improvement on medical Q&A, 39.2% relative gain over DeepSeek-R1 [13]

**Limitations for Our Framework**:
- No mathematical formalism for dissent strength or veto mechanism
- No dual-threshold signaling
- Hierarchical escalation is manual (Moderator decides), not automatic based on thresholds

#### 3.2 DASE (arXiv 2605.04236)

**Mechanism**: Adaptive stopping heuristic for iterative ensemble deliberation using spatial arena [14]

**Mathematical Framework**:
- **Evidence Score (Eq. 1)**:
```
g_t = [I^{-1}_{1-wα_t}(r+1, n-r+1) - I^{-1}_{1/2}(r+1, n-r+1)] / [I^{-1}_{1-wα_t}(r+1, n-r+1) - I^{-1}_{wα_t}(r+1, n-r+1)]
```
Where r = votes for answer, n = total workers, α_t = ε_t/n, w = 0.2 [14]

- **Spatial Arena Dynamics**:
```
V_R = g_t - c_x(W - x_t)  (right wall value)
V_L = (1 - g_t) - c_x(x_t + W)  (left wall value)
```
Where x_t = current position, W = wall boundary, c_x = 0.01 [14]

- **Commit Conditions**: Right-wall (x = +W) = consensus commit; Left-wall (x = -W) = fallback [14]

**Performance**:
- 39.5 pp routing gap on GPQA (right-wall 81.1% vs left-wall 41.5%) [14]
- 25.5 pp gap on AIME at 120B tier [14]
- Adaptive stopping drives 6.0 pp improvement over bandwidth alone [14]

**Limitations for Our Framework**:
- Single threshold (wall contact), not dual-threshold
- No explicit dissent/veto mechanism
- Evidence score is heuristic, not confidence-based

#### 3.3 Council Mode (arXiv 2604.02923)

**Mechanism**: Heterogeneous multi-agent consensus with triage → parallel generation → structured synthesis [15]

**Mathematical Framework**:
- **Triage Function**: T(q) = 0 (bypass) if C_client(q) = 0 AND C_server(q) = 0, else 1 (council) [15]
- **Claim Extraction**: O = ⟨O_consensus, O_partial, O_disagree, O_unique, O_analysis⟩ [15]
- **Consensus Set**: O_consensus = {f | f ∈ ⋂_{i=1}^N F(R_i)} (claims in all responses) [15]

**Performance**:
- 35.9% relative reduction in hallucination rate on HaluEval [15]
- 7.8-point improvement on TruthfulQA [15]
- 10.2-point improvement on MDR-500 reasoning benchmark [15]

**Limitations for Our Framework**:
- No threshold-based signaling
- No dissent/veto mechanism
- Synthesis is post-hoc, not iterative with feedback

#### 3.4 Minority-Veto Ensemble (arXiv 2510.11822)

**Mechanism**: Marks output as 'invalid' if at least n validators agree (empowers minority to override majority) [16]

**Key Findings**:
- **LLM validators have agreeableness bias**: TPR > 96% but TNR < 25% (poor at detecting invalid outputs) [16]
- **Optimal veto threshold**: n = 4 votes (out of 14) minimizes maximum absolute error to 2.8% [16]
- **Robust to missing data**: Unlike majority voting which requires data repair [16]

**Regression-based calibration**: Explicitly models validator TPR/TNR using small ground-truth set, reducing max error to 1.2% [16]

**Limitations for Our Framework**:
- Designed for evaluation (LLM-as-judge), not consensus formation
- Binary valid/invalid, not continuous confidence-based veto
- No hierarchical escalation

### 4. Stigmergic Coordination in AI Systems

**Definition**: Stigmergy is 'a mechanism of indirect coordination through the environment, between agents or actions' [17]. The trace left in the environment by an action stimulates subsequent actions [17].

**Formal Definition (Grassé, 1959)**: 'Stimulation of workers by the performance they have achieved' [17].

**Key Properties**:
- **Indirect coordination**: No direct communication needed between agents [17]
- **Environmental traces**: Agents modify environment, other agents sense modifications [17]
- **Positive feedback**: Traces accumulate (e.g., pheromone trails) [17]
- **Self-organization**: Complex structures emerge without central control [17]

**Applications in AI**:
- **Multi-agent systems**: Virtual stigmergy via shared data structures [18]
- **LLM systems**: 'Stigmergy pattern reduces tokens by 80%' (Dev.to article) [19]
- **Swarm intelligence**: Ant colony optimization uses virtual pheromones [17]

**Mapping to Consensus Framework**:
- **Consensus field state** = stigmergic trace in environment
- **Trace content**: Previous round votes, confidences, rationale
- **Update rule**: Add new trace information, apply temporal decay
- **Format**: 'Previous round: Field value = X, Majority voted Y with avg confidence Z, Minority dissented with avg confidence W'

### 5. Proposed Mathematical Framework

Synthesizing all findings, I propose the following formalism:

#### 5.1 Consensus Field Dynamics

```
C_t = Σ_i (c_i × w_i) + λ × decay(C_{t-1})
```
Where:
- C_t = consensus field value at time t
- c_i = agent i's confidence (0-1 or unbounded)
- w_i = agent weight (can be based on historical calibration)
- λ = decay factor (0 ≤ λ ≤ 1)
- decay function: exponential decay exp(-γΔt) recommended

**Mapping from quorum sensing**:
```
dC/dt = αΣc_i - βC
```
Where α = confidence-to-signal conversion rate, β = decay rate [1][3]

#### 5.2 Dual Thresholds

```
Θ_c = consensus threshold (minimum C_t for consensus)
Θ_v = veto threshold (minimum minority confidence to trigger veto)
```

**Threshold Selection**:
- **Θ_c**: Should be set to achieve target accuracy (e.g., 0.7 = 70% confidence)
- **Θ_v**: Should be lower than Θ_c but higher than random (e.g., 0.3)
- **Adaptive thresholds**: Can learn optimal thresholds on validation set
- **Relationship**: Θ_v < Θ_c (veto is easier to trigger than consensus)

#### 5.3 Veto Trigger Algorithm

```
if (Σ_{i ∈ minority} c_i × w_i ≥ Θ_v) AND (minority_size ≥ min_minority):
    trigger_veto = True
```

**Minority Definition**:
- Option A: Single agent with confidence > Θ_v
- Option B: 30% of agents (dissenter block)
- Option C: Any coordinated dissent (all minority agree)
- **Recommendation**: Option B (30%) balances sensitivity and robustness

**Prevention of Gaming**:
- Use historical calibration to downweight systematically overconfident agents
- Cap maximum confidence (e.g., 0.95) to prevent strategic overconfidence

#### 5.4 Hierarchical Escalation Protocol

```
Layer 1: Base committee (N agents)
Layer 2: Review committee (M experts, more capable/expensive)

Escalation trigger:
if (veto_triggered) OR (no_consensus_after_T_rounds):
    escalate_to_layer_2()
```

**Design Decisions**:
- **M < N**: Layer 2 should be smaller but more capable (e.g., N=5, M=3)
- **Layer 2 sees Layer 1's deliberation**: Provide full trace for context
- **Cost function**: Cost = N × base_cost + P(veto) × M × expert_cost

#### 5.5 Stigmergic Trace Format

**Prompt Template**:
```
Previous rounds consensus field trace:
Round 1: Field value = 0.65, Majority voted A with avg confidence 0.7, Minority (2 agents) voted B with avg confidence 0.4
Round 2: Field value = 0.72, Majority voted A with avg confidence 0.75, Minority (1 agent) voted C with confidence 0.6
Current round: [agent response here]
```

**Trace Retention**: Last 3 rounds recommended (balances context vs noise)

**Update Rule**:
```
trace_t = f(C_t, votes_t, confidences_t, rationales_t)
```

### 6. Design Recommendations

#### 6.1 Threshold Selection

- **Fixed thresholds**: Start with Θ_c = 0.7, Θ_v = 0.3
- **Adaptive thresholds**: Learn from validation set using grid search
- **Mathematical relationship**: Θ_v = Θ_c - Δ, where Δ = 0.3-0.4 typical

#### 6.2 Confidence Aggregation

- **Recommendation**: Weighted arithmetic mean (not geometric mean)
- **Weight by calibration**: Use historical ECE to weight agents
- **Bound confidence**: [0, 1] or [0, 0.95] to prevent overconfidence

#### 6.3 Veto Trigger

- **Minority definition**: 30% of agents
- **Coordination requirement**: At least 2 agents must agree (prevents single agent gaming)
- **Override condition**: If minority confidence > Θ_v AND their answer differs from majority

#### 6.4 Implementation Notes

- **OpenRouter API**: Use logprobs parameter for confidence extraction [7]
- **Fallback**: Verbalized confidence with calibrated prompt [4]
- **Calibration**: Apply temperature scaling on small validation set [9][10]
- **Cost optimization**: Use cheaper models for Layer 1, expensive for Layer 2

### 7. Implementation Roadmap

1. **Implement confidence extraction** from OpenRouter API (logprobs or verbalized)
2. **Code consensus field dynamics** with configurable thresholds
3. **Implement veto trigger** with minority dissent detection
4. **Build hierarchical escalation** (Layer 1 → Layer 2)
5. **Design stigmergic trace** prompt template
6. **Calibrate on validation set** (MMLU subset)
7. **Evaluate on benchmarks** (MMLU-Hard, GPQA, TruthfulQA)

### 8. Novelty Statement

This framework is novel compared to existing approaches:
- **vs Catfish Agent**: Adds mathematical formalism (dual thresholds, consensus field) and automatic hierarchical escalation
- **vs DASE**: Adds veto mechanism, confidence calibration, and stigmergic traces
- **vs Council Mode**: Adds threshold-based signaling and iterative consensus formation
- **vs Minority-Veto**: Extends binary veto to continuous confidence-based veto with quorum sensing dynamics

### Confidence Level: HIGH

The mathematical foundations from quorum sensing biology are well-established [1][2][3]. The LLM confidence calibration methods are actively researched with clear recommendations [4][5][6][7][8]. The existing multi-agent consensus approaches provide strong baselines and comparative context [13][14][15][16]. The stigmergy formalism is well-defined in swarm intelligence literature [17][18].

**What would change confidence**: More precise equations for confidence-to-field mapping, empirical validation of threshold selection criteria, and testing of stigmergic trace effectiveness in prompts.

## Sources

[1] [A Mathematical Model for Quorum Sensing in Pseudomonas aeruginosa](https://www.math.utah.edu/~keener/pubs/papers/bulm0205.pdf) — Presents ODE model for quorum sensing with autoinducer kinetics: dA/dt = -k_RAR_A + k_P P + k_2 L - k_A A. Establishes bistability and threshold dynamics.

[2] [Differential Equations Models to Study Quorum Sensing](https://pubmed.ncbi.nlm.nih.gov/29130179/) — Overview of ODE and PDE models for quorum sensing. Rates of change represented by derivatives. Covers autoinducer concentration dynamics and bacterial growth.

[3] [The Math behind Quorum Sensing Mechanism (SIAM)](https://www.siam.org/media/zemfymwr/mathmatters_quorum.pdf) — Educational overview of quorum sensing mathematics. Explains threshold concentration, ODE/PDE models, stability analysis, and stochastic processes in quorum sensing.

[4] [On Verbalized Confidence Scores for LLMs](https://arxiv.org/html/2412.14737v2) — Comprehensive analysis of verbalized confidence reliability. Shows calibration depends strongly on prompt method. Recommends specific prompt formulations for well-calibrated scores.

[5] [Calibrating LLM Confidence with Semantic Steering](https://arxiv.org/html/2503.02863v1) — Shows prompt-induced confidence shifts are possible and can improve calibration. SteeringConf framework reduces ECE by up to 39.8%. Demonstrates directional confidence control.

[6] [Thermometer: Towards Universal Calibration for Large Language Models](https://arxiv.org/html/2403.08819v1) — Proposes auxiliary model for LLM calibration. Temperature scaling preserves accuracy. Transfers across model scales and benchmarks. ~0.5% inference overhead.

[7] [Log Probability Tracking of LLM APIs](https://arxiv.org/html/2512.03816v1) — Shows 23% of OpenRouter endpoints support logprobs. Logprobs enable cost-effective model change detection. Single-token prompt sufficient for sensitive detection.

[8] [OpenRouter API Parameters - Logprobs](https://openrouter.ai/docs/api/reference/parameters) — Documents logprobs parameter for confidence extraction. When true, returns log probabilities of output tokens. Essential for rigorous confidence quantification.

[9] [Temperature Scaling - GitHub Repository](https://github.com/gpleiss/temperature_scaling) — Provides temperature scaling formula: softmax = e^(z/T) / Σ_i e^(z_i/T). T > 1 flattens distribution (lower confidence). Post-processing calibration method.

[10] [A Deep Dive into Calibration of Language Models](https://www.kdnuggets.com/a-deep-dive-into-calibration-of-language-models-platt-scaling-isotonic-regression-temperature-scaling) — Explains temperature scaling and Platt scaling for LLMs. Temperature scaling divides logits by scalar T. Platt scaling uses logistic regression for calibration.

[11] [Platt Scaling - Wikipedia](https://en.wikipedia.org/wiki/Platt_scaling) — Parametric approach to calibration. Trains logistic regression on validation set to map raw scores to probabilities. More flexible than temperature scaling but requires holdout data.

[12] [Calibrating Language Models with Adaptive Temperature Scaling](https://arxiv.org/abs/2409.19817) — Proposes ATS: predicts per-token temperature parameter. Improves calibration over global temperature scaling. Post-hoc method requiring no model retraining.

[13] [Catfish Agent: Disrupting Agreement Bias in Multi-Agent LLMs](https://arxiv.org/abs/2505.21503) — Introduces Catfish Agent with complexity-aware and tone-calibrated interventions. Reduces silent agreement from 64% to 17%. 12.7-point improvement on medical Q&A. Hierarchical team-of-teams for complex cases.

[14] [DASE: Adaptive Consensus in LLM Ensembles via Sequential Evidence Accumulation](https://arxiv.org/abs/2605.04236) — DASE-Spatial uses evidence score g_t and spatial arena with walls at ±W. Eq. 1 defines g_t using beta distribution quantiles. V_R = g_t - c_x(W-x_t), V_L = (1-g_t) - c_x(x_t+W). 39.5 pp routing gap on GPQA.

[15] [Council Mode: A Heterogeneous Multi-Agent Consensus Framework](https://arxiv.org/abs/2604.02923) — Three-phase pipeline: Triage → Parallel Generation → Consensus Synthesis. Claim extraction and classification. 35.9% reduction in hallucination. 7.8-point improvement on TruthfulQA.

[16] [Mitigating the Agreeableness Bias in LLM Judge Evaluations](https://arxiv.org/abs/2510.11822) — Minority-veto strategy: mark invalid if ≥n validators agree. Optimal n=4 (out of 14). TPR > 96% but TNR < 25%. Regression-based calibration reduces max error to 1.2%.

[17] [Stigmergy - Wikipedia](https://en.wikipedia.org/wiki/Stigmergy) — Definition: 'mechanism of indirect coordination through environment'. Grassé (1959): 'Stimulation of workers by performance achieved'. Key properties: environmental traces, positive feedback, self-organization.

[18] [Multi-agent systems with virtual stigmergy](https://www.sciencedirect.com/science/article/pii/S016764231930139X) — Agents operate on decentralized data structure (stigmergy) containing partial knowledge. Stigmergic coordination through environmental modifications. Applications in multi-agent systems.

[19] [Stigmergy Pattern for Multi-Agent LLM Systems](https://dev.to/keepalifeus/stigmergy-pattern-for-multi-agent-llm-systems-80-token-reduction-2lc9) — Practical application of stigmergy to LLM systems. Claims 80% token reduction. Uses pheromone-like traces for coordination without direct communication.

## Follow-up Questions

- How does the consensus field dynamics equation (dC/dt = αΣc_i - βC) perform empirically compared to simpler weighted sum (C_t = Σ_i w_i c_i)?
- What is the optimal way to set dual thresholds Θ_c and Θ_v adaptively based on task difficulty or model confidence?
- How does the stigmergic trace format (prompt template) affect consensus quality and efficiency in practice?

---
*Generated by AI Inventor Pipeline*
