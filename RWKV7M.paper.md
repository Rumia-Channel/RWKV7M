# RWKV系 Recurrent LLMにおける State-Level Screening

## Capacity-Calibrated Absolute Read, Confidence-Preserving Memory Editing, and Sparse Novel Allocation

**版**: design-locked-draft-v5

**対象**: RWKV-7系、または固定容量recurrent stateを持つefficient sequence model

**基礎実装**: JAX/Flax NNX、portable reference recurrence、v4 GPU/TPU Pallas recurrence

**意味論バージョン**:

```text
screening-v4-legacy
screening-v4-competitive
screening-v5-core
screening-v5-retention
```

**実装状況**:

* v4 legacyおよびcompetitive screeningはopt-in実装済み
* v4にはportable referenceとGPU/TPU Pallas pathが存在する
* 補正後v4 recurrenceはTPU v5e-4で実機parityおよび性能測定済み
* corrected v4 competitive GPU pathの実機検証は未実施
* v5 coreのPhase 1 portable referenceおよびNNX統合は実装済みである
* portable v5 coreはMI300X単基で0.185B 2,000-stepと0.3B 400-stepのfinite runを通過したが、memory quality gateは失敗した
* tracked v5 recovery configは`tied` edit、redundancy-aware victim、soft-to-hard read、hard write率feedback controller、temporary residual floor、およびScreening入力gradient分離を用いる
* v5 Pallas、v5 checkpoint redesign、v5 retentionは未実装である
* warm-up限定admission floor、self-index loss、上限write budgetはablation用に実装済みだが、tracked configではMI300X反証結果に基づき無効化される
* 短時間MI300X runではmemory residualのcollapseと高いslot redundancyが観測され、v5のmodel quality改善は未実証である

**主張の強さ**:

本稿は固定容量recurrent memoryの設計仮説および実装契約である。長文recall、memory hygiene、学習効率、wall-clock性能の改善は、matched baselineと複数seedによる実験でのみ主張される。

---

# 1. Abstract

本稿は、RWKV系recurrent language modelに固定容量のslot memoryを追加し、絶対relevanceに基づくread screening、confidence-preserving matched write、admission-controlled novel allocation、および明示的なmemory lifecycle管理を行う設計を提案する。

標準softmax attentionは候補集合内の相対重みを生成するため、すべての候補が無関係でも総和1のread massを割り当てる。一方、State-Level Screeningは各slotを独立に評価し、閾値以下のslotを完全にrejectできる。

ただし、absolute screeningを固定容量memoryへ単純に適用するだけでは不十分である。従来設計には少なくとも以下の根本問題が存在する。

1. novelty thresholdがslot数とkey次元に依存する
2. 空slotと低品質slotの意味論が分離されていない
3. matched writeとnovel allocationのwrite amplitudeが不連続になり得る
4. hard address selectionと書込み量が結合している
5. 複数slotへの曖昧な一致が強いmatched writeへ変換され得る
6. read relevanceの単純和がslot数に応じて増幅または飽和する
7. admissionおよびvictim selectionが明示的なcapacity objectiveから導出されていない
8. straight-through estimatorの実装parityと学習上の妥当性が区別されていない
9. checkpoint reverse recurrenceの代数的可逆性と数値安定性が混同されている
10. backend、意味論、benchmark provenanceがpaper-gradeに固定されていない

v5では、State-Level Screeningを固定容量のbounded online dictionaryとして再定式化し、次の機構を導入する。

* occupied slotのみを対象とするcapacity-calibrated screening
* explicit occupancy state
* bounded、non-amplifying read aggregation
* ambiguity-aware matched-write confidence
* hard addressとcontinuous write amplitudeの分離
* empty-first allocation
* age、usage、redundancy、将来利用価値を考慮するvictim policy
* interval-aware checkpoint stability budget
* versioned semantic golden vectors
* optional adaptive retentionおよびerase/write分離

中心仮説は次である。

> RWKV coreを主経路として維持しながら、固定容量memoryを絶対relevance、容量校正、疎なaddressing、連続的なmemory editingによって管理すれば、token-to-token attentionを復活させずに、irrelevant read、memory collision、stale retrieval、破壊的なslot replacementを減らせる可能性がある。

---

# 2. Scope and Non-Claims

本稿が対象とするのは、context全体を保存する外部databaseではなく、各screened layerに付加される小規模なpersistent differentiable memoryである。

```text
number of slots M:
    context lengthに依存しない

slot dimension d_slot:
    model configurationで固定

inference memory:
    O(M * d_slot)

token recurrence:
    O(M) または tiled O(M)
```

本設計は以下を自動的には保証しない。

* slotの人間可読な意味分解
* context lengthに比例する情報容量
* attentionと同等のexact retrieval
* hard routingのunbiased gradient
* wall-clock speedup
* long-context quality改善
* upstream RWKV checkpoint互換性
* production-grade multi-host scaling
* KDAまたはDeltaNetと同等の学習挙動

slotは観測可能なmemory unitではあるが、symbolic memoryまたはhuman-interpretable entity storeとはみなさない。

---

# 3. Memory as a Bounded Online Dictionary

## 3.1 State

screened layer (l) は、最大 (M) 個のslotからなるmemoryを持つ。

```text
S_t = {s_t,1, ..., s_t,M}
s_t,m in R^{d_slot}
```

各slotには次のmetadataを持たせる。

```text
o_t,m:
    occupancy bit

age_t,m:
    最後の適用writeからの経過clock

usage_t,m:
    absolute read activityのEMA

bank_m:
    short / mid / long policy class
```

retention profileでは追加で、

```text
vitality_t,m in [0, 1]
```

を持つ。

persistent projected stateは、

```text
k_read_t,m
k_write_t,m
v_t,m
```

である。

## 3.2 Online Decision Variables

各tokenでmemory policyは次を決める。

```text
read relevance:
    r_read_t,m

matched-write route:
    r_match_t,m

novel decision:
    n_t in {0, 1}

admission decision:
    a_t in {0, 1}

victim address:
    q_t,m in {0, 1}

erase mass:
    e_t,m in [0, 1]

write mass:
    w_t,m in [0, 1]
```

hard decisionは原則としてaddressに限定する。

```text
hard:
    novelか
    保存するか
    どのslotへ書くか

continuous:
    どれだけ消すか
    どれだけ書くか
```

## 3.3 Constrained Objective Interpretation

本設計は、次の制約付きonline memory problemを近似するamortized policyとして解釈する。

```text
minimize

    expected language-model loss
  + lambda_read      * irrelevant read cost
  + lambda_write     * write frequency
  + lambda_collision * memory collision
  + lambda_duplicate * slot redundancy
  + lambda_eviction  * future eviction regret
  + lambda_stability * recurrent-state instability

subject to

    number of occupied slots <= M
    read may reject all slots
    rejected writes do not alter state
    hard allocation changes at most top-k slots
    recurrent state is chunk-invariant
```

本稿は、全routing ruleがこの目的関数の厳密解であるとは主張しない。

重要なのは、各heuristicを独立した設計選択ではなく、

```text
retrieval
collision avoidance
write cost
capacity allocation
state stability
```

の近似として位置づけることである。

---

# 4. Slot Occupancy and Initialization

## 4.1 Explicit Occupancy

各slotにhard occupancy stateを持たせる。

```text
o_t,m in {0, 1}
```

初期状態は、

```text
o_0,m = 0
s_0,m = 0
k_read_0,m = 0
k_write_0,m = 0
v_0,m = 0
age_0,m = 0
usage_0,m = 0
```

とする。

空slotはreadおよびmatched writeの候補にならない。

```text
read_rel_t,m *= o_{t-1,m}
write_eligibility_t,m *= o_{t-1,m}
```

これにより、zero slotとの偶然一致、unit normalizationの不定性、warm-up用tiny update floorへの依存を減らす。

## 4.2 Empty-First Allocation

novel writeでは、occupied slotをevictする前にempty slotを使用する。

```text
if any empty slot exists in selected bank:
    choose empty slot
else:
    choose occupied victim
```

empty allocationはevictionとして数えない。

```text
allocation_count += 1
eviction_count += occupied_before_write
```

empty slotへの初回write後、

```text
o_t,m = 1
age_t,m = 0
usage_t,m = 0
```

とする。

## 4.3 Deterministic Tie-Breaking

hard argmaxが同値の場合、backend間で結果が変わらないよう、tie-break contractを固定する。

既定は、

```text
lowest slot index wins
```

とする。

random tie-breakingは独立ablationとし、semantic parity testでは使用しない。

---

# 5. Absolute Read Screening

## 5.1 Query, Key, and Value

現在hidden stateを (x_t)、RWKV core出力を (h^{base}_t) とする。

```text
q_read_t = W_q_read LN(x_t)

k_read_t,m = persistent read-key state
v_t,m      = persistent value state
```

queryとkeyはfloat32でunit normalizationする。

```text
q_hat_t = q_read_t / max(||q_read_t||_2, eps_norm)
k_hat_t,m = k_read_t,m / max(||k_read_t,m||_2, eps_norm)

sim_t,m = <q_hat_t, k_hat_t,m>
```

empty slotでは、

```text
sim_t,m = -infinity
```

として扱う。

## 5.2 Safe Threshold Parameterization

従来の、

```text
tau = 2 sigmoid(theta) - 1
```

は、(\tau \to 1) のときTrim-and-Squareの分母および勾配を不安定にする。

v5では、

```text
tau =
    tau_min
    + (tau_max - tau_min) * sigmoid(theta)

-1 < tau_min < tau_max < 1
```

とする。

`tau_max`は必ず、

```text
tau_max <= 1 - delta_tau
```

を満たす。

`eps`だけに依存して(\tau \to 1)を許可しない。

## 5.3 Capacity-Calibrated Threshold

slotごとのfalse positive rateが一定でも、slot数が増えると「少なくとも一つの無関係slotを読む確率」は増加する。

そこで、read thresholdは、

```text
number of occupied slots
key dimension
number of read tiles
target family-wise false-read rate
```

に対して校正する。

null similarityのCDFを $F_{d_{\mathrm{test}}}$ とすると、analytic initializationは、

$$
\tau^{\mathrm{base}}_{\mathrm{read}}
=
F_{d_{\mathrm{test}}}^{-1}
\left(
(1-\delta_{\mathrm{read}})^{1/N_{\mathrm{tests}}}
\right)
$$

とする。

このfamily-wise式はnull similarity testの独立性を仮定する。tile間またはslot間に
相関がある場合は初期化近似としてのみ使用し、empirical false-read curveを
最終calibrationの根拠とする。

```text
N_tests =
    max(1, occupied_slot_count * n_read_tiles)
```

一回のsimilarity testに使うkey次元を $d_{\mathrm{test}}$ とする。single-readでは
$d_{\mathrm{test}}=d_k$、multi-readでは
$d_{\mathrm{test}}=d_{k,\mathrm{tile}}$ である。高次元unit vectorのGaussian null
近似では、family-wise CDFを

$$
q = (1-\delta_{\mathrm{read}})^{1/N_{\mathrm{tests}}}
$$

として、

$$
\tau^{\mathrm{base}}_{\mathrm{read}}
\approx
\frac{\Phi^{-1}(q)}{\sqrt{d_{\mathrm{test}}}}
$$

を用いる。より緩い上界として、

$$
\tau^{\mathrm{base}}_{\mathrm{read}}
\approx
\sqrt{
\frac{
2\log(N_{\mathrm{tests}}/\delta_{\mathrm{read}})
}{
d_{\mathrm{test}}
}
}
$$

も書けるが、小さいtile次元と多いtest数では過度に保守的になるため、実装の
既定値には用いない。実際、$N_{\mathrm{tests}}=64$、$d_{\mathrm{test}}=16$、
$\delta=0.05$では上界が0.946、Gaussian CDF近似が0.789となる。前者はtracked
MI300X runのread starvationを直接誘発し得る値だった。

ただし、学習後のkey分布は等方的ではないため、analytic thresholdを保証として扱わない。

実装では、

```text
tau_read =
    clip(
        tau_base(M_occupied, d_test, n_tiles)
        + learned_tau_offset,
        tau_min,
        tau_max
    )
```

とする。

headline configurationでは、固定thresholdだけでなく、shuffled query-key pairまたはknown-nonmatch pairによるempirical false-read curveを報告する。

## 5.4 Trim-and-Square

absolute relevanceは、

```text
trim_t,m =
    relu(
        (sim_t,m - tau_read)
        / (1 - tau_read)
    )

rel_raw_t,m =
    trim_t,m^2
```

とする。

```text
sim <= tau:
    relevance = 0

sim = 1:
    relevance = 1
```

read relevanceはslot間softmaxで正規化しない。

## 5.5 Read Dead-Zone

Trim-and-Squareは閾値以下でquery、key、thresholdへの勾配をゼロにする。

これはhard rejectionの意味論としては正しいが、初期学習で全slotがrejectされるとmemory branchが学習できない。

v5では、以下のいずれかを明示的に選択する。

```text
hard_read:
    forwardもbackwardもTrim-and-Square

shadow_surrogate:
    forwardはTrim-and-Square
    backwardはsmooth surrogate

threshold_warmup:
    初期tauを低くし、学習中に校正値へ移行
```

初期設計の既定は、

```text
hard_read + threshold_warmup
```

だった。しかしMI300X短時間runでhard readが消失したため、tracked recovery
profileではtraining中だけ、

```text
smooth forward / smooth backward
    -> anneal
hard forward / hard backward
```

へ移行する。smooth relevanceは正規化座標
$x=(sim-\tau)/(1-\tau)$に対するsquared softplusとし、slot間で総和1へ正規化
しない。deterministic evaluationとinferenceは常にhard readを使う。この
curriculumはtraining semanticsを一時的に変更するため、hard-only ablationと
区別して報告する。

shadow surrogateは独立ablationとし、hard readと同じモデルとして報告しない。

## 5.6 Occupancy and Optional Vitality

v5 coreでは、

```text
rel_t,m =
    o_{t-1,m} * rel_raw_t,m
```

とする。

retention profileでは、

```text
rel_t,m =
    o_{t-1,m}
    * vitality_pre_t,m^rho_read
    * rel_raw_t,m
```

とする。

vitalityはrelevanceを増幅せず、抑制方向にのみ作用する。

## 5.7 Bounded Non-Amplifying Aggregation

単純な、

```text
sum_m rel_m unit(v_m)
```

は、関連slot数が増えたときvector normが増大し、TanhNormやresidual branchを飽和させ得る。

一方、通常の平均やsoftmax normalizationは、弱い総relevanceを強いreadへ増幅し得る。

v5では、弱いread vectorを増幅せず、合成後のvector normが1を超える場合だけ抑える。

```text
value_hat_t,m =
    v_t,m / max(||v_t,m||_2, eps_norm)

z_sum_t =
    sum_m rel_t,m * value_hat_t,m

read_energy_t =
    ||z_sum_t||_2^2

read_denom_t =
    sqrt(
        1 + relu(read_energy_t - 1)
    )

z_t =
    z_sum_t / read_denom_t
```

このとき、

```text
read_energy <= 1:
    normalizationなし

read_energy > 1:
    ||z_t||_2 = 1
```

となる。

これはslot probability normalizationではなく、合成vectorに対するnorm clippingである。
したがって、value vectorの向きが揃っている場合を含め、slot数に依存せず
`||z_t||_2 <= 1` が成立する。

全slotがrejectされた場合、

```text
z_t = 0
```

を維持する。

## 5.8 TanhNorm

必要に応じて、

$$
\operatorname{TanhNorm}(z)
=
\tanh(\lVert z\rVert_2)
\frac{z}{\lVert z\rVert_2+\epsilon}
$$

を適用する。

```text
u_t = TanhNorm(z_t)
```

TanhNormの有無はablationとする。

## 5.9 Value-Space Gate and Residual Fusion

```text
g_value_t =
    sigmoid(
        W_gate LN(x_t) + b_gate
    )

read_out_t =
    W_out(
        u_t * g_value_t
    )
```

screening residualは、

```text
h_t =
    h_base_t
    + lambda_effective * read_out_t
```

とする。

multi-read時は、

```text
lambda_effective =
    softplus(lambda_raw)
    / sqrt(n_read_tiles)
```

とする。

screened layer数による一律除算は既定にしない。

代わりに、各layerについて次を記録する。

```text
screening residual RMS
base residual RMS
screening/base RMS ratio
gradient RMS
```

全screened layerの残差エネルギーが増大する場合のみ、depth scalingまたはresidual budget regularizationを比較する。

---

# 6. Write Candidate

## 6.1 Factorized Candidate

candidateはslotごとに生成する。

```text
x_latent =
    W_x LN(x_t)

h_latent =
    W_h h_base_t

slot_latent_m =
    W_e e_m

delta_s_t,m =
    tanh(
        W_candidate_out
        silu(
            x_latent
            + h_latent
            + slot_latent_m
        )
    )
```

rankは、

```text
32 / 64 / 128
```

を比較する。

## 6.2 Projection Consistency

persistent projected candidateは、同じcandidate slotから計算する。

```text
delta_k_read_t,m =
    P_k_read delta_s_t,m

delta_k_write_t,m =
    P_k_write delta_s_t,m

delta_v_t,m =
    P_v delta_s_t,m
```

slot candidateとprojected candidateを独立networkで生成してはならない。

これにより、

```text
persistent projected state
=
projection of recurrent slot state
```

という意味論を維持する。

## 6.3 Parameter and FLOP Contract

factorizationを採用する条件は、parameter数だけでなく、主要MACsも減少することである。

報告する値は次とする。

```text
parameter count
theoretical MACs
activation bytes
parameter bytes
projection latency
end-to-end step latency
```

FLOPs削減だけでwall-clock改善を主張しない。

rankごとに、

```text
legacy candidate
factorized rank 32
factorized rank 64
factorized rank 128
```

を同一shapeで比較する。

---

# 7. Capacity-Calibrated Write Matching

## 7.1 Separate Write Query

read queryとwrite queryは共有しない。

```text
q_write_t =
    W_q_write
    LN(
        concat(
            x_t,
            h_base_t
        )
    )
```

write keyは、

```text
k_write_t,m
```

を使用する。

```text
sim_write_t,m =
    <unit(q_write_t), unit(k_write_t,m)>
```

empty slotはmatched write候補から除外する。

## 7.2 Absolute Eligibility

```text
eligibility_t,m =
    o_{t-1,m}
    * TrimSquare(
        sim_write_t,m,
        tau_write
    )
```

`tau_write`もoccupied slot数およびkey次元に応じて校正する。

read thresholdとwrite thresholdは共有しない。

## 7.3 Competitive Route

eligible slot間のrouteは、

```text
route_power_t,m =
    eligibility_t,m^gamma

route_denom_t =
    where(
        sum_j route_power_t,j > 0,
        sum_j route_power_t,j,
        1
    )

p_t,m =
    route_power_t,m
    / route_denom_t
```

とする。

ただし、`p`はaddress distributionであり、総write massではない。

## 7.4 Ambiguity-Aware Confidence

従来の、

```text
confidence =
    max_m eligibility_m
```

だけでは、複数の類似slotが存在する曖昧な状態でも強いmatched writeが発生し得る。

v5ではroute concentrationを考慮する。

```text
c_abs_t =
    max_m eligibility_t,m

route_concentration_t =
    sum_m p_t,m^2
```

`route_concentration`は、

```text
one clear slot:
    1に近い

K個へ均等:
    1/Kに近い
```

となる。

effective matched confidenceは、

```text
if c_abs_t == 0:
    c_match_t = 0

elif eta_ambiguity == 0:
    c_match_t = c_abs_t

else:
    c_match_t =
        c_abs_t
        * route_concentration_t^eta_ambiguity
```

とする。

この分岐により、eligible slotがない場合の`0^0`を意味論から排除する。

これにより、

```text
weak match
diffuse match
duplicate-slot ambiguity
```

が強いwriteへ変換されることを抑制する。

`eta_ambiguity=0`でv4 behaviorへ戻る。

## 7.5 Capacity-Calibrated Novelty

noveltyは固定thresholdではなく、occupied slot数およびkey次元を考慮して判定する。

```text
novel_hard_t =
    c_match_t
    < tau_novel(
        M_occupied,
        d_k,
        target_false_match_rate
    )
```

soft surrogateは、

```text
novel_soft_t =
    sigmoid(
        (
            tau_novel
            - c_match_t
        )
        / novelty_temperature
    )
```

とする。

straight-through形式は、

```text
novel_st_t =
    novel_soft_t
    + stop_gradient(
        novel_hard_t
        - novel_soft_t
    )
```

とする。

固定thresholdはlegacy modeとして残す。

headline v5 configurationでは、少なくとも、

```text
M = 16 / 64 / 256
d_k = 32 / 64 / 128
```

におけるfalse-match curveを報告する。

---

# 8. Admission

## 8.1 Novelty Is Not Utility

既存slotと一致しないことは、保存価値を意味しない。

novel tokenはadmission policyを通す。

```text
admission_soft_t =
    sigmoid(
        admission_logit(
            x_t,
            h_base_t,
            c_match_t,
            memory_load,
            bank_load
        )
    )
```

```text
admission_hard_t =
    admission_soft_t
    >= admission_threshold
```

```text
admission_st_t =
    admission_soft_t
    + stop_gradient(
        admission_hard_t
        - admission_soft_t
    )
```

## 8.2 Hard Write-Rate Feedback Control

write率をloss gradientで制御すると、soft surrogateだけを満たすGoodhart解、
hard thresholdによる吸収状態、および急峻な二次penaltyが生じ得る。したがって
tracked profileでは、write率とcontent selectionの役割を分離する。

```text
rate control:
    per-layer hard accepted-novel rateを観測するfeedback controller

content selection:
    CE gradientによるtoken rankingとmemory content学習
```

各screened layerについて、

```text
e_t = target_rate - hard_write_rate_t

delta_bias_t =
    clip(
        kp * (e_t - e_{t-1})
        + ki * e_t,
        -max_step,
        +max_step
    )

admission_bias_{t+1} =
    clip(
        admission_bias_t + delta_bias_t,
        -bias_limit,
        +bias_limit
    )
```

とする。controller biasはloss内でstop-gradientし、optimizerのgradientや
Adam momentへ率制御を混入させない。controllerはscreening activation後だけ更新し、
non-finite観測は無視する。hard実測率を使うため、soft量だけで制約を満たす経路はない。

tracked profileのtargetは0.05であるが、最適値とは主張しない。初期
`admission_init=0.16`は、route scoreを近似的なunit normalと見たときに5% tailへ
近いlogit biasを与えるheuristicであり、実shapeでの初期hard率を必ず記録する。

controller biasは小さい制御incrementをBF16丸めで失わないFP32 portable model
parameterとしてexportする。rate EMAと前回errorは
学習再開用stateとしてfull checkpointへ保存し、portable safetensorsには含めない。

従来のupper write-budget lossはablation用に残すが、controllerとの同時有効化を
config validationで拒否する。低utilization時のbudgetはhard disableせず、utilizationに
比例して連続的にscaleする。

write率だけではmemory利用を証明しない。常に、

```text
per-layer/bank hard write rate
slot utilization
read energy
residual/base RMS
memory-on/off causal delta
```

を併記する。

## 8.3 Warm-up-Limited Admission Floor

上限write budgetはalways-writeを抑制するが、never-writeには罰則を与えない。
Phase 1の旧opt-in curriculumとして、空容量が残る初期期間だけsoft admissionへ下限を置く。

```text
remaining_empty_fraction_t =
    1 - mean_m(o_t,m)

target_min_rate(step) =
    initial_min_rate
    * max(
        1 - step / admission_floor_steps,
        0
      )
    * stop_gradient(
        remaining_empty_fraction_t
      )
```

```text
L_admission_floor =
    lambda_floor
    * relu(
        target_min_rate(step)
        - mean_t(
            novel_soft_t
            * admission_soft_t
          )
      )^2
```

この補助損失は、warm-up終了後またはmemory満杯時に厳密に0とする。恒久的なwrite quota、
固定最適write rate、memory利用の証拠として扱わない。headline比較では、floorなし、floorあり、
およびmemory branch counterfactualを分けて報告する。

MI300Xでは、lower/upper lossが安定した中間hard write率を作らず、soft/hard差と
layer平均による退化解を許した。従ってtracked profileではfloor/budget weightを0とし、
§8.2のfeedback controllerを使用する。この節の損失は比較ablationである。

同じwarm-up区間では、`lambda_screen`にもannealする非負下限を設定できる。これは初期に
residual scaleだけを0へ落とす退化解を抑えるためのcurriculumであり、warm-up終了後はlearned
scaleだけを使用する。gate biasは飽和初期化せず、read RMS、base RMS、両者の比、learned scale、
適用中のfloorを記録する。

## 8.4 Temporary Self-Index Curriculum (Ablation)

empty-first allocationだけでは、書き込まれたcandidate keyが、そのwriteを
発生させたqueryから再検索可能であることを保証しない。query/key geometryが
未学習のままでは、全tokenがnovelとなりslot置換だけが継続し得る。

新規writeをhard-forwardで受理したtokenについて、選択されたcandidateの
write keyとread keyへ一時的なmargin lossを課す。

```text
target_write = tau_novel_similarity + margin
target_read_tile = tau_read_tile + margin

L_self_index =
    relu(target_write - sim(q_write, candidate_write_key))^2
    + mean_tile(
        relu(target_read_tile - sim(q_read_tile, candidate_read_key_tile))^2
      )
```

admission、bank、victim addressはstop-gradientしたhard decisionとして扱い、
query/key/candidate projectionへだけ幾何学習信号を流す。これによりloss回避の
ためにadmissionを下げる経路を作らない。係数はwarm-up中にゼロへannealし、
恒久的な同一query再構成目的にはしない。

ただし、MI300Xのself-index-only runではraw gradient L2が最大5.56e11となり、
直後にほぼ全拒否へcollapseした。従って現tracked profileではこのlossを無効化する。
content学習はfeedback controllerが確保したwriteとtemporary residual floorを介する
CEを主信号とする。self-indexを再採用するには、candidate norm境界を含むbounded
gradient契約と独立ablationが必要である。

---

# 9. Hard Address and Continuous Memory Editing

## 9.1 Address and Amplitude Separation

v5では、hard routingを離散decisionに限定し、accepted writeのcontent editing量は
continuousに保つ。

```text
hard:
    matchedかnovelか
    admissionするか
    selected bank
    selected allocation slot

continuous:
    erase mass
    write mass
```

noveltyまたはadmission境界を越えただけで、accepted writeのerase/write massを
自動的に1へ固定する設計を既定にしない。

## 9.2 Matched Route

matched address massは、

```text
matched_address_t,m =
    (1 - novel_st_t)
    * c_match_t
    * p_t,m
```

とする。

## 9.3 Erase and Write Gates

matched writeの各slotについて、

```text
erase_gate_t,m =
    sigmoid(
        erase_logit_t,m
    )

write_gate_t,m =
    sigmoid(
        write_logit_t,m
    )
```

を計算する。

安定性を優先する既定modeは、

```text
capacity_conserving
```

とする。

```text
erase_mass_t,m =
    base_mass_t,m
    * erase_gate_t,m

write_mass_t,m =
    erase_mass_t,m
    * write_gate_t,m
```

したがって、

```text
0 <= write_mass <= erase_mass <= 1
```

が成立する。

比較modeは次とする。

```text
tied:
    erase_mass = write_mass = base_mass

capacity_conserving:
    write_mass <= erase_mass

free_edit:
    eraseとwriteを独立予測
```

`free_edit`はnorm drift監視なしにheadline configurationへ使用しない。

## 9.4 Matched Update

matched writeでは、

```text
base_mass_t,m =
    mu_bank(m)
    * matched_address_t,m
```

とする。

slot updateは、

```text
s_t,m =
    (
        1 - erase_mass_t,m
    )
    * s_{t-1,m}
    +
    write_mass_t,m
    * delta_s_t,m
```

とする。

projected statesも同じscalarで更新する。

```text
k_read_t,m =
    (
        1 - erase_mass_t,m
    )
    * k_read_{t-1,m}
    +
    write_mass_t,m
    * delta_k_read_t,m
```

```text
k_write_t,m =
    (
        1 - erase_mass_t,m
    )
    * k_write_{t-1,m}
    +
    write_mass_t,m
    * delta_k_write_t,m
```

```text
v_t,m =
    (
        1 - erase_mass_t,m
    )
    * v_{t-1,m}
    +
    write_mass_t,m
    * delta_v_t,m
```

## 9.5 Exact Reduction to the v4 Content Update

次の設定でv4のcontent update recurrenceへ代数的に還元される。

```text
eta_ambiguity = 0
edit_mode = tied
capacity_calibration = fixed
occupancy = all occupied
```

このとき、

```text
erase_mass =
    write_mass =
    mu_bank * matched_route
```

となる。

これはcontent updateの還元条件であり、query/key target、threshold、novelty、admission、
bank routeを含む完全なv4互換性には、それらも`screening-v4-competitive`契約へ固定する必要がある。
`capacity_conserving` modeのsigmoid gate biasを大きい正値にするだけでは近似にすぎず、
exact compatibilityとは呼ばない。

---

# 10. Sparse Novel Allocation

## 10.1 Hierarchical Bank Route

bank routeは、

```text
bank_soft_t =
    softmax(
        bank_logit_t
        / bank_temperature
    )
```

```text
bank_hard_t =
    one_hot(
        argmax(bank_logit_t)
    )
```

```text
bank_st_t =
    bank_soft_t
    + stop_gradient(
        bank_hard_t
        - bank_soft_t
    )
```

とする。

forwardでは1 bankを選択する。

## 10.2 Empty Slot Selection

selected bankにempty slotがある場合、occupied victim scoringを使用しない。

bankごとのempty slot maskは、

```text
empty_mask_b,m =
    1[bank_m == b]
    * (1 - o_{t-1,m})
```

とする。`has_empty_b = any_m(empty_mask_b,m)`の場合、

```text
empty_hard_b =
    one_hot(
        lowest empty slot index in bank b
    )
```

を使用する。empty address自体にはsoft victim scoreを使用しないが、どのbankを選ぶかの
surrogate gradientは`bank_st`を通して維持する。

## 10.3 Victim Score

empty slotがない場合のみvictim scoreを計算する。

```text
victim_score_t,m =
      age_weight
      * normalized_age_t,m

    - usage_weight
      * normalized_usage_t,m

    - vitality_weight
      * vitality_pre_t,m

    - predicted_use_weight
      * predicted_future_use_t,m

    + redundancy_weight
      * redundancy_t,m
```

高いscoreほどeviction候補とする。

v5 coreでvitalityを使用しない場合、

```text
vitality_t,m = 1
```

として項を無効化する。

## 10.4 Redundancy

duplicate slotはcapacityを消費し、曖昧なmatched routeを生む。

slot redundancyは例えば、

```text
redundancy_t,m =
    max_{j != m}
    cosine(
        unit(k_write_t,m),
        unit(k_write_t,j)
    )
```

とする。

(O(M^2))計算を避ける場合、

```text
periodic update
sampled pairs
bank-local pairs
low-rank Gram approximation
```

を用いる。

redundancyはread relevanceではなくvictim protectionにのみ使用することを既定とする。

## 10.5 Predicted Future Use

ageとusageは過去利用のみを表し、将来利用価値を直接表現しない。

optional eviction criticは、

```text
predicted_future_use_t,m =
    sigmoid(
        utility_critic(
            slot_summary_t,m,
            age_t,m,
            usage_t,m,
            bank_m
        )
    )
```

を予測する。

training targetは、slotがoverwriteされるまでの範囲で、

```text
future_read_target_t,m =
    max_{u in [t+1, t+H]}
        stop_gradient(
            rel_u,m
        )
```

とする。

```text
L_eviction_critic =
    binary_cross_entropy(
        predicted_future_use,
        future_read_target
    )
```

このcriticはoptionalであり、v5 coreの成立条件ではない。

ただし、age/usage heuristicのみを最終方式として理論的最適と主張しない。

## 10.6 Empty-First Allocation Route

empty slotがないbankについてのみ、occupied slot間で、

```text
slot_soft_b =
    masked_softmax(
        victim_score
        / victim_temperature,
        bank=b
    )
```

```text
slot_hard_b =
    one_hot(
        argmax(
            victim_score within bank b
        )
    )
```

```text
victim_st_b =
    slot_soft_b
    + stop_gradient(
        slot_hard_b
        - slot_soft_b
    )
```

とする。

bank内allocation addressは、

```text
allocation_st_b =
    where(
        has_empty_b,
        stop_gradient(empty_hard_b),
        victim_st_b
    )
```

とする。最終allocation addressは、

```text
allocation_st_t,m =
    sum_b
        bank_st_b
        * allocation_st_b,m
```

となる。

metadata更新に使うhard-forward addressは、同じ分岐の`bank_hard`、`empty_hard`、
`slot_hard`から`allocation_hard_t,m`として構成する。

## 10.7 Novel Write Amplitude

novel addressはhardだが、amplitudeはcontinuousとする。

```text
novel_base_t,m =
    novel_st_t
    * admission_st_t
    * allocation_st_t,m
```

hard metadata eventは、

```text
accepted_novel_hard_t,m =
    novel_hard_t
    * admission_hard_t
    * allocation_hard_t,m
```

とする。

novel erase/write gateを、

```text
novel_erase_gate_t =
    sigmoid(novel_erase_logit_t)

novel_write_gate_t =
    sigmoid(novel_write_logit_t)
```

とする。occupied victimへのeraseは、

```text
novel_erase_mass_t,m =
    o_{t-1,m}
    * novel_base_t,m
    * novel_erase_gate_t
```

既定の`capacity_conserving` modeでは、writeを、

```text
novel_write_mass_t,m =
    novel_base_t,m
    * novel_write_gate_t
    * (
        (1 - o_{t-1,m})
        + o_{t-1,m}
          * novel_erase_gate_t
    )
```

とする。

したがってoccupied victimでは、

```text
novel_write_mass <= novel_erase_mass
```

となる。empty slotではeraseすべき旧contentがないため、

```text
novel_erase_mass = 0
novel_write_mass =
    novel_base * novel_write_gate
```

とする。`tied` modeでは`erase_mass=o_prev*novel_base`、
`write_mass=novel_base`とする。`free_edit` modeだけがoccupied victimでも
erase/write gateを独立に適用できる。

novel replacementを完全置換に近づける場合でも、addressとamplitudeは別に保つ。

## 10.8 Final Write

```text
erase_mass_final_t,m =
    matched_erase_mass_t,m
    + novel_erase_mass_t,m
```

```text
write_mass_final_t,m =
    matched_write_mass_t,m
    + novel_write_mass_t,m
```

matchedとnovelはhard-forwardで排他的なので、同一tokenで両方が同じslotへ適用されない。

---

# 11. Accounting Semantics

slot contentとwrite metricsはraw eligibilityではなく、実際に適用されたupdateから更新する。
ageはaccepted allocationまたはapplied updateでresetし、occupancyはaccepted hard allocationで更新する。

## 11.1 Applied Write

```text
applied_mass_t,m =
    max(
        erase_mass_final_t,m,
        write_mass_final_t,m
    )

accounted_write_hard_t,m =
    applied_mass_t,m
    >= write_accounting_floor
```

`write_accounting_floor`は明示的なsemantics fieldとし、backendごとのmachine epsilonへ
暗黙に依存させない。continuous massの統計はfloor適用前、write countとage resetは
`accounted_write_hard`から計算する。

## 11.2 Age

```text
if accepted_novel_hard_t,m == 1:
    age_t,m = 0

elif o_{t-1,m} == 0:
    age_t,m = 0

elif accounted_write_hard_t,m == 1:
    age_t,m = 0

else:
    age_t,m =
        age_{t-1,m}
        + delta_clock_t
```

rejectされたtokenはageをresetしない。

## 11.3 Usage

usageはabsolute read activityから更新する。

```text
usage_signal_t,m =
    max_over_tiles(
        rel_t,m,tile
    )
```

```text
usage_t,m =
    usage_decay
    * usage_{t-1,m}
    +
    (
        1 - usage_decay
    )
    * usage_signal_t,m
```

write eligibilityをusageとして数えない。

## 11.4 Occupancy

```text
o_t,m =
    o_{t-1,m}
    OR
    accepted_novel_hard_t,m
```

occupancyはcontinuous write amplitudeではなく、hard-forward allocation eventを表す。
これにより、sub-threshold content updateをunoccupied slotへ残す矛盾を避ける。
matched writeはempty slotへ発生しないため、occupancyを新規に立てない。

## 11.5 Metrics

最低限、次を別々に記録する。

```text
matched address mass
matched erase mass
matched write mass

novel decision rate
admission rate
accepted novel rate

empty allocation count
occupied eviction count

bank selection count
slot selection count

write reject count
write saturation count
write accounting floor
```

---

# 12. Multi-Timescale Banks

slotは、

```text
short
mid
long
```

のpolicy classへ分ける。

bankは意味カテゴリではなく、memory lifecycle priorである。

```text
short:
    high plasticity
    short retention prior
    low eviction protection

mid:
    balanced plasticity
    medium retention prior

long:
    low plasticity
    long retention prior
    high eviction protection
```

初期update-rate ceilingの例は、

```text
mu_short_max = 0.05
mu_mid_max   = 0.02
mu_long_max  = 0.005
```

とする。

これらは理論値ではなく初期探索点である。

`mu_*_max`はmatched editのceilingである。novel allocationにbank別ceilingを
導入する場合は別fieldとして明示し、matched `mu`へ暗黙に連結しない。

bank collapseが観測される前から強いquota regularizerを導入しない。

観測する指標は、

```text
bank occupancy
bank allocation rate
bank eviction rate
bank read mass
bank write mass
bank mean age
bank mean usage
bank redundancy
```

である。

collapse時のみ、

```text
minimum occupancy quota
soft balance loss
bank prior adjustment
```

を比較する。

---

# 13. Optional Adaptive Retention

本節は`screening-v5-retention` profileにのみ適用する。

v5 coreの成立にKDA型retentionは必須ではない。

## 13.1 Slot Vitality

各occupied slotに、

```text
vitality_t,m in [0, 1]
```

を持たせる。

vitalityはslot contentのnormではなく、memory confidenceとして使用する。

unit-normalized key/valueへscalar decayを掛けても、

```text
unit(alpha k) = unit(k)
```

となるため、単純なcontent scalingではread behaviorを十分に変えられない。

## 13.2 Retention Rate

```text
half_life_t,m > 0

alpha_t,m =
    2^(
        -delta_clock_t,m
        / half_life_t,m
    )
```

half-lifeは、

```text
log half_life_t,m =
      bank_half_life_prior[bank_m]
    + token_retention_bias_t
    + optional usage correction
    + optional relevance correction
    - optional conflict correction
```

とする。

初期実装では、

```text
bank prior
+ token-level scalar correction
```

までに限定する。

## 13.3 Vitality Update

```text
vitality_pre_t,m =
    alpha_t,m
    * vitality_{t-1,m}
```

token $t$のread relevance、write eligibility、victim scoringは、このpre-update
`vitality_pre_t,m`を使用する。write後の`vitality_t,m`を同じtokenのroutingへ戻してはならない。

actual writeが適用された場合、

```text
renew_t,m =
    clip(
        applied_mass_t,m,
        0,
        1
    )
```

```text
vitality_t,m =
    vitality_pre_t,m
    +
    renew_t,m
    * (
        1 - vitality_pre_t,m
    )
```

empty allocationでは、

```text
vitality_t,m = 1
```

とする。

reject writeはvitalityをrenewしない。

## 13.4 Read and Write Gating

readでは、

```text
rel_t,m *= vitality_pre_t,m^rho_read
```

write matchingでは任意に、

```text
eligibility_t,m *=
    vitality_pre_t,m^rho_write
```

を適用する。

```text
rho_write = 0
```

を安全な既定候補とする。

古いslotをmatched candidateから急速に除外すると、同一entityが過剰にnovel判定される可能性があるためである。

## 13.5 Retention Clock

比較するclockは次とする。

```text
token clock:
    delta_clock = 1

idle clock:
    delta_clock =
        1 - read_activity

conflict clock:
    delta_clock =
        conflict_signal

hybrid clock:
    delta_clock =
          eta_token
        + eta_idle * idle_signal
        + eta_conflict * conflict_signal
```

長期設定と一時情報で最適clockが異なる可能性がある。

## 13.6 Grouped Retention

slot channelごとのdecayは、一般にprojectionと可換ではない。

```text
P(Ds) != D P(s)
```

したがって、任意channel-wise retentionを既存projected-state recurrenceへ直接導入してはならない。

grouped retentionを導入する場合、slotを、

```text
z_m =
    concat(
        z_m,1,
        ...,
        z_m,G
    )
```

へ分割し、persistent projectionをblock-separableにする。

```text
P =
    block_diag(
        P_1,
        ...,
        P_G
    )
```

group decayを、

```text
D =
    block_diag(
        alpha_1 I,
        ...,
        alpha_G I
    )
```

とすると、

```text
P D z =
    D_projected P z
```

が成立する。

grouped retentionは、

```text
G = 1 / 4 / 8 / 16
```

を比較する。

block restrictionによる表現力低下を補うため、memoryへの入力前およびmemory出力後ではdense mixingを許可する。

---

# 14. Straight-Through Estimator Governance

## 14.1 Three Different Questions

次の三つを区別する。

```text
numerical parity:
    referenceとacceleratorが同じsurrogateを計算するか

estimator validity:
    surrogateがhard-forward modelに有効な学習信号を与えるか

optimization stability:
    temperature、threshold、seedに対して安定か
```

gradient parityはestimator validityを証明しない。

## 14.2 Required Metrics

```text
route flip rate
novel decision flip rate
admission flip rate
bank flip rate
victim flip rate

gradient cosine across temperatures
gradient norm across temperatures

admission saturation
novelty saturation
bank entropy
victim concentration
```

## 14.3 Required Comparisons

```text
hard-forward / soft-backward
soft-forward / soft-backward
hard-forward with stopped routing gradient
temperature sweeps
fixed route control
random route control
```

STEを使用したという理由だけでend-to-end改善をrouting学習の成功とみなさない。

---

# 15. Checkpointing and Numerical Stability

## 15.1 General Recurrence

erase/write分離後のrecurrenceは、

$$
s_t
=
A_t s_{t-1}
+
B_t \Delta_t
$$

と書ける。

```text
A_t =
    1 - erase_mass_t

B_t =
    write_mass_t
```

逆算は、

$$
s_{t-1}
=
\frac{
s_t - B_t\Delta_t
}{
A_t
}
$$

である。

## 15.2 Algebraic and Numerical Conditions

```text
A_t > 0
```

は代数的可逆性の条件にすぎない。

数値安定性はcheckpoint interval全体の誤差増幅で決まる。

interval $\mathcal I$について、

$$
K_{\mathcal I}
=
\sum_{t\in\mathcal I}
-\log\left(
\max(A_t,\epsilon)
\right)
$$

をreconstruction budgetと定義する。

`A_t`がbatch/slot/groupごとに異なる場合は各reversible state trajectoryについて
$K_{\mathcal I}$を計算し、interval平均ではなくtrajectory間の最大値を
worst-case budgetとして扱う。

単純な、

```text
max strength <= 0.95
```

だけを数値保証として使用しない。

validationでは、

```text
worst-case interval budget
observed interval budget histogram
gradient absolute error
gradient relative L2 error
age-gradient error
usage-gradient error
```

を使用する。

許容budgetはdtype、shape、backend、intervalごとに実測校正する。

universal constantとして主張しない。

## 15.3 Irreversible Novel Replacement

強いnovel eraseで、

```text
A_t < reverse_floor
```

となる場合、通常の逆算を行わない。

選択肢は次とする。

```text
checkpoint boundary insertion
local forward recomputation
sparse reset journal
```

reset journalは、

```text
token index
batch index
slot index
pre-update slot
pre-update read key
pre-update write key
pre-update value
pre-update vitality
```

を保存する。

追加memoryは、

```text
O(
    number of irreversible events
    * state per slot
)
```

である。

novel rateが高い場合、journalが全state tapeへ近づくため、

```text
journal events per token
journal bytes per token
recompute time
```

を必ず報告する。

## 15.4 Tape Contents

erase/write分離後は、少なくとも、

```text
erase mass
write mass
routing decisions
candidate state
scalar metadata
```

が必要である。

旧applied-strength一つだけではrecurrenceを復元できない。

---

# 16. Multi-Read Tiles

総key/value dimensionは固定する。

```text
d_k_tile =
    D_k_total / n_read_tiles

d_v_tile =
    D_v_total / n_read_tiles
```

projectionは一度だけ行い、reshapeしてtileへ分ける。

各tileは独立した、

```text
query normalization
key normalization
capacity-calibrated threshold
Trim-and-Square
reject-all aggregation
```

を持つ。

capacity calibrationに使うkey dimensionはtotal dimensionではなく
`d_k_tile`である。

threshold calibrationでは、

```text
occupied slots * number of tiles
```

をtotal test countとして扱う。

usage signalはtile和ではなく、

```text
max relevance across tiles
```

を使用する。

tile数による残差増幅は、

```text
1 / sqrt(n_read_tiles)
```

で抑える。

write pathは初期実装では一系統とする。

---

# 17. Semantic Versioning and Compatibility

write modeとsemantics versionは別fieldとして管理する。

```text
write_mode:
    disabled
    legacy_unconditional
    legacy_threshold
    competitive_novel

semantics_version:
    screening-v4-legacy
    screening-v4-competitive
    screening-v5-core
    screening-v5-retention
```

有効な組合せは明示的に検証する。

| semantics version | valid write mode | status |
| --- | --- | --- |
| `screening-v4-legacy` | `disabled`, `legacy_unconditional`, `legacy_threshold` | implemented compatibility contract |
| `screening-v4-competitive` | `competitive_novel` | implemented predecessor contract |
| `screening-v5-core` | `competitive_novel` | portable Phase 1 candidate; MI300X 0.185B 2,000-step / 0.3B 400-step finite gate通過、memory quality/Pallas accelerator gate未通過 |
| `screening-v5-retention` | `competitive_novel` | design-only |

同じ`write_mode=competitive_novel`でも、semantics versionが異なればoccupancy、
threshold、routing、editing、metadataの契約は異なる。

既存checkpointを読み込んだ場合、暗黙にv5へ変更しない。

```text
missing semantics version:
    write_mode == competitive_novelならscreening-v4-competitive
    それ以外はscreening-v4-legacy

v4 checkpoint:
    declared v4 semanticsを維持

v5 core checkpoint:
    occupancyおよびnew metadata必須

v5 retention checkpoint:
    vitality state必須
```

v5 benchmark toolingでは、headline commandが必ずmodeを明示する。

```text
--write-mode
--semantics-version
--backend-policy
```

未指定でheadline benchmarkを実行しない。

## 17.1 Recovery profileのMI300X反証結果

2026-07-22のMI300X再測定では、Gaussian-null threshold、soft-to-hard read、
self-index loss、上下write budget、redundancy-aware victimを含む当時のtracked profileを
評価した。0.185Bは2,000 stepをfinite完走したが、step 500以降のslot利用は
1 / 16、step 2,000のmemory residual/base RMSは9.42e-6だった。0.3Bも400 stepを
finite完走したが、2 screened layerのaggregate slot利用は3.125%、residual比は
5.37e-7だった。memory-off counterfactualのloss差はrun間で-5e-6から+1.6e-5で、
一貫した改善を示さなかった。

したがって、recovery profileは旧all-write/high-redundancy collapseを解消したが、
memory利用を成立させたのではなく、under-allocationと層間collapseへ退化解を
移したと判定する。次版はglobal平均後のbudgetではなく、screened layer・bank別の
occupancy/allocation curriculumを用い、empty capacityを各層で埋める必要がある。
このgateを通るまでretention、checkpoint redesign、v5 Pallas最適化を品質改善の
根拠として進めない。

全step時系列の再解析では、0.185B/0.3Bとも最初の非ゼロoptimizer更新を反映する
step 3でloss spikeを示し、memory collapseの主要部分は最初の数十stepで生じて
いた。さらに、補助損失がsoft write率とscreened-layer平均に作用していたため、
hard forwardではほぼwriteしないままsoft制約だけを満たす解と、片方の層が他方の
floor不足を隠す解が可能だった。

修正版では、write floorとupper budgetを層別に計算し、forward値をhard実現率、
backwardをsoft novelty/admission経路とするstraight-through surrogateへ変更する。
empty-memory bootstrap中はupper budgetを停止する。またtrunkの初期transient中は
Screening recurrence、residual、補助損失、optimizer更新（weight decayを含む）を
停止する。その後hard occupancyを維持したrecurrenceを開始し、residual、補助損失、
optimizer更新を独立LR multiplierとともに線形起動する。

MiniPile magic samplerかつ`carry_state=false`は数値健全性とthroughputのgateには
使えるが、long-memory品質の主gateにはしない。主品質gateはdistractor付きdelayed
key/value retrievalとし、学習時は文書全体を同一optimizer step内でrecurrent
chunkingする。optimizer stepを跨ぐ単純なstate carryはBPTTを切断し、後段answer
lossから前段write pathを学習できないためである。streaming評価では文書境界を
認識した行別state resetを用い、memory-off loss/accuracy deltaを必須とする。

## 17.2 Delayed Retrieval反証と次期tracked profile

2026-07-26に、0.185Bモデルをdocument-aligned delayed key/value retrievalで
MI300X単基検証した。1 documentを512 tokenとして同一optimizer step内で学習し、
held-out評価は4 x 128-token chunkをstate carryしながら処理した。

旧tracked profileはstep 102--121にall-writeへ転移し、step 122--248に
empty-memoryへcollapseした後、step 249以降に再びall-writeへ転移した。
checkpoint 200のmemory-on/off loss、accuracy、logitsは完全に一致し、
memory branchがcausally unusedであることを直接確認した。step 262では最大
gradient要素がfiniteな`1.4979e22`であったにもかかわらずFP32二乗和がoverflowし、
既存finite gateで停止した。

この結果から、tracked profileを次のように変更する。

```text
write-rate loss:
    admission floor = 0
    upper write budget = 0

per-layer rate control:
    target hard write rate = 0.05
    bounded incremental PI admission-bias controller

content signal:
    temporary pre-tile residual floor = 0.2
    four-tile effective initial floor = 0.1
    smooth-to-hard read curriculum
    self-index loss = 0

gradient isolation:
    first 2,000 Screening stepsはScreening入力をstop-gradient
    RWKV base residualのCE経路は維持

corruption guard:
    overflow-safe scaled global norm
    max absolute gradient > 1e6ならoptimizer update全体をskip
```

`admission_init=0.16`はunit-normal route scoreの5% tailに近いbiasを与える
初期化heuristicであり、controllerの正しさや目標率達成を保証しない。controllerの
rate EMA/errorはfull checkpointへ保存し、learned biasだけをportable artifactへ
exportする。

この修正は、rateの吸収状態、aux lossのGoodhart経路、trunkへのauxiliary gradient
漏出、およびfinite spikeによるoptimizer state汚染を構造的に防ぐ。しかし、
memory quality改善は未検証である。同一delayed-retrieval protocolで安定した
multi-slot利用、非ゼロread、positive memory-on/off loss delta、およびaccuracy
deltaを再現するまで、v5の有効性を主張しない。

---

# 18. Backend Validation Policy

backendについて、

```text
available
validated
headline-eligible
experimental
```

を分離する。

例:

```text
portable reference:
    available
    validated
    headline-eligible for correctness

pallas_tpu:
    available
    validated for specified shapes/devices

pallas_gpu_triton:
    availability depends on environment
    v5 requires new real-device gate

pallas_gpu_mosaic:
    experimental until device-native v5 parity
```

`auto` backendは、利用可能であるという理由だけで未検証kernelをheadline実験へ使用しない。

未検証backendは、

```text
explicit experimental flag
```

を要求する。

---

# 19. Semantic Golden Vectors

reference、GPU、TPUが一致しただけでは、正しい意味論を実装したことにならない。

各semantics versionについて、versioned golden vectorsを保存する。

```text
screening-v4-legacy
screening-v4-competitive
screening-v5-core
screening-v5-retention
```

各golden setは次を含む。

```text
input tensors
initial slot state
expected read relevance
expected matched route
expected novelty decision
expected admission decision
expected victim address
expected erase/write mass
expected final state
expected metadata
expected gradients
```

境界caseは最低限、次を含む。

```text
all empty
one occupied slot
all occupied

all rejected
exactly at threshold
slightly below threshold
slightly above threshold

weak aggregate below norm cap
aggregate exactly at norm cap
multiple aligned values above norm cap
canceling values

one eligible slot
multiple equal eligible slots
diffuse eligibility
duplicate keys

admission exactly at threshold
novelty exactly at threshold
applied mass below accounting floor
applied mass exactly at accounting floor
applied mass above accounting floor

empty-slot allocation
occupied eviction

bank tie
slot tie

checkpoint boundary write
irreversible novel replacement
```

---

# 20. Benchmark Provenance

v5実装以降のすべてのbenchmark artifactは次を保存する。

```text
schema version

semantics version

git commit SHA
dirty tree state
tree hash

config file
config hash
lockfile hash

Python version
JAX version
jaxlib version
Flax version
Optax version

CUDA / libtpu / plugin version
driver version
device name
device topology

dtype policy
compile flags
backend policy

input seed
input tensor hash

warm-up count
iteration count
raw timing samples

compile included or excluded
device transfer included or excluded
synchronization policy

output errors
per-input gradient errors
loss difference

peak memory
collective counts
```

medianだけでなく、

```text
raw samples
mean
median
standard deviation
p10
p90
```

を保存する。

kernel speedupとmodel qualityを同一表の単一指標へ統合しない。

---

# 21. Implementation Status of the Predecessor

補正後v4 recurrenceでは、2026年7月18日にTPU v5e-4上で以下が確認されている。

```text
device-native lowering
6 recurrence outputs
17 differentiable input gradients
T = 128 recurrence parity
checkpointed T = 512 parity
4-device 0.185B optimizer-step execution
```

tracked recurrence shapeは、

```text
T = 128
B = 1
M = 16
d_slot = 128
d_k = 64
d_v = 64
```

である。

報告されたmedianは、

```text
Pallas forward:
    0.578 ms

reference forward:
    3.949 ms

Pallas forward + backward:
    4.888 ms

reference forward + backward:
    13.171 ms
```

である。

これはv4 predecessor kernelに関する測定であり、次の証拠ではない。

```text
v5 correctness
v5 quality
v5 GPU performance
full-model steady-state throughput
measured peak-memory reduction
multi-host scaling efficiency
production stability
```

v5では新しいstate、routing、erase/write semanticsを追加するため、既存parity結果を流用しない。

---

# 22. Evaluation Plan

## 22.1 Baselines

最低限、以下を比較する。

1. RWKV-7 baseline
2. parameter-matched RWKV-7
3. FLOPs-matched RWKV-7
4. compute-budget-matched RWKV-7
5. legacy unconditional screening
6. legacy threshold screening
7. v4 competitive novel
8. v5 core
9. v5 core without capacity calibration
10. v5 core without occupancy
11. v5 core without ambiguity penalty
12. v5 core with tied erase/write
13. v5 core with capacity-conserving erase/write
14. v5 retention
15. softmax-over-slots memory
16. random read/write control
17. parameter-matched FFN branch
18. compute-budget-matched FFN branch
19. Mamba系matched-size model
20. RetNet系matched-size model
21. DeltaNetまたはGated DeltaNet系matched-size model

可能な場合はKDA系も比較するが、異なるstate geometryとkernel成熟度を考慮し、単純な一対一比較として扱わない。

## 22.2 Training Match

最低限、次を揃える。

```text
training tokens
optimizer
learning-rate schedule
batch tokens
context distribution
tokenizer
dataset order
parameter budget
compute budget
wall-clock budget
```

parameter-matchedとcompute-budget-matchedを同じbaselineとして扱わない。
ここでFLOPs-matchedはtokenあたりの理論演算量、compute-budget-matchedは
学習run全体の理論演算量、wall-clock-matchedは実時間budgetを指し、別々に報告する。

## 22.3 Retrieval Tasks

```text
passkey retrieval
Needle-in-a-Haystack
RULER multi-needle retrieval
RULER aggregation
RULER multi-hop tracing
MQAR
delayed key-value recall
key-value recall with distractors
same-key multiple-value recall
```

## 22.4 Load-Factor Evaluation

memory load factorを、

$$
\text{load factor}
=
\frac{
\text{independent active associations}
}{
\text{occupied slots}
}
$$

と定義する。

単一scoreだけでなく、

```text
recall vs load factor
false match vs load factor
novel rate vs load factor
eviction rate vs load factor
stale read vs load factor
```

を報告する。

slot数を増やしてscoreが上がっただけではarchitecture改善とみなさない。

## 22.5 Revision and Forgetting Tasks

```text
fact revision:
    A is X
    later A is Y
    query A

temporary override:
    normal state A
    temporary state B
    later restore A

entity relocation:
    location changes repeatedly

same-name conflict:
    distinct entities share surface name

obsolete instruction replacement

distractor contamination

short-lived state:
    should expire

persistent setting:
    should not expire
```

主要指標は、

```text
new-fact accuracy
old-fact intrusion
revision latency
stale read rate
false expiration
over-retention
retain-revise Pareto frontier
```

とする。

## 22.6 Position Sensitivity

relevant informationを、

```text
beginning
middle
end
uniform random position
```

へ置き、Lost-in-the-Middle型の位置依存性を測る。

## 22.7 Slot and System Metrics

```text
slot occupancy
dead-slot rate
duplicate-slot rate
cross-bank redundancy
cross-layer redundancy

read mass
read energy
reject-all rate

matched confidence
route concentration
route entropy

novel rate
admission rate
write rate

erase mass
write mass

eviction age
eviction usage
eviction regret proxy

tokens/s
step latency
compile time
peak memory
communication time
scaling efficiency
```

## 22.8 Causal Analysis

```text
slot ablation
slot patching
read relevance shuffle
matched-write suppression
admission suppression
novel allocation suppression
erase-gate suppression
write-gate suppression
retention freeze
vitality reset
bank-preserving shuffle
memory reset
memory branch zeroing
```

可視化だけをcausal evidenceとして扱わない。

## 22.9 通常RWKVの保持距離：入力できる長さと検索できる距離は違う

下記は自然言語モデルの長距離性能についての主張ではなく、合成の最新値検索に対する診断である。対象はScreeningなしのRWKV-7（2層、幅128、504,960パラメーター）。3つの[公開重み](experiments/retention/parents/)は検索を学んだ親モデルから、長さ96の7更新と352の1更新を混ぜて追加学習したもの。実験用の語彙は128種類の合成IDであり、一般文章用の重みではない。

16件のkey/valueを最初に書き、8件の値だけを交換した別履歴も作る。key、valueの集合・出現回数、質問は同じだが正解だけは変わる。スコアは、同一keyを両履歴とも正解した件数。数える距離は最後の初期書込みから質問までの介在token数 `D`（対象key別には `D` ～ `D+56`）。`filler`は同じ無情報token、`other-writes`は64 tokenごとに別の8 keyだけを同じ値で再書込みする。途中で質問対象のkey/valueは見せない。

| `D` | filler：両履歴正答 | 他key再書込み：両履歴正答 |
| ---: | ---: | ---: |
| 1,024 | 181/192 | 0/192 |
| 4,096 | 137/192 | 0/192 |
| 16,384 | 38/192 | 0/192 |
| 1,000,000 | 0/8 | 0/8 |
| 10,000,000 | 0/8 | 0/8（16回答すべて非有限） |

上3行はdata seed 9502の8履歴対を3つの学習初期値で共有した結果で、192の独立文書ではない。100万・1,000万は代表のseed42、data seed 9501の1履歴対。FP32 WKV・BF16活性のCPU推論で、両条件とも周期の省略はなく10Mまで入力した。fillerの10Mは有限な16予測がすべて誤り。再書込みの10Mは両層のWKVが非有限で、有効な予測がない。失敗を分母から除かない。これはこの重み・入力による反例であり、RWKV一般の限界距離ではない。

[seed別計測値・分母・実入力数](experiments/retention/distance-baseline.csv)。代表例の再現:

```bash
JAX_PLATFORMS=cpu python scripts/probe_retrieval_distance.py \
  experiments/retention/parents/long-s42.safetensors out/long-s42-other.json \
  --gap-kind other-writes --data-seed 9501 --gaps 0 1024 1000000 10000000
```

**同じ距離・同じ書込み回数でも難易度は変わる。** 親3個を凍結し、新しい64履歴対（data seed 26519、各8問）で、距離128・8 tokenごとに計16回の書込みを固定した。変えたのは、その16回で順番に再書込みする、質問に無関係なkey/valueの種類数。開始スロットは8通りを均等にし、質問・両世界の正解・書込み位置は全条件同一。途中で対象keyも正解値も再提示しない。

| 他keyの種類数 | seed 42 | seed 43 | seed 44 | 両履歴正答率 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 338/512 | 380/512 | 241/512 | 62.4% |
| 2 | 298/512 | 285/512 | 192/512 | 50.5% |
| 4 | 166/512 | 190/512 | 128/512 | 31.5% |
| 8 | 109/512 | 126/512 | 88/512 | 21.0% |

1種類から8種類への低下は各seedで29.9–49.6ポイント。非有限回答は0。1,536は独立文書数ではなく、同じ64履歴対を8問・3重みで評価した数。異なるkeyの競合と、同じkey/valueの反復による利得を個別に識別する実験ではない。それでも距離と書込み回数だけではこの親の難易度を説明できない。[履歴対ごとのカウント](experiments/retention/distractor-diversity.csv)は `python scripts/probe_retrieval_diversity.py` で再現できる。

**種類数と出現回数まで揃えた順序対照。** 別の64履歴対（seed 26520）で、同じ8個の他key/valueを各2回、同じ16箇所に書いた。`ABCDEFGHABCDEFGH` は363/1,536（23.6%）、`AABBCCDDEEFFGGHH` は405/1,536（26.4%）。2回ずつまとめた順序の利得は各seedで+2.54、+3.91、+1.76ポイント、非有限0。順序は影響するが、同じkeyを繰り返す条件の62.4%には遠い。比較対象の履歴対が異なるので、先の差の何割が順序に起因するかは推定しない。[生のペア比較](experiments/retention/distractor-order.csv): `python scripts/probe_retrieval_diversity.py --contrast-order --seed 26520`。

## 22.10 学習距離・干渉を自動で選ぶ：事前比較設計

問いは二つある。**他keyの書込みを含む学習に価値があるか**と、**自分の成績に応じて距離を選ぶこと自体に価値があるか**である。前者を親との比較だけで後者の根拠にはしない。ここでは状態やパラメーターを追加しない。§23の本試験より小さい、3初期値の合成課題パイロットとする。

- 上記3親からoptimizerを新規に開始。各腕96更新、1更新に2履歴対×2履歴×2問＝8回答、長さ256。**各腕196,608入力token、768回答**。入力は答えの直前で終わり、同じ親の各腕は同じkey、値、質問、keyごとの履歴を使う。配置と他keyの再書込み時刻のみが変わる。全入力tokenを学習計算に含め、校正と最終評価は別勘定とする。
- 学習区分は最後の基本prefixからの距離32・64・128と、その後の他key再書込み0・4・8回。25%の更新は、対象keyに新しい値を実際に書いて質問する短距離の復習に固定する。
- **固定**は全区分を均等に巡回。**自動**は訓練と最終評価のどちらとも異なる校正データで各区分を測る。他key再書込み数ごとに、全4回答正解の履歴対を1標本とし、片側95% Wilson下限が2回続けて90%以上で次の距離を開く。短距離復習の正答率が初期より1ポイント以上落ちたら昇格しない。**別seed再生**は次の親（42→43→44→42）の自動スケジュールを固定して同じ予算で学ぶ。発見された距離分布と、自身の学習状態への適応を切り分ける。
- 学習データseedは親ごとに変え、腕間では対にする。最終評価は全腕で共通の未使用履歴を用い、校正から分離する。主指標は§22.9の直接入力、**距離128・別keyを16回再書込み**したときの両履歴正答率。学習分布そのものの成績ではない。副指標は距離64・256・1,024、無情報token、短距離上書き、非有限数。
- 採用判定を結果より前に固定する。主指標で自動が固定・別seed再生の双方より平均2ポイント以上高く、各seedの差が正。短距離上書きおよび直接入力の距離0で、親に対する各seedの低下が1ポイント以内、最終評価で非有限0。達しなければ「適応が勝った」としない。校正も含む時間は報告し、等GPU時間を主張しない。1M/10Mや自然言語への外挿もしない。

最終評価器 [`score_frontier_panel.py`](scripts/score_frontier_panel.py) は保存済み重みだけを読む。直接入力のdata seedは96173、短距離上書きは96174・index 2,000,000、各32履歴対。各履歴対の入力ハッシュ、両世界の正解と予測を保存する。全腕を結合する際にデータの一致と両履歴正解数を再検算し、揃わない比較を拒否する。

**複数質問の評価コスト。** 同じ履歴のprefixを一度だけNNXで読み、得たstateを質問ごとに複製して末尾3 tokenだけ独立に読む。異なるprefixの誤共有は拒否する。学習の勾配経路は変えない。A6000・8履歴対・256 token・8質問で、BF16のXLA実行ファイルの一時バッファは103,126,320から22,457,648 bytesへ（78.2%減）。これはGPU全体のピーク使用量ではない。30回交互に同期測定した中央値は20.35→19.49 ms。BF16/FP32・2/8問の計320回答でargmaxは一致したが、BF16の8問ではlogits最大差0.0312があり、ビット一致は主張しない。[全測定値](experiments/retention/readout-a6000.csv)は `python scripts/benchmark_retrieval_readout.py --repeats 30` で再測定できる。

## 22.11 同予算の結果：干渉学習は効いたが、自動スケジュールは勝っていない

RTX A6000・JAX 0.10.0・Flax 0.12.7で9ランを実行した。全モデル504,960パラメーター、FP32重み・BF16活性・FP32 WKV。親から重みだけ引き継ぎ、Optax cosine（初期学習率0.00015、最終0.00010、warmup 20更新、weight decay 0.001）を各腕で新規開始。実績は各腕196,608 token・768回答・96個の有限な勾配更新。校正は各腕1,600履歴対（6,400回答）、最終評価は別データ。下表の分母は両世界とも正解を要求するkey対であり、独立文書数ではない。

| 未使用入力での両履歴正解数 | 親 | 固定 | 自動 | 別seed再生 |
| --- | ---: | ---: | ---: | ---: |
| 他key再書込み、距離64 | 491/768 | 577/768 | 570/768 | 572/768 |
| **他key再書込み、距離128（主指標）** | **166/768** | **281/768** | **282/768** | **283/768** |
| 他key再書込み、距離256 | 23/768 | 48/768 | 40/768 | 41/768 |
| 他key再書込み、距離1,024 | 0/768 | 3/768 | 3/768 | 3/768 |
| filler、距離0 | 752/768 | 752/768 | 754/768 | 754/768 |
| filler、距離1,024 | 730/768 | 728/768 | 732/768 | 729/768 |
| 真の上書き、距離32 | 192/192 | 190/192 | 191/192 | 191/192 |

**何が改善したか。** 固定方式でも主指標は21.61%→36.59%（+14.97ポイント、各seed +3.52、+7.03、+34.38）。自動は36.72%。改善はパラメーター追加では説明できず、追加学習後の合成課題への転移として観測された。ただし親だけは追加更新0のため、一般的な追加学習と、このデータに特有な効果を分離した数値ではない。他key再書込みの全seedで90%を満たした連続測定点は全腕で距離0まで。64ですでに下回る。微調整後の1M/10Mも自然言語の有効文脈長も測っていない。

**何が改善しなかったか。** 自動−固定は+0.13ポイント（各seed +3.91、0.00、−3.52、seed・履歴対を交差再標本化した記述的95%区間 −4.56～+4.30）。自動−別seed再生は−0.13ポイント。短距離ガードもseed 42の上書きで−1.56、seed 44の距離0で−1.95ポイントと不合格。最終評価の非有限回答は全腕0。**事前判定は不採用**。32履歴対ではWilson下限90%に届くのに32/32が必要で、現に昇格は全seedで0回。自動と別seed再生は各seedで**72/72の非アンカー更新が同じセル**（全て距離32、書込み0/4/8回が23/24/25更新）だった。この条件では個体別適応の反実仮想を作れず、僅かな得点差は適応効果ではない。同一セル順でも最終テンソルはビット一致せず、差の原因は同定していない。

**実行コスト。** 3seed合計の学習／校正秒は固定37.7／19.0、自動88.7／26.9、再生36.0／18.1。自動seed 42は最初のコンパイルを含み、24更新でプロセスを終了してからネイティブcheckpointを復元した。等GPU時間・方式固有の速度差を示す値ではない。[生の両世界の予測](experiments/retention/frontier-families.csv)、[全更新](experiments/retention/frontier-updates.csv)、[全校正](experiments/retention/frontier-calibrations.csv)、[実測予算・重みSHA-256](experiments/retention/frontier-costs.csv)。判定の再現:

```bash
python scripts/analyze_retrieval_frontier.py experiments/retention/frontier-families.csv \
  --schedule-csv experiments/retention/frontier-updates.csv
```

## 22.12 短期保持を守るための独立した追試

§22.11の短距離ガード不合格を受け、係数探索をせず一つだけ介入を固定する。自動方式ではなく**均等固定方式**の重みとその初期親の同じテンソルから、$\theta_\text{mid}=\theta_\text{parent}+0.5(\theta_\text{fixed}-\theta_\text{parent})$ をFP32で1回生成する。追加学習0、アンサンブルなし、推論は同じ504,960パラメーターの1モデル。混ぜる相手は埋込みSHAで同じ初期点に限定する。これは最初の事前比較とは別の追試で、後から係数を調整して救済しない。

評価する前に、直接入力seed 96175、上書きseed 96176、それぞれ新しい64履歴対に固定する。3親、それぞれの元の固定学習と中点を同一入力で比較。中点を保持改善候補とする条件は、①距離128・他key16書込みで親を全seedで上回り平均+5ポイント以上、②距離0と短距離上書きの両方で親に対して各seed −1ポイント以内、③全条件で非有限0。元の固定学習との得失も併記する。64履歴対に増やす目的は、2問の上書き1失敗が1.56ポイントとなる旧パネルより短距離ガードを細かく測ること（今回は0.78ポイント）。

| 新規64履歴対 × 3初期値 | 親 | 均等固定 | 親との中点 |
| --- | ---: | ---: | ---: |
| 他key再書込み・距離128 | 331/1,536 (21.55%) | 596/1,536 (38.80%) | 471/1,536 (30.66%) |
| filler・距離0 | 1,501/1,536 | 1,509/1,536 | 1,510/1,536 |
| 真の上書き・距離32 | 383/384 | 384/384 | 384/384 |
| 他key再書込み・距離1,024 | 0/1,536 | 3/1,536 | 1/1,536 |

**中点はこの追試の採用条件を満たした。** 主指標の親との差は平均+9.11ポイント（seed 42/43/44で+6.05、+3.13、+18.16）。距離0の各seed差は+0.39、+1.76、−0.39ポイント、上書きは0、+0.78、0。全条件で非有限0。対して元の固定方式は平均+17.25ポイントと検索をより改善したが、seed 44の距離0が−1.17ポイントで短距離条件に届かなかった。中点は最大の検索成績を得る手法ではない。親の短期保持を守る制約との交換で、固定に対する主指標は8.14ポイント低い。一般言語モデルの改善や長距離保証ではなく、同じ3初期値の合成課題に限った候補とする。

[生の両世界の予測](experiments/retention/midpoint-families.csv)、[評価した中点3個](experiments/retention/midpoint/)、[比較用に公開した元の9重み](experiments/retention/trained/)。以下は係数探索なしで同じパラメーターを生成し、公開された判定を再計算する例。

```bash
python scripts/retrieval_parent_midpoint.py experiments/retention/parents/long-s42.safetensors \
  experiments/retention/trained/fixed-s42.safetensors out/midpoint-s42.safetensors
python scripts/analyze_retrieval_frontier.py --midpoint experiments/retention/midpoint-families.csv
```

## 22.13 式の計算と保持を分けて測る

§22.9–22.12の実測は「以前のキーの最新値は何か」という検索であり、複数の値から新しい答えを導く能力は測っていない。既存の近い課題には、変数の別名をたどる [RULER の variable tracking](https://github.com/NVIDIA/RULER/blob/main/scripts/data/synthetic/variable_tracking.py) と、算術・代数問題を合成する [DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset) がある。以下はそれらの公式スコアではなく、式の**依存の深さ**と、途中に別の式を読ませる**保持距離**を別々に変えるための自作課題である。

例として `B=3, C=1, D=2, V=9i, A=B+C, C=C+DV` を上から実行すると、`A` は代入時の `4` のまま、最後の `C` は `1+18i`、`ACD` は `8+144i`。ここで `ACD` は三変数の積、`i²=-1` とする。`C=C+DV` を連立方程式と解釈すると別問題になるため、各入力に規則を明示する。**逐次代入**のほか、同じ計算を `C0, C1, …` のように版分けして提示順をシャッフルする**一意定義**も用意した。後者では前方参照を許すが、循環と未定義変数は認めない。一般の方程式求解や定理証明を扱う課題ではない。

各問題には、使用する変数と囮変数の初期値を入れ替えた別履歴を作る。文字の出現回数、式の配置、問いは同一のまま、どちらの問いでも正解が変わることを生成時に確認する。距離0と距離ありでは初期値・計算・正解を変えず、計算結果を書き換えない囮の式だけを末尾に追加する。深さはリテラルから `A` までの最長依存鎖の代入数で、`ACD` の最後の積は数えない。個別正答に加え、同じ問題で「値を交換した両履歴が正解」、さらに「近距離と遠距離の両方で各2履歴が正解」を数える。欠測・形式違反・未完の生成も分母に残す。

**採点系の対照実験。** データseed 42101、深さ2・8・24それぞれ16プログラム（計48）。2提示形式 × 2履歴 × 2質問 × 3距離で1,152問だが、1,152個の独立プログラムではない。生成器は式を作りながら整数対で答えを計算し、対照の解釈器は正解表を読まず、公開テキストを別実装で構文解析する。下の入力長はプロジェクト既定のRWKV語彙で測った**指示を含む全文**の長さであり、依存距離そのものではない。

| 末尾に追加した囮の式 | 実入力長 (token) | 全文を読む解釈器 | 末尾16代入だけを読む解釈器 |
| ---: | ---: | ---: | ---: |
| 0 | 118–513 | 384/384 | 256/384 |
| 8 | 191–587 | 384/384 | 128/384 |
| 48 | 576–1,002 | 384/384 | 0/384 |

末尾16代入の対照は、問い合わせに必要な値が見えなければ棄答する。無関係な式の入力が欠けただけでは誤答にしない。これは単純な読み取り範囲の対照であり、学習モデルの性能やRWKVの限界ではない。[条件ごとの分母・両履歴・近遠同時正答](experiments/symbolic-bindings/reference-panel.csv)と[実行方法](README.md#symbolic-equations-reasoning-depth-vs-retention-distance)を公開する。再生成したUTF-8/LFの `questions.jsonl` のSHA-256は `c9a1b5b839d91d3f68152e051a04ae8a41426b7c98cb56bd6a13cf036c543279`、`key.jsonl` は `f2587eb90785d2f836fd15d82a1c318c5552435fb438d692c17d5276e44e6924`。

**同じ16代入の上限で、直近ではなく必要な式を再提示するとどうなるか。** `replay` は公開入力の変数名と参照関係だけを使い、逐次代入なら参照時点の版、一意定義なら参照先を遡る。必要な原文行を元の順で返すだけで計算や値の代入は行わない。16行を超える問題は棄答として元の分母に残した。[同じ1,152問での測定](experiments/symbolic-bindings/dependency-replay.csv)は末尾16行の384問正解に対し、必要行を選ぶと768問正解。囮48行では0/384から256/384。深さ24の384問は必要行そのものが16を超え、改善しない。全文を参照できる前処理と同一の記号解釈器の結果で、16行しか保存しないモデルや学習済み言語モデルの結果ではない。

深さ24が落ちた原因を切り分けるため、両方式の上限を[32代入に倍増](experiments/symbolic-bindings/reader-budget32.csv)した。末尾32行は512/1,152（深さ24は0/384）。参照関係で選ぶ32行は1,152/1,152（深さ24は384/384）。この入力群では、難しい式にも必要な行を渡せば同じ計算器が解ける。一方、全文から必要行を抽出する索引と計算器の働きは、ニューラルな記憶や自力の算術とは区別する。

このパネルで**学習済みRWKVの数値はまだない**。§22.9の128-ID重みはこのテキストを符号化できない。モデル比較ではテキスト語彙・学習データと順序・トークン数・生成上限・学習に使わない評価seedを揃え、通常RWKVを含める。パラメーター数を揃えた対照と学習計算量を揃えた対照は別々に扱い、同じプログラムを共有する観測を独立サンプルとして数えない。今回は最大1,002入力tokenであり、100万・1,000万tokenの式追跡の証拠ではない。

## 22.14 問い合わせ先ごとに書き込み直後の内部状態を保存する

**実測前の設計。** §22.9では、同じ距離でも他キーへの書き込みによって検索が崩れた。学習済みの通常RWKVはそのままに、公開の `STORE, key, value` を読み終えた時点の**ニューラルな再帰状態**を保存する。質問にあるキーに対応した最新の状態から、元と同じRWKVに質問の3 tokenを通す。保存する内容は入力値の正解表ではなくモデル状態、答えを出すのも元の語彙ヘッド。同じキーの新しい書き込みは古い状態を置き換える。これは構造化されたキーを利用する外部メモリであり、RWKV単体の記憶容量や自然文からのキー発見の改善とは異なる。

- 参照は生の最終状態 `live`、対照 `fifo` は書き込み順に直近16状態を保存し、該当キーが複数あれば最後のものを読む。候補 `keyed` は異なるキー16個の最新状態を保存する。満杯なら書き込みの最も古いキーを追い出す。どちらもキーが無ければ最終状態で回答する。同じ重み・同じ履歴・同じ質問、追学習なし。追加パラメーター数とは別に、枠あたりの状態バイト数と枠数を報告する。
- データseed 731907、16未使用履歴対 × 8問、固定した親3重み(42/43/44)。距離0、128・1,024での他キーの同値再書き込みと**毎回違う値への変更**、1,024のfillerを比較する。別seed 731908では実際に質問対象を上書きしてから距離32で問う（16履歴対 × 2問）。キー単位の「両履歴正解」を使用し、非有限出力は除外しない。
- 主判定は、キー単位で**近距離と、変更を伴う距離1,024の両方において各2履歴正解**。`keyed` が各重みで `live` と `fifo` の双方を10ポイント以上上回り、距離0および真の上書きで `live` から各重み2ポイントを超えて落ちず、候補の非有限回答がゼロなら、この課題の改善候補とする。
- 二次評価では、各履歴の8つの質問を4つずつに分けて取得し、実際のモデル予測を `B,C,D,V/i` の整数オペランドにする。公開の逐次代入プログラム（深さ2・8・24）の `A` と `ACD` を、全方式共通の厳密な計算器で算出する。**両履歴の両回答**を要求し、未知キー・無効値・同値になる世界も都合よく除外しない。これは「学習した検索＋外部の算術」の測定であり、§22.13の生テキスト問題での自力の推論ではない。

**結果：遠距離は改善したが、無条件の状態復元は不採用。** 以下は同じキーの両履歴が正解した数。分母384は同じ16履歴対 × 8質問 × 3重みで、384個の独立サンプルではない。

| 評価 | 通常の最終状態 | 直近16書き込み順 | キー別最新16状態 |
| --- | ---: | ---: | ---: |
| 距離0 | 374/384 | 370/384 | 370/384 |
| 別キーの値を変更・距離1,024 | 3/384 | 3/384 | 370/384 |
| 質問対象を上書き・距離32 | 96/96 | 91/96 | 91/96 |
| 近距離と変更1,024を同じキーで同時正解 | 3/384 | 3/384 | 370/384 |

主指標は各重みで改善したものの、seed 43の真の上書きは通常比−3.125ポイント、seed 44は近距離−9.375、上書き−12.5ポイントで制約に違反した。非有限を除外して得た数字ではない。FIFOは16回以上の他キー書き込みで対象の状態を失い、規則通り通常状態に戻る。深さ24の4オペランド式で、距離1,024・毎回値を変更し、両履歴の`A`・`ACD`とも正しいプログラム対は通常・FIFOとも0/96、キー別83/96だった。ここで算術は共通の計算器で、差はモデルから取得したオペランドによる。

3個とも504,960パラメーターの同一親を各方式で使い、追加学習は0。1状態は67,584バイト、16枠の追加状態は1履歴あたり1,081,344バイト（1,056 KiB、索引・ワークスペース別）。履歴入力そのものは実験実装で全長を保持する。出力を得るまで古い状態に戻すため、他キーの新しい事実も同じ読み取りに必要な場合は、ひとつのスナップショットだけでは足りない。[生の予測と入力ハッシュ](experiments/binding-state/readouts.csv)、[重みと状態量](experiments/binding-state/weights.csv)、[検索](experiments/binding-state/retrieval.csv)、[式](experiments/binding-state/equations.csv)。

### 短距離を守るための独立追試：最近書いたキーは戻さない

**追試の結果を見る前に固定。** 直近16回の書き込みの中に質問キーがあれば通常状態から読む。なければそのキーの保存状態、保存していなければ通常状態から読む。履歴全体の長さや問題の種類を使わず、同じ履歴でもキーごとに選ぶ。閾値の探索はしない。単独推論で必要なのはキー別16状態と直近16個のキーの記録で、FIFO用の状態をさらに16個置く必要はない。

独立したデータseed 731927（真の上書き731928）。同じ3親・同じ16履歴対・同じ主判定と非有限・短距離条件で `guarded` を通常・FIFOと比較する。さらに干渉区間の後で質問8キーのうち4キーだけを別の値に本当に書き換え、同じ入力で古い4キーと最近の4キーを質問する対照を追加する（距離0/1,024はこの書き換えの**前**の区間）。変更の対象は履歴ごとに選び、両世界と近遠で一致させる。混合・距離1,024でも古いキーの両履歴正答が各重みで通常比+10ポイント以上、最近のキーの全予測が通常と一致することを追加条件とする。最近のキーの絶対正答率も併記する。一致自体は設計上の性質で、精度改善の証拠ではない。

混合条件の共通末尾は24 token（4組の `filler,STORE,key,value` と最後のfiller 8個）。単独の `predict_guarded` では、生の状態に加えて、16個の保存状態と196バイトの整数メタデータ（保存キー・順位、直近キーのリングとカーソル）を保持し、選んだ状態から語彙ヘッドを1回使う。比較実行では既に得た通常状態とキー別状態の出力を同じ規則で選んでいる。「最近」は経過token数ではなく書き込みイベント数であり、fillerだけではリングの内容は変わらない。その場合の長距離をこの方策だけで保証することはできない。

**独立追試の結果。** ガードつきは3重みすべてで事前条件を満たした。主指標（近距離と変更1,024それぞれで両履歴正解）はseed 42/43/44で順に128・127・116（各128対中）。通常状態とFIFOは4・0・0。距離0と真の上書きにおける通常との差はいずれの重みでも0、ガードの非有限回答は全条件で0だった。

| 新規履歴での両履歴正答 | 通常 | FIFO 16枠 | キー別16枠・常に復元 | キー別16枠・最近は通常 |
| --- | ---: | ---: | ---: | ---: |
| 距離0 | 380/384 | 374/384 | 374/384 | 380/384 |
| 真の上書き・距離32 | 96/96 | 88/96 | 88/96 | 96/96 |
| 近距離と変更1,024を同じキーで同時正解 | 4/384 | 4/384 | 374/384 | 371/384 |
| 混合1,024・古いキー | 0/192 | 0/192 | 188/192 | 188/192 |
| 混合1,024・新しく値を変えたキー | 139/192 | 136/192 | 136/192 | 139/192 |

古い値の保護と、長い履歴の末尾での新しい値の読み取りは別問題である。混合で最近書いたキーのガード正答はseed別に64/64、62/64、**13/64**。最後の重みは通常状態からでも新しい値をほとんど読めず、ガードはこれを直していない。切り替えは素のRWKVの短距離性能を守るためのもので、常に復元する方式を全指標で上回るものでもない。fillerだけの1,024は通常と同じ369/384。100万・1,000万token、16種類を超えた対象の保持、自然文中のキー発見はこの追試の対象外である。

深さ24の式では、別キーに新しい値を書き続けた距離1,024で両履歴の `A` と `ACD` をすべて正しく求めたのは通常0/96、ガード88/96。近遠とも正解の厳しい判定では0/96対85/96。混合1,024では各々0/96対62/96（近遠とも正解は0/96対51/96）。16履歴対 × 各2プログラムを同じ3重みで評価しており、96個の独立プログラムではない。計算器とプログラムは同一で、変えたのはオペランドを読む状態だけ。単独の1バンク・1ヘッド実装でも、3重みそれぞれで混合遠距離と上書きから各1履歴対を実行し、計60個の予測tokenが比較実行のガードと全件一致した。

**単純な対照。** この人工文法は書き込み命令・キー・値をすべて明示するため、RWKVなしで16アドレスの値を1バイトずつ直接保存できる。この[16バイト表](experiments/binding-state/guarded/literal.csv)は9条件の全1,056キー対で両履歴正解（決定的なため重み別に重複計上しない）。明示的な台帳を解くことが目的ならこの方が小さく正確である。本試験が示すのは、**凍結したRWKVの出力でも、どの時点の状態から読むかを変えると干渉を防げる**こと。自然文向けにはキーの発見・更新の学習、状態の圧縮、長い干渉の直後の新情報の読み取りを別々に検証する必要がある。

[生予測・公開入力ハッシュ](experiments/binding-state/guarded/readouts.csv)、[検索の分母と近遠同時正答](experiments/binding-state/guarded/retrieval.csv)、[混合条件のキー別結果](experiments/binding-state/guarded/mixed-age.csv)、[式の全結果](experiments/binding-state/guarded/equations.csv)、[重みとメモリ量](experiments/binding-state/guarded/weights.csv)。[実行方法](README.md#comparison-experiments)。

## 22.15 干渉の直後に書いた値をどう読むか

§22.14の混合条件の生予測を、**同じ4つの書き換えを問う**距離0と1,024で比べた。末尾24 token・キー・正解は一致する。各64キー対の通常状態による両履歴正答は、重み42が59→64、43が60→62、44が58→13。44の遠距離での個別誤答92件（128回答中）のうち、干渉区間で書いた別キーの値が85件、別の質問対象の値が5件、問われたキー自身の上書き前の値は2件。単に「古い版を選ぶ」ことが主因とはいいにくい。他の2重みでは同じ低下は見られず、共通の上書き規則だけの欠陥とも断定できない。

**距離を固定した対照。** 元の63 token・最後の書き換え24 token・質問を固定し、間の1,024 tokenだけを同じ長さのfillerに置き換えた。重み42/43/44の両履歴正答は、fillerで62/64・64/64・62/64、他キーの書き換えで64/64・62/64・13/64。44で別キーの値を返した85件のうち、最後の干渉書き込みの値だったのは4件だけだった。[同一キー、同一正解での比較](experiments/binding-state/replay/length-control.csv)。この重みでは単なる経過時間より途中で読んだ内容に感度があるが、これだけでは失敗箇所がニューラルな書き込みと読み出しのどちらかまでは特定できない。

**独立データの結果を見る前に固定する対策。** 長期側のキー別状態と以前のガードは消さない。履歴が64 tokenを超え、質問キーへの正規の書き込み3 token全体が末尾64 tokenにあり、その範囲の直近16書き込みに含まれる場合だけ、同じ凍結RWKVをゼロ状態から末尾64 tokenに通して読む。それ以外は以前と同じ状態から読む。64は既存プローブの処理ブロック幅で、性能から選び直さない。切替に正解・距離ラベル・直近上書きの境界は使わない。別途、実験用に境界が既知の末尾24 tokenのみをゼロから読ませ、書き込みだけで読めるかを調べる。これは配備用の方式や幅選択には使わない。

開発用には公開済みの履歴（seed 731927/731928）を使い、判定には新しい731947/731948の16履歴対と同じ3親重みを使う。合格条件は、①変更の多い1,024 tokenの後の真の書き換え4キーで、両履歴正答率が重み44で旧ガード比10ポイント以上改善し、42・43では2ポイントを超えて落ちない、②混合遠距離の古い4キーの予測は全件不変、③距離0、真の上書き距離32、同じキーの近距離と変更1,024の同時正答は各重みで2ポイントを超えて落ちない、④新方式の非有限回答は0。全方式に同じ問いを出し、計算器が同じ深さ24の式も副指標とする。追加の末尾再計算を計算量ゼロとは扱わない。

副比較として、重みごとの有効化も事前に決める。開発用データだけで「混合遠距離の新しいキーで10ポイント以上改善、古いキーの予測不変、近距離・距離32の上書き・近距離と変更1,024の同時正答で2ポイントを超える低下なし、非有限ゼロ」をすべて満たす重みだけに再読を有効化する。それ以外の重みでは従来のガードを使う。重み番号で切り替えず、決めたフラグは独立データの全質問に固定する。

**独立データでの結果。** 1,024 tokenの他キー書き込みの後、4キーを更新し、更新した4キーと以前の4キーを同じ履歴で問う。分母は重み・区分ごとに16履歴対×4キー。途中の値だけが異なる2履歴の**両方**で正解して1と数える。「一律」は条件を満たす質問で全重みに末尾再読を使う方式、「選択」は上の開発用判定で44のみに有効化した方式。

| 親重み | 更新キー：旧ガード | 更新キー：一律 | 更新キー：選択 | 古いキー：3方式共通 |
|---|---:|---:|---:|---:|
| 42 | 63/64 | 64/64 | 63/64 | 64/64 |
| 43 | 63/64 | 64/64 | 63/64 | 64/64 |
| 44 | 9/64 | 64/64 | 64/64 | 60/64 |

両方式とも事前の全条件を満たした。混合遠距離の古い4キーでは正答数だけでなく個々の予測も旧ガードと一致し、距離0・上書き32・近距離と変更1,024の同時正答も各重みで低下なし。これは全区分の単調改善を意味しない。例えば44の古いキーが**混合の近・遠ともに**正解した数は59/64から58/64になった。一律に末尾64 token**だけ**を読む別対照では、古いキーが全重みで0/64。長期状態を廃棄するのでなく、質問に応じて残す必要がある。

深さ24の式でAとACDの両方が両履歴で合った混合遠距離の件数は、重み順に旧ガード31・31・0、一律32・32・28、選択31・31・28（各32式対）。計算器は全方式で同じで、測っているのは式からの言語推論ではなく、参照値の読み違いが最終解に与える影響である。

**採否。** この明示キー課題では、追加計算を改善の大きい重みに限定する選択方式を次の基準にする。一律方式も2つの重みでさらに1対ずつ救うため精度優先の比較として残す。新規学習パラメータは0。混合する質問群は従来の状態に加えて末尾64 tokenをゼロから1回再計算して共用し、各問の答えを出すニューラル読み出しは1回。長期記憶そのものが改善した証拠ではない。最大履歴は1,111 token、3親重み・独立16履歴対であり、自然文・100万/1,000万 tokenには外挿しない。次の検証候補は、この再読結果を教師とし、同じ末尾・異なる干渉履歴では同じ新値、同じ履歴・異なる最終書き込みでは異なる新値を訓練すること。通常RWKVとパラメータ数・訓練計算量を揃え、再読なしでも効果が残るかを問う。明示アドレスに対しては§22.14の16-byte値表も引き続き対照とする。

再集計用：[開発CSV](experiments/binding-state/replay/development.csv)・[独立CSV](experiments/binding-state/replay/heldout.csv)・[選択フラグ](experiments/binding-state/replay/calibration.csv)・[採否](experiments/binding-state/replay/gates.csv)・[キー単位](experiments/binding-state/replay/retrieval.csv)・[式単位](experiments/binding-state/replay/equations.csv)・[誤答分類](experiments/binding-state/replay/mixed-recent-errors.csv)。

## 22.16 末尾の再読を教師にして、再読しない通常RWKVを鍛える

**実験前の取り決め。** §22.15で直近の上書きが読めたのは、長期側の状態に加えて、推論中に末尾を再計算した場合である。ここでは親42・43・44から各2個の通常RWKVを作り、教師もキャッシュもない**全履歴1回読み**に効果が移るかを試す。対照CEは正解トークンの交差エントロピーだけ、蒸留は同じCEに、凍結した対応親が末尾64 tokenから正解できた更新キーだけ $0.5 T^2\,\mathrm{KL}(p_\mathrm{teacher}\|p_\mathrm{student})$ を加える（$T=2$、対象回答内の平均）。教師は新しい値を問うときだけ使い、古い値には教えない。無学習の親も同じテストに通す。

両腕とも504,960パラメータ、同じ初期重み・訓練例・質問数・更新回数。64更新、Adam（学習率 $3\times10^{-5}$、勾配ノルム上限1）、1更新2 family。4更新のうち3回は、128 tokenの干渉の前半だけを変える2種類の履歴と、最後の4変数への代入を変える2種類の末尾を直交させる。各履歴で更新キーと非更新キーを1個ずつ問う。残る1回は従来の近距離と上書き32を交互に使う。両腕とも同じ教師出力を計測するがCEの蒸留係数は0。教師の計算は実質無料とは扱わない。訓練seedは64100+親seed。途中で独立評価を見て設定や重みを選び直さない。

**独立評価と採否。** 未使用の入力seed 731967/731968、32履歴対を固定。混合1,024で実際に更新された4キーの両履歴正答率を主指標とする。蒸留がCEより3親合計で5ポイント以上高く、3親中2親以上で改善し、各親の非更新4キー・近距離0・上書き32に2ポイントを超える低下がなく、非有限回答ゼロなら蒸留を採用する。併せて、CE単独を無学習親と比べ、合計10ポイント以上改善し、同じ保護条件を満たすかも検証する。距離を固定したfiller・変更書き込み、同じキーの近遠同時正答、深さ24のAとACDが両履歴でともに合うかは副指標。学習するのは記号値の取得であり、式の計算器は共通。これは3親の小規模比較であり、§23の5つの独立初期化による確認とは区別する。

**長さの反例探し（採否には使わない）。** §22.15で干渉に弱かった親44と、そのCE・蒸留の3重みに、別seed 731987の同一1履歴対を与える。8つの非対象キーの値を変え続ける区間を0、16,384、1,000,000 tokenとし、最後の4つの書き換えと8問は同じにする。各長さで新4キーと古4キーを別集計し、全語彙の非有限予測も数える。各区間を物理的に通す（周期スキップなし）。うまくいっても1履歴対で一般的な百万token保持を主張せず、失敗すれば当該重みでの反例とする。

効果量には32 familyを単位とした対応付きbootstrap（20,000回、seed 424216、百分位95%区間）を添える。同じfamilyの全キー・両履歴・3つの親は一緒に再標本化し、この区間を初期重みの母集団への区間とは解釈しない。親ごとの差には全8通りの符号反転による両側確率を別記する。これらは採否条件を変更しない。

**A6000実測。** 各学生は64更新、105,856個の入力位置（共有した履歴は1回分）、1,024回答で学習した。親ごとに同じバッチのハッシュと会計がCE・蒸留で全更新一致した。教師の12,864入力位置は両腕の計測で共有し、更新キーへの384回答は各親とも384件すべて正解だった。学生の推論は教師を呼ばない。実行時間は逐次実行で最初の腕にJITコンパイルが寄るため、学習速度の優劣には使わない。

| 推論時はいずれも通常RWKV | 混合1,024 更新キー42 /128 | 同43 /128 | 同44 /128 | 混合1,024 非更新キー /384 | 近距離0 /768 | 上書き32 /192 | 混合1,024 深さ24のA・ACD /192 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 追加学習前の親 | 127 | 122 | 9 | 0 | 756 | 192 | 4 |
| 正解だけの追加学習（CE） | 126 | 124 | 59 | 4 | 758 | 191 | 4 |
| 同じ追加学習＋末尾教師（蒸留） | 127 | 124 | 66 | 3 | 758 | 191 | 4 |

全件、値が異なる両履歴に正解して1件。**CEを次の追加学習の基準に採用し、蒸留は採用しない。** 更新キーはCEが親比+51/384（+13.28ポイント、固定3親でのfamily区間+10.68〜+15.89）。51件の純増のうち50件は44で、親間の効果差は大きい（標準偏差22.36ポイント）。蒸留はCE比+8/384（+2.08ポイント、同+0.52〜+3.91、親間SD 2.96ポイント）で、事前の+5ポイントに届かない。保護条件は両比較とも満たし、独立パネルの非有限回答は全方式0。親単位の符号反転は両比較とも $p=0.50$ で、広い初期重み集団での優位性ではない。CE対親では追加学習の有無まで変わるので、対の組み方だけの効果とまでは分離していない。

**古い値は依然として別問題。** CEの混合遠距離でも非更新キーは4/384。たとえば純粋な1,024 tokenのfillerならCEは737/768キー対正解だが、同じ長さで別キーへの書き込みを続けると8/768。式のA・ACDは全方式4/192で、新しい値を部分的に救っても式全体の正答は増えなかった。

**百万tokenの反例。** 親44・CE44・蒸留44はいずれも、干渉0では更新4/4・非更新4/4、16,384と1,000,000では更新0/4・非更新0/4。各条件の全16予測は有限だった。長い間に古い値を失うだけでなく、**その後に代入した値**にも正答できない履歴対がある。同じ全入力をCPUで走査した16-byteの明示キー値表（採否とは別の陽性対照）は、両区分ともすべての長さで4/4だった。学習済みモデルの10Mは今回測っていない。

**次に切り分けること。** 干渉長・書き込み密度を独立に振って、直近の再代入さえ読めなくなる距離を測る。そのうえで、直近を読む状態と、変更されたアドレスだけを書き替える小容量の長期状態を分ける。明示キーの16-byte表・以前のキー別ニューラル状態・同じ状態バイト数と計算量の通常RWKVを対照に置き、更新・非更新の両方と式の最終解が改善するまで長期保持とは呼ばない。

公開物：[学習済み6重みと更新ログ](experiments/binding-state/distill/)・[全予測](experiments/binding-state/distill/panel/readouts.csv)・[キー別](experiments/binding-state/distill/panel/retrieval.csv)・[式](experiments/binding-state/distill/panel/equations.csv)・[採否](experiments/binding-state/distill/panel/gates.csv)・[区間](experiments/binding-state/distill/panel/intervals.csv)・[百万tokenのキーごとの結果](experiments/binding-state/distill/stress/)。

## 22.17 書き込みの干渉を遮る：双対アドレス型の状態screening

**測定前の設計。** v5では16スロット中1つに集中し、メモリを切っても損失差がほとんどなかった実測がある。そこで採用率の再調整ではなく、「何を残すか」と「書き込みでどの読み出しを変えてよいか」を分ける。v5の設定・既存重みは変更しない。まず、書き込み・問い合わせ先が公開トークンで明示される記号課題で、後者を単独に検証する。アドレス検出や自然文理解まで解いたとは扱わない。

16個のキーの凍結RWKV埋め込みを正規化し、薄いSVDで内積を変えず16次元の読み方向 $q_i$ に写す。対照は書き方向 $u_i=q_i$、提案は $QU^\top=I$ を満たす双対方向。どちらも同じFP32状態 $M\in\mathbb{R}^{16\times128}$、同じ演算 $M\leftarrow M+u_i^\top(v-q_iM)$ を使う（$q_i,u_i,v$ は行ベクトル）。実数演算では、双対方向を使うと他キー $j$ の読み出し $q_jM$ を変えずに $q_iM$ を $v$ にできる。読み書きの度に減衰させない。実際のFP32誤差は実測し、特異・悪条件のアドレスは拒否する。

リテラルの内容 $v$ は生の値IDではない。§22.16の凍結CEモデルで、ゼロ状態から `STORE key value`、filler 8個、`QUERY key ANSWER` を読んだ最後の隠れ表現である。256通りを前計算できる。問い合わせは回収した隠れ表現を同じ最終正規化・全128語ヘッドに通す。新命令 `COPY dst src` は読み出した連続表現を**その時点で**書き込み、後日srcを書き換えてもdstは追従しない。値未定義のsrcはdstも未定義とする。動的状態は行列8,192 byte・定義済み16フラグ・字句解析の末尾2 token。静的な256表現のキャッシュは別途131,072 byteであり、無料の追加パラメータとはみなさない。これは通常の本文を読むトランクに代わるものではなく、明示的な束縛を扱う実験用読み書き経路である。

何が証明できないかも明確である。16個の既知キーでは $R=QM$ と変数変換すると、実数精度の双対更新は $R$ の対応行だけを書き替える**連続値レジスタ表**と等価である。値表を超える汎用推論の証拠ではない。独立な読み方向が状態のアドレス次元を超えたり、2つの名前が同じ方向に写ったりすると、全キーを同時に保護する双対は存在しない。距離・同時に保持する種類数・アドレスを本文から発見する能力は別々の軸である。

**事前に固定する比較。** 追加学習なし、CE42/43/44、通常パネルは新seed 732007/732008の32履歴対。通常RWKVの全履歴読み、§22.15のキー別状態＋再読、同一コストの通常方向、双対方向、文法を知る値表を比較する。新seed 732013の別32履歴対では4本のコピー鎖について、各コピー直後に元変数を別値で上書きする。深さ2/24、途中の干渉0/1,024を交差させ、最後に生きている4値、別値にされた4値、同じ正確な計算器による深さ24の `A` と `ACD` を両履歴で採点する。標準RWKVに未学習の新命令を課して対照にするのではなく、同じ命令を扱える2種類の行列と値表を主比較とする。

**コピー先の偶奇ショートカットを除く。** 深さ2と24はどちらも偶数なので、固定した2変数間を往復させるだけでは、元の変数名を答えるだけでコピー先が正解になる。コピー課題の予測を取る前に、各段で生きている4本の値を、空いた側の4変数へseedで並べ替えて配線することを明記した。経路とクエリ順は2世界・近距離と遠距離で共通、干渉列は鎖の中間に置く。元のprefixだけを値表で読む場合、`COPY`だけ除いて全`STORE`を読む場合の実測も残す。いずれも採否条件は変えない。

候補をこの記号課題で次に進める条件は、各重みで混合1,024の更新・非更新キーおよび深さ24・干渉1,024のコピー先がそれぞれ90%以上、通常方向比で混合1,024の非更新キーと新コピー課題の最終式が各々3重み合計で10ポイント以上改善、近距離0と上書き32で通常RWKV比2ポイントを超える低下なし、非有限予測ゼロ。通常方向でも同等なら、双対化の必要性を主張しない。長さは別指標とし、§22.16と同じ重み44・seed 731987の1履歴対に対し、0・16,384・1,000,000・10,000,000の変更tokenを実際に逐次処理し、更新・非更新各4キーを別々に報告する。一律な有効文脈長には換算しない。

**比較予算。** RWKVを使う経路は各seedで同一の504,960パラメーターを凍結し、追加学習は0。通常方向と双対方向には同数のアドレス定数2,048 byte、同じ動的状態8,216 byte/履歴、同じ字句規則と計算式を与えた。256個の局所表現には1重みあたり256×14 tokenの前計算と128 KiBの共有キャッシュが要る。通常RWKVの再帰状態は67,584 byte、キー別状態＋再読は16枠にさらに1,081,344 byte（索引・再読用の履歴は別）で、これらとはメモリも帰納バイアスも同一ではない。値IDだけを覚える文法既知の表は16 byteで、RWKVは不要。同じパラメーター数だけを根拠に比較を公平とはしない。この節の推論はWindows AMD64 CPU、Python 3.13.13、JAX 0.10.0、Flax 0.12.7、NumPy 2.3.5。

**距離だけが原因か。** 同じ長さ1,024でも、通常RWKVは空白のfillerだけなら724/768、別キーの値を繰り返し変更すると17/768に落ちた（2世界とも正しい場合だけ1件）。同じ重み・対象の問い・距離で、介在する情報が違う。更新済み4キーと古い4キーを混ぜた場合の通常RWKVも、直近308/384に対し古いものは1/384。少なくともこの課題では、一律に減衰時間を延ばすことと、書き込み先を分離することは同じ対策ではない。

**主結果。** 32履歴対×3重み。数字は正解件数で、同じ問いに対して値を変えた2世界を両方当てる必要がある。式は深さ24で `A` と `ACD` が2世界とも正しい場合のみ1件。通常課題は1家族2式、コピー課題は生き残った4値を使う1式。「—」は測定なしで、誤答扱いではない。

|読み出し経路|近距離 /768|上書き32 /192|混合1,024：更新 /384|同：古い /384|同：式 /192|コピー24段・1,024：生存値 /384|同：式 /96|
|---|---:|---:|---:|---:|---:|---:|---:|
|通常RWKV|757|191|308|1|2|—|—|
|キー別状態＋再読|757|191|384|366|177|—|—|
|行列・読み方向に書く|724|192|382|225|68|0|0|
|行列・双対方向に書く|768|192|384|384|192|384|96|
|直接レジスタ（追加実験）|768|192|384|384|192|384|96|
|公開文法の値表|768|192|384|384|192|384|96|

**事前の採否条件は通過。** 各重みで更新・古い・深いコピー先がそれぞれ128/128、近距離・上書きの保護条件を満たし、非有限予測は0。通常方向比で古いキーは+41.4ポイント（履歴家族を単位とする対応bootstrap 95%区間 +37.5〜+45.3）、コピー式は+100ポイント。後者は32家族すべてで差が同じため区間も[100,100]であり、未知の家族やモデルへの保証ではない。両世界**かつ**同一問題の近・遠で両方正解、という厳しい採点でも、双対は混合の更新384/384・古い384/384、深いコピー384/384。通常方向は順に381/384・220/384・0/384。

3個の局所符号器はいずれも256/256の値を正しく復号した。書き込み方向が他の読み出しにかかる係数 $\max_{i\ne j}|q_j u_i^\top|$ は通常方向で0.380〜0.473、双対で最大 $3.62\times10^{-8}$。コピー2段・干渉1,024でも、通常方向の保持値は46/384、24段では0/384なのに、最後の上書き値は372/384。単に全体が壊れたのではなく、更新前の値を次の変数へ受け継ぐ経路が先に壊れた。深い課題でprefixだけを読む値表と`COPY`だけ消した値表は、いずれも生存値0/128（モデル非依存につき3重計上しない）。一段の4コピーを先にまとめてから4上書きした予備走行は主集計に混ぜず、[別ファイル](experiments/binding-state/address-screening/pilot-grouped-version.csv)に残した。その深い式も通常方向0/96・双対96/96だった。

[全4,992件の家族・経路別レコード、入力ハッシュ、干渉係数、物理ストレス測定](experiments/binding-state/address-screening/)／[採否](experiments/binding-state/address-screening/scores/gates.csv)／[式の内訳](experiments/binding-state/address-screening/scores/equations.csv)。

**本当に途中のコピーを追う必要があるか（追加の反証実験）。** 主パネルの上書き値には、同じ元値から来たことを示す目印が残り得る。そこで新seed 732019の32家族で全上書きを家族内の同一リテラルにそろえ、2世界の最初の値も同一にした。異なる入力は**第12段のコピー先を入れ替えた2 tokenだけ**。その後の1,024 tokenと12段のコピー、最後の問いは完全に同じ。正解は交換した2本の経路をたどって求め、公開トークンだけを実行する値表でも一致を確かめた。96/96の式対で正解が2世界間で実際に変わった。

双対方向と直接レジスタは、交換の影響を受ける値192/192、`A` と `ACD` の同時正解96/96。通常方向は0/192、0/96。第1世界の**全正解を知っていても**中央の交換を無視して同じ答えを返す対照は0/192、0/96。これで、初期値・末尾・上書き値の目印だけでは上の成功を説明できない。ただし演算は共通の正確な計算器で、自然文から代入を学んだ証拠ではない。[この追加実験の個票と採点](experiments/binding-state/address-screening/middle-swap/)は事前の採否と別集計。

**1,000万tokenで状態は壊れるか。** 重み44、1履歴対、512 token周期で8個の別キーの値を入れ替える。1,000万tokenは各世界で実際に字句解析し、1,250,000回の書き込みを実行した数で、周期を一度で済ませた計算ではない。全トークンを通常RWKVに通す時間ではない。予測を確定した後に、回収した隠れベクトルと正しいリテラルを局所符号化したベクトルの相対L2誤差を測った。表の誤差は8キー・2世界の最大値。

|干渉token数|通常方向：更新 / 古い（各4）|双対方向：更新 / 古い（各4）|双対方向：最大相対誤差|
|---:|---:|---:|---:|
|0|4 / 4|4 / 4|$1.73\times10^{-7}$|
|16,384|4 / 3|4 / 4|$6.11\times10^{-6}$|
|1,000,000|4 / 3|4 / 4|$3.72\times10^{-4}$|
|10,000,000|4 / 2|4 / 4|$3.70\times10^{-3}$|

この例では正解トークンは保ったが、FP32状態は不変ではない。通常方向の1,000万時点の最大相対誤差は1.30だった。§22.16の同じ通常RWKV（CE44）は100万で更新0/4・古い0/4。通常RWKVの1,000万走行は測っていない。有限の既知キーと周期的干渉での一反例探索を、未知の100万個の事実の記憶能力やテキストの文脈長と同一視しない。

**誤差を見てからの追加実験（採否に使わない）。** 離散アドレスが確定している場合には、$M$ ではなく $R=QM$ 自身を同じ16×128のFP32状態として持てる。書き込みは宛先1行への代入、コピーは元の1行の値の転送になる。隠れ表現・RWKVヘッド・字句解析は一切変えない。この `register` 実装は入力と重みのハッシュが一致する上の4距離すべてで更新4/4・古い4/4、最大相対誤差 **0** だった。1回の更新で扱う値要素は $K V$ ではなく $V$。ラベル表への置換ではないが、スロットが既知の課題では密な双対行列より直接レジスタを選ぶ理由になる。学習した曖昧なアドレスをこの方法で解決できるわけではない。

**次に外す仮定。** まず格納・保持の側を固定し、RWKVの各時点の隠れ状態から「無視・代入・コピー」とその参照先を選ぶ部分だけを学習する。評価時には命令境界・正しいスロットを与えない。名前の単純な出現と再代入を区別する問題、訓練に現れない複数トークンの名前、同時変数8/16/32と距離を独立に振る問題を分ける。失敗を誤書き込み・書き逃し・同じ場所への衝突・正しく読んだ後の計算ミスに分解する。同じトークン数・更新回数の通常RWKV、追加パラメーターをトランクに割いた対照、文法既知のレジスタ上限を保つ。ここからは**未検証の設計**であり、本節の正答率には含めない。

## 22.18 記憶の再利用と、別の問い方への自力転移を分ける

**予測前の設計。** §22.17の満点は16種類の既知の名前に対する格納・読み出しであり、式は外部計算器で解いた。今回はチェックポイント・字句解析器・RWKVヘッドを変更せず、新しい課題でこの区別を直接測る。CE42/43/44、データseed 732041/732043/732047/732049/732053に各8家族。1家族はデモ用9駅と評価用7駅の独立な環状路線。2世界で駅名・事実の表示順・値の出現回数・デモを同じにし、評価側だけを違う路線にする。評価7駅をそれぞれ問い、同じ駅について両世界が正解して初めて1件。距離条件は別路線への干渉0/1,024 token。

課題は「次の駅」「4駅先」「ひとつ前」「偶数IDからは1駅、奇数IDからは2駅」。駅も終端の問いも共通で、どの規則かを伝えるのは**評価路線と交わらない3つの解答付き例**だけ。課題名を推論器に渡さない。書き込み元の名前には既知形式と、家族ごとに対応が変わる別名形式を用意し、どちらにも同じ公開対応表を先頭に置く。

**採点を混ぜない。** (1) 通常RWKVまたは§22.17の読み出しを例付きの入力にそのまま適用。(2) 別名の書き込みだけ公開表で標準化し、問いはそのまま。(3) 標準化後、各方式から16個の1駅読みを取り、外部手順がデモ3例すべてに合う移動量を−4〜+4から一意に選んで実行。判定不能なら棄権する。外部手順は評価路線の答えも規則名も見ないが、「一定歩数の移動」という仮定をあらかじめ持つ。偶奇規則はこの仮定の外に置く。採否は(1)の既知名で、各重み・両距離において「4駅先」と「ひとつ前」がともに90%以上なら、この2課題への自力転移を支持する。(2)(3)の成功でこの条件を代用しない。

**結果：新しい地図を覚えられたが、新しい問いには自力で切り替わらなかった。** 各セルは干渉 `0 → 1,024` tokenでの正解件数、各時点の分母は840（40家族×7駅×3重み）。同じ40家族を各重みで使っており、840問を独立な標本とは見なさない。以下は既知名。`raw` は3例を含む未加工の入力に答えたもの、`walk` は外から移動手順を足したもの。

|読み出し|手順|次の駅|4駅先|ひとつ前|偶奇で1駅／2駅|
|---|---|---:|---:|---:|---:|
|通常RWKV|raw|152 → 9|0 → 0|1 → 1|82 → 5|
|同規模行列・読み方向へ書く|raw|666 → 602|0 → 0|0 → 0|360 → 326|
|双対方向 / 直接レジスタ（各々）|raw|840 → 840|0 → 0|0 → 0|429 → 429|
|通常RWKV|walk|340 → 4|24 → 0|111 → 0|0 → 0|
|同規模行列・読み方向へ書く|walk|666 → 526|285 → 134|411 → 261|0 → 2|
|双対方向 / 直接レジスタ（各々）|walk|840 → 840|840 → 840|840 → 840|0 → 0|

自力転移の事前条件を満たした読み出しは**0/4方式**。双対方式は干渉1,024のあとでも、デモを読む前の16駅すべての1駅読みが1,920/1,920正解だった。にもかかわらず4駅先と逆向きは0。4種類のデモに差し替えたときに返したトークンを突き合わせると、双対は1,680/1,680で全く同じだった。現在の書き込み経路では解答例は`STORE`でも`COPY`でもなく、同じ最終キーは同じ値を読む。偶奇の429正解も「次の駅でよい偶数駅」に限られ、規則を学んだ結果ではない。通常RWKVは全デモで不変が1,086/1,680と反応自体はしたが、未学習の2規則を安定して実行できなかった。

**名前の認識と手順の再利用も別問題。** 別名形式では、双対方式は対応表が見えていても「次の駅」で0/840（2世界の個別回答1,680件すべて未定義）。手書きの辞書解釈だけを加えると次の駅は840/840に戻り、4駅先はまだ0/840。さらに外部手順が、正解ではなく**凍結した読み出し自身の辺予測**を使って例から歩数を選ぶと、4駅先・逆向きも各840/840になった。これは一定歩数という操作の範囲内の再利用であり、モデルが問題の解き方を発見した証拠ではない。歩数が一定でない偶奇課題にはこの手順は240/240世界で棄権した。壊れた辺表で偶然当たることはあり、通常方向行列の2/840を条件分岐の学習とは解釈しない。

CPU推論、追加学習・調整なし。`walk` は各履歴の16辺を読み、逆引きと最大4歩の探索を外部で行うため、1回の通常推論と同計算量ではない。通用したのは既知スロットへの状態保持であり、自然文・未知の操作への一般化は未実証。次の設計で試すべきなのは、読み出した値をすぐ最終回答にせず、問いとデモを見て**次の読み先と停止を決める**学習経路に戻すこと。その経路も、訓練から丸ごと除いた操作・名前で、同じ計算予算の通常RWKVと比較する必要がある。[各家族の回答・ハッシュと各モデルの辺予測](experiments/binding-state/route-transfer/)／[採否](experiments/binding-state/route-transfer/gates.csv)。

## 22.19 読んだ値を次のアドレスへ戻し、停止も学習する

**予測前に固定した設計。** §22.18で凍結RWKVは地図を保持できても解答例に応じて操作を変えなかった。今回は規則リストや歩数探索を推論器に渡さず、16駅についての凍結RWKVの実測読み出しをレコードとする小さな共有コントローラを学習する。2個の注意ヘッドは、現在の読み位置とレコードの両フィールドの照合、取り出すフィールドの重み、続行・どちらのヘッドで回答するかを学習する。問い合わせとデモ始点の公開IDの4ビットに対する関連度も学習する。1ラウンドの出力を次の読み位置へ戻し、同じ重みを最長6ラウンド使う。推論に時刻別の重み、正解歩数、方向、条件分岐のラベル、評価駅の正答はない。上限まで停止しなければ未回答とする。

**訓練と対照。** 訓練で使う真の移動量は `+1,+2,+3,−2` のみ。偶数駅・奇数駅それぞれに割り当てる16組のうち `(+1,+2)` を丸ごと除いた15組を使う。公開デモは常に3組、訓練信号は評価駅役の最終回答のみ（途中のアドレス・停止ラベルなし）。同じ3つのCE重みを凍結し、同一初期化・訓練バッチ・640更新で、双対記憶＋フィードバック、同じ双対記憶で毎ラウンド初期位置に戻す対照、通常RWKVの辺読み出し＋同じ学習ヘッドを比べる。各方式とも16個の基本読みと最大6ラウンド、追加の学習パラメーター数をそろえる。記憶形成側の計算や実時間まで等しいとはしない。

**評価。** 訓練・開発・最終のグラフseedは分離する。最終は5 seed（743101/103/107/109/113）、各8家族×2世界、干渉0/1,024。本文に公開した別名宣言による対応付けは共通の入力処理であり、自然文の名前解決を学んだとは扱わない。評価時は訓練と重ならない別名IDを使う。既知の「次の駅」を守り、**訓練にない**「4駅先」「ひとつ前」「偶数1・奇数2」が3重み・両距離の各々で両世界正解90%以上かを判定する。7駅の周期上で+4と同じになる−3も訓練しない。さらに成分自体も新しい「偶数4・奇数−1」は別集計。初期重み、デモの正答を隠す介入、1ラウンド制限も報告し、組み込んだ帰納バイアスと学習の寄与を区別する。

**実測：移動手順を外から与えなくても、訓練にない移動量（4駅先・1駅前）と、偶数駅・奇数駅に割り当てる移動量の新しい組合せに転移した。** 2駅前（−2）は訓練済みであり、1駅前（−1）の成功は逆向き自体の初見転移ではない。上記条件を `bccd089e85` で登録し、評価前に3方式×3本の学習を終了した。凍結した合成語彙RWKVは504,960パラメーター、追加学習は各方式とも67個のFP32スカラーのみ。訓練64地図・開発16地図、バッチ32、Adam（学習率0.03、勾配ノルム上限1）、640更新。損失は最終回答の負の対数尤度（到達不能な対照にも定義するため一様分布を $10^{-6}$ 混合）＋期待ラウンド数×0.002。中間経路は教師にしていない。以下の各セルは干渉 `0 → 1,024` tokenの**両世界正解件数**。分母は各距離840（5データseed×8地図×7問×3重み）。同じ地図・同じ問いを使うので840独立試行ではない。

|入力記憶と読み方|次の駅（既知）|4駅先|ひとつ前|偶数1・奇数2|偶数4・奇数−1|
|---|---:|---:|---:|---:|---:|
|双対記憶＋学習した反復|840 → 840|840 → 840|840 → 840|840 → 840|840 → 840|
|同じ双対記憶・毎回初期位置に戻す（同数パラメーター、別途学習）|840 → 840|0 → 0|840 → 840|414 → 414|426 → 426|
|通常RWKVの辺＋同じ反復ヘッド（同数パラメーター、別途学習）|23 → 2|0 → 0|6 → 0|10 → 3|8 → 2|
|通常RWKVの辺を直接回答（学習ヘッド・デモなし）|377 → 5|0 → 0|0 → 3|173 → 4|0 → 2|

双対＋反復は**各重み・各距離・各課題で280/280**。リセット対照でも逆向き1歩は満点だった。逆引き自体を反復の成果としない。差が出たのは、一度読んだ駅からさらに進む問いである。「偶数1・奇数2」の改善は **+50.7ポイント**（地図seed→家族の順に10,000回対応付き再標本化した95%区間 **+42.5～+58.6**、3重みは固定）。4駅先の差は全地図で+100ポイントなので標本内のbootstrap区間も一点になる。未知の全地図・全重みでの完全性の保証ではない。

**何を使って正答したか。** 4駅先では各問がちょうど4回、ひとつ前では1回で自発停止した。偶数1・奇数2で平均1.507回、偶数4・奇数−1で2.479回（個別の問いの論理ラウンド。バッチ実行は最後の問いまで）。同じ学習済み重みを1回に制限すると4駅先は0/840、偶数1・奇数2は414/840。デモの答えを隠すと全課題で全問が上限6回で未回答、各学習開始時の重みのままでも同じだった。公開例 `episode.json.example` には地図・3デモ・3問しかない。保存したCE42用ヘッドを別プロセスで読み込むと、答えトークン `[44,42,47]` を `[1,4,1]` 回で返し、手計算の3答と一致した。

**計算量と範囲。** 通常RWKVの16辺の正答は近距離3,181/3,840、遠距離1,504/3,840（双対はともに3,840/3,840）。通常側の学習ヘッドは既知の「次の駅」で直接読みより悪く、その弱さを反復の改善幅とは見なさない。双対とリセットの対は同じ辺を使い、記憶形成・初期化・訓練バッチ・追加パラメーターまで同一。対して通常RWKVとの比較では、凍結重みと読出し回数の上限は一致するが、記憶容量・履歴処理のFLOPsは一致しない。双対側は既知の16アドレス、学習済みRWKVの256組の局所表現、連続値の外部状態、公開辞書を使う。各ラウンドは2ヘッドで全16レコードを照合し、最大6回で明示的に止まる。CPUで9ヘッドの更新区間合計43.9秒（初回コンパイル込み、開発評価・辺の準備は別）。履歴から実際に辺を取る費用も`cache/*-cost.csv`に分けて記録した。学習・評価にGPUは使用していない。

**追加検査1：依存距離と記憶容量を区別する。** 別の3地図・各2世界に、デモ路線の駅への乱数生成の上書きを8 tokenに1回流した。途中結果は実際に100万／1,000万tokenを逐次読んだ状態から分岐し、デモ路線を修復してから、書き直していない評価路線を問う。1,000万時点で各世界125万回の書き込み、固定ブロックの繰返しや期間のスキップなし。3重み×3地図で、0・100万・1,000万の各時点とも辺288/288、**5規則それぞれ両世界63/63**、未回答なし。これは**16駅を保持したままの距離**の実測であり、1,000万件の独立事実・自然文1,000万token・通常RWKVに対する同時間比較ではない。

**追加検査2：歩き方の転移と、条件そのものの発見は違う。** 距離1,024の同じ40地図で、偶数／奇数（駅IDの第0ビット）で1歩／2歩を選ぶ問題と、**第1ビット**で同じ2択を選ぶ問題を比べた。両条件が公開デモ3件に異なる区切りを与え、他ビットとの向きの入替えでも説明できない3駅を、答えを使わず選択できた地図は40/40。地図・デモの出発地・評価駅は固定し、各問題を指定するデモの答えだけを入れ替えた。第0ビットは840/840、第1ビットは**369/840（43.9%）、7駅すべてに両世界正解した地図は0/40（各重み）**。2条件の正解が本当に違う駅だけでも417/417対192/417。学習したID関連度では第0ビットの重みが2.12～2.15、残る3ビットは約−0.06だった。**同じ条件軸で新しい操作を組み合わせられたが、新しい条件軸を例から選び直す能力は示せなかった。** 次に外すべき仮定はここである。駅数や読出し上限を増やすだけでは、この失敗を解決したことにならない。

[学習済み9ヘッドと学習曲線](experiments/binding-state/iterative-reader/heads/)／[実測辺・教師用の真値・コスト](experiments/binding-state/iterative-reader/cache/)／[生の予測・判定・対応付き区間](experiments/binding-state/iterative-reader/panel/)／[長距離と条件切替の生データ](experiments/binding-state/iterative-reader/diagnostics/)。

## 22.20 問いを知らずに書き、後から場面を一番地だけで引けるか

**予測を見る前の実験計画。** §22.19は公開された駅名で書けた。ここでは書き込み命令・名前の対応表・正しい番地を取り去る。8人×8状況×8種類の物を表す制御語彙で、同じ人・別の状況と、別の人・同じ状況を両方含む履歴を作る。書き込み時には後の問いを渡さない。2世界の違いは、二つの場面で物を交換することだけ。語の出現数とそれ以降の文章は同じで、両方の場面を別々に問う。8トークンごとの区切りと語彙は既知の実験上の足場であり、自然文の任意の位置から出来事を発見する試験ではない。

**なぜこの比較か。** 同じ凍結RWKVから文ごとの特徴を取り、1個の書き込みゲート、書く番地・読む番地の各6ビット、格納する8クラスの連続値を学習する。推論では64スロットのうち1か所だけに書き、問いから作った1番地だけを直接読む。過去の全文や64スロットとの類似度計算はしない。訓練時のみ、離散番地の確率を全64通りに展開する。最終回答だけで学習する方式と、**学習時に限り根拠文の位置も教える方式**を、同じパラメーター・データ・初期化で比べる。後者にも番地の正解は与えない。対照は通常RWKVの連続した全履歴状態に、ほぼ同規模の別途学習した回答ヘッドを付ける。実際に根拠文を選んで読ませた場合の復号率、および問いをすり替えた対照も分ける。

**何をもって成功とするか。** 訓練は事実4/8件、末尾の背景文0/8件。最終は5データseed×8家族、3つの凍結重み。事実4/16件と、背景文0/128件（干渉0/1,024 token）を独立に振る。初見の履歴、訓練から除外した人物と状況の組、さらに同じ組・同じ履歴で問いの語順だけを変えた条件を区別する。主要判定は**未訓練の組・通常の問い方**で、同じ問いに対する両世界正解率と、両世界で正しい根拠文を読んだ率が、ともに各重み・4負荷条件のすべてで90%以上。初見語順は別判定。失敗は「元の場面を書かなかった」「書いたが問いの番地が不一致」「番地は合ったが後から上書き」「正しい場面なのに答えを復号できない」に分ける。距離をさらに延ばす前に、どこが律速かを測る。

**必要になった場合だけ行う切り分け。** 根拠文の位置を教えても主要判定に不合格なら、「RWKVの文表現から手掛かりを読めない」のか「6ビットの番地へ自力でまとめられない」のかが残る。その場合に限り、同じ文表現・同じ訓練例で、訓練中に人物と状況の教師ラベルも与える追加方式を試す。2軸各8クラスの予測値から64番地の1か所を直接選ぶ。評価時にはラベルも文字列照合も使わない。これは構造と教師を与えた到達点であり、最終回答だけからの自力獲得とは呼ばない。

**結果：一番地だけを使う実装は動いたが、隠れ状態だけからは書く番地と読む番地が揃わなかった。** 訓練に使った48組は正解文・問い・背景文の各々で全48組が出現し、除外した16組の出現はいずれも0。単語自体は既知で、言い換え試験は同じ単語を並べ替えるもの。各セルは初見の組に対する「両世界で答えが正しい件数／両世界で元の文を選べた件数」で、**各指標の分母は240**（40家族×2問×3重み）。通常RWKVは個別の文を選ばないので回答だけを載せる。

|訓練時の教師・読み方|4事実・干渉0|4事実・干渉1,024|16事実・干渉0|16事実・干渉1,024|
|---|---:|---:|---:|---:|
|最終回答のみ・一番地|6 / 26|2 / 7|2 / 8|1 / 2|
|＋根拠位置・一番地|26 / 49|21 / 41|13 / 23|10 / 20|
|＋根拠位置と人物・状況・一番地|6 / 15|3 / 7|6 / 14|3 / 7|
|通常RWKV＋学習した回答ヘッド|1|0|1|0|

各方式とも凍結したCE42/43/44（各504,960パラメーター）、同じバッチ32世界×2問・960更新。局所文のRWKVは8 tokenごとに初期化し、文の実BF16隠れ出力を取り出してからFP32で平均を取った。正規化は訓練観測のみで推定。追加学習は一番地方式17,813個、通常RWKV側18,028個。アドレス方式の永続状態は1世界につき64×8個のFP32（2,048 byte）と64個の使用フラグ。実行時は各文で1候補の書き込み、各問で1スロットの読み。通常側は全履歴を連続して通すので状態サイズ・FLOPsまで一致する対照ではない。根拠位置あり対回答だけの回答差は4事実・干渉0で **+8.3ポイント**（対応付き95%区間 +3.8～+13.8）。再標本化は地図ではなくこの実験の履歴seed→家族で行い、同じ問い・世界・3重みを一緒に扱った。

**失敗の中身。** 根拠位置付き、4事実・干渉0の個別480回答の内訳は、対象を未書き込み36、問いと書き込みの番地違い211、合った番地の上書き70、正しい文だが復号失敗66、正しい文から正答97。誤った文で偶然答えが一致しても根拠正解にはしていない。対象文を直接指定して復号しても、両世界正答は97/240で、番地選択**だけ**が問題ではない。1,024 tokenの背景では、事実ではない文に17,958/30,720回書き込み、同じ4事実で上書きによる失敗は70→97。根拠が分かること、書いたものを守ること、物を復号することは別の指標である。

条件付き追加実験は実施した。人物3ビット＋状況3ビットを、訓練時の根拠文と問いでのみビット教師とし、推論器・重み数・更新数は変更しなかった。4事実・干渉0で、訓練に含まれる組の新しい履歴では71/240正答だったが、未見の組では6/240。個別480問のうち409問で問いと対象文の番地が違った。§22.20の主要ゲートは全方式・全重みで不合格。したがってこの条件で100万tokenへ延長するより、**干渉がない時点でなぜ局所表現から復号・番地照合ができないのか**を次に調べた。[生の予測・内訳・対応付き区間](experiments/binding-state/episodic-address/panel/)／[条件付き追加実験](experiments/binding-state/episodic-address/factor-panel/)。

## 22.21 番地の学習以前に、文の手掛かりを圧縮しすぎていないか

**追加実験の理由と事前条件。** §22.20の未見の組・4事実・背景なしでも、根拠位置付き学習は両世界正答26/240、正しい根拠を外から直接渡しても97/240だった。人物・状況の教師を追加した場合も6/240にとどまった。距離を延ばすより先に、1文を1ベクトルにする段階で必要語を使えるかを切り分ける。既存の256次元特徴 `平均RWKV隠れ状態128＋末尾隠れ状態128` の後半だけを、**その8語の語彙別出現回数÷8**（128語彙・128次元）に置き換える。全語を同じように数え、辞書から人物・状況・物の役割や番地を引く処理は推論に加えない。重み数、64スロット、一番地読み、訓練例・初期化・960更新は固定する。通常RWKV側にも問いの局所語彙経路を同じように加えるが、問いに物の名前は出ない。

**独立追試。** 上記の結果を見て選んだ特徴変更なので、旧テストでの再採点は採否に使わない。人物・状況の組を外す条件と4負荷を保ち、新しい5 seed（757101/103/107/109/113）×8家族、通常語順の未見組のみで、元の重みと新しい重みを同じ生履歴上で比べる。3重み・4負荷の各々で両世界の回答・根拠とも90%以上というゲートは据え置く。最終回答のみ、根拠文の教師あり、さらに人物・状況の教師ありを混ぜない。この128次元での語の保存は小語彙の利点であり、大語彙で同じコストになるとは仮定しない。

**帰属の追加対照。** 語数経路＋人物・状況教師の方式が追試のゲートを通った場合、それだけでは成功をRWKVに帰属できない。そこでこの方式に限り、同じ物語・教師・更新回数で、RWKVの特徴を一切入力せず128次元の局所語数だけから1番地を学習する対照を別途訓練する。隠れ幅118、17,721パラメーター（複合方式より92少ない）、同じ64スロット・1読出し。新規のタスク達成判定ではなく、すでに使った追試面での機構特異性の確認とする。

**追試結果。** 同じ未見履歴について、表は「隠れ状態のみ → 後半を局所語数に置換」の両世界正答数（各240件。同じ40家族を複数の問い・重みで共有し、240件を独立試行とはしない）。語数を残すだけでは解決しない。大きく改善したのは、**訓練時に人物と状況を明示した場合**だった。

|教師・読み方|4事実・干渉0|4事実・干渉1,024|16事実・干渉0|16事実・干渉1,024|
|---|---:|---:|---:|---:|
|最終回答のみ・一番地|8 → 16|0 → 9|3 → 8|0 → 4|
|＋根拠位置・一番地|19 → 33|18 → 31|16 → 15|14 → 15|
|＋根拠位置と人物・状況・一番地|7 → 238|5 → 227|7 → 238|5 → 227|
|通常RWKV＋回答ヘッド|1 → 1|0 → 0|1 → 0|0 → 0|

表の4方式では、人物・状況教師＋語数経路だけが、3初期値すべてで4負荷の回答・根拠の事前ゲートを通過した。最大負荷でCE42/43/44はそれぞれ78/80、77/80、72/80（両世界で正しい文を読み、その文から正答）。隠れ状態のみとの差は+92.5ポイント、履歴seed→家族の対応付き95%区間は[+85.0, +97.9]。問いだけをすり替えると全負荷で0/240。最大負荷の個別480回答では454で正しい文から正答、4は読み書きの番地違い、22は背景文による上書きだった。書かれなかった対象、正しい文からの復号失敗はともに0。対して「根拠位置だけ＋語数経路」は対象文を直接渡すと232/240正答なのに、自力の番地では15/240。**物の識別と、人物・状況を値に依存しない共通番地にする学習は別の課題**である。[全予測・対応付き区間](experiments/binding-state/episodic-address/replication/)。

**RWKVなしの対照。** 元の文章から直接8語の出現回数を取り、凍結RWKVもそのキャッシュも使わず、同じバッチ・人物と状況の教師・960更新で3初期値を別々に訓練した。17,721個の学習パラメーターと同じ1回読みで、**全4負荷とも両世界の根拠・回答は240/240**。問いをすり替えると0/240。最大負荷で背景への書き込みは複合方式2,010/30,720に対してこちらは0/30,720。よってこの成績は追加パラメーターの多さでなく、適切な手掛かりと教師を持つ一番地記憶で説明でき、**この課題からRWKV固有の利得は主張できない**。公開した固定例では物の交換（リボン／羽根）に対し、複合方式の初期値42は2問中片方のみ正答、他方は両世界で棄権。語数対照は両問とも交換に追従した。[対照の全予測](experiments/binding-state/episodic-address/counts-control/)。

**距離をどう調べるか。** 成功した複合方式と語数のみの対照について、重みを変えず、上と同じ追試40家族・16事実で上書きを切り分ける。世界間で正解は別、問い・末尾は同じであり、ある問いの番地が末尾で一度でも上書きされると、以後その問いには両世界正解できない。各家族の背景文は人物・状況・物の512通り（その家族内で語順は固定）の独立一様抽出なので、全形に実際のゲートと番地を適用して衝突確率を厳密に求める。干渉0/1,024 tokenの通常実行との全問一致を検算し、同じ生成規則の8,192、32,768、100万、1,000万token時点の生存と期待値を出す。有限語彙・固定事実数での厳密な短絡計算であって、1,000万tokenをRWKVに実時間で流した速度測定でも、1,000万個の事実を覚えた試験でもない。

|16事実の後ろの背景token|複合方式：両世界の正答・根拠 / 240|複合方式：分布上の期待正答数 / 240|語数のみ：両世界の正答・根拠 / 240|
|---:|---:|---:|---:|
|0|238|238.0|240|
|1,024|227|229.2|240|
|8,192|217|218.3|240|
|32,768|217|217.0|240|
|1,000,000|217|217.0|240|
|10,000,000|217|217.0|240|

各家族で512形すべてが初めて出揃った位置は最大5,598文（44,784 token）。対象番地に書く形が $m$ 個なら、独立に $G$ 文が追加されても元の文が残る確率は $(1-m/512)^G$。実現した系列ではその $m$ 形の最初の出現位置だけで生存が厳密に決まる。複合方式で最初に正答した238件中、217件では $m=0$、残る21件はすべて816文目までに失われた。初期値別の最終値は78/80、74/80、65/80で、合算90.4%でも**全初期値90%以上ではない**。語数対照は全240件で $m=0$。両方式合計480件で、1,024 token時点の結果を実際の逐次推論と照合した。これは「距離に応じて一様に忘れる」のではなく、**この背景生成器の下で上書きされうる番地と、決して上書きされない番地が混在する**結果である。新事実の追加・言い換え・語彙拡大後にも残るという意味ではない。[問ごとの衝突数・最初の上書き・距離別集計](experiments/binding-state/episodic-address/survival/)。

**再現と次の判別点。** 27個の学習済みヘッド、学習経過、世界ごとの値・選んだ番地・実際の根拠、計測した[CPU観測コスト](experiments/binding-state/episodic-address/feature-costs.csv)、[再現手順](README.md#writing-scenes-before-the-question-is-known)を公開する。今回、事実は最大16件、物は8種類、人物と状況の語彙は重ならない。次に判別力があるのは距離をもう一桁延ばすことではなく、同じ単語で**誰が誰に渡したか**を逆転させる試験である。例えば「AがBに石を、BがAに葉を」と「BがAに石を、AがBに葉を」なら、各文の語数まで一致するが「AがBに渡した物」の正解は逆になる。その上で人物・状況ラベルを与えず、書く番地と読む番地を一致させられるかを測る。これは未実施であり、一般言語理解・複雑な推論の改善は今回の結論に含めない。

## 22.22 基礎と入試風の応用で学び、別形式の中テストで役割を問う

**結果を開く前に固定した計画。** §22.21では語数だけで解けた。今回は各文の語数まで一致させ、「AがBに石を渡す」と「BがAに石を渡す」を区別しないと解けないようにする。人物は渡す側・受け取る側で同じ8名。生成した演習であり、実在の入試問題の引用や、人間の難度尺度への校正ではない。日本語で読める問題用紙とモデルに入れた制御トークン列を両方公開する。

|用途|題材・必要な手順|訓練への使用|
|---|---|---|
|基礎問題|一件の受け渡しを問う。逆方向にも別の物を渡した記録があり、事実の後に仮の話が続く|最終の物だけで学習。二世界|
|入試風の応用問題|記録の訂正と別紙の照合表を使う。最初の答えで次に調べる問いが変わる3段照合|最終の物だけで学習。四世界|
|本番の中テスト|別の校内備品台帳と別形式の問題用紙。最新の確定記録を選び、表で**2段**照合する|問題・正解・途中経路を訓練・選択から除外。四世界|

**テストを何が別物にするか。** 3個の句の並べ方を訓練の3通りと本番の残り3通りに分ける。人物の28通りの無向組のうち8組を、基礎だけでなく応用の全記録・背景・照合表からも除外する。本番では出発と到着の両方に未訓練の組を使う。名前・物・助詞の意味は共通にし、知らない記号を当てさせる問題にはしない。出発記録の主客反転と、最終の有効な訂正の主客反転を独立に行う。同じ問いにこの四世界で全部正答したときだけ成功と数え、出発だけ、最後だけを見た正答と区別する。

**比較する機構。** 同じ公開手順（答えた物に対応するカードの問いを次に読む）を全方式に与える。手順自体を自力で発見したとはしない。通常の凍結RWKV＋回答ヘッド、語数だけの学習番地、RWKV局所平均＋語数の学習番地と、訓練の問いで変化する二つの「語＋目印」を見つけ同じ番地規則を読み書きに使う方式（RWKVあり／なし）を比較する。後者の仮定は「役割の印が語の直後にある」であり、与えないのは人物ラベル・正しい番地・根拠・途中の答えである。記憶は64スロット。問いより先に書き、推論では各段で一つの番地しか読まない。通常RWKVも同じ過去を読み、予測した物に応じて同じ表をたどる。

**学習と判定。** 基礎と応用を同数の更新に使い、各方式・各初期値で32世界×2問、計960更新。全方式で同じ例順、同じ最終正解のみ。訓練時の複数段は中間の予測分布を公開表に沿って周辺化し、推論では最大の予測一つだけをたどる。3初期値（42–44）、本番は独立の5データseed×8家族。末尾の背景0/128文（0/1,024 token）を分け、各初期値・各距離で四世界すべての正答率90%以上、番地方式は全段の根拠も四世界とも90%以上を目安とする。正しい途中の物を外から渡したときの最終成績、1段で止めた場合、語順だけ訓練形式に戻した同じ問題を別集計する。主結果を見てから中テストで調整しない。

---

# 23. Statistical Protocol

synthetic memory tasksは最低5 seed、matched language-model trainingは最低3 seedとする。

報告するものは、

```text
mean
standard deviation
confidence interval
effect size
per-seed values
```

である。

task-level比較にはpaired bootstrapを使用できる。

seed-level aggregateには、

```text
paired permutation test
paired t-statistic
```

のいずれかを効果量と併記する。

単一runのroute visualizationを一般化しない。

---

# 24. Acceptance Criteria

## 24.1 Semantic Acceptance

1. legacy modeが旧結果を許容誤差内で再現する
2. empty slotがreadおよびmatched writeへ参加しない
3. empty slotがoccupied victimより先に選ばれる
4. weak eligibilityがnormalizationだけで強いwriteへ増幅されない
5. diffuse matchがclear matchと同じconfidenceを持たない
6. hard addressとwrite amplitudeが分離されている
7. rejectされたtokenがcontent、age、write count、vitalityを変更しない
8. all-irrelevant条件でread outputが0近傍になる
9. read aggregationがrelevant slot数だけで無制限に増幅しない
10. sequence chunk sizeで意味論が変わらない
11. slotとprojected statesが同じerase/write係数で更新される
12. thresholdが1へ接近して数値不安定になるconfigを拒否する

## 24.2 Numerical Acceptance

1. referenceとacceleratorでforward parityが成立する
2. 全differentiable inputのgradient parityが成立する
3. threshold境界caseのgolden vectorが一致する
4. checkpoint intervalごとのreconstruction budgetを報告する
5. irreversible eventを不安定な逆算で処理しない
6. absolute errorとrelative errorを入力別に評価する
7. ageおよびusage gradientを独立に評価する
8. NaN、Inf、silent clampを許可しない

## 24.3 Scientific Acceptance

1. matched baselineに対するretrieval改善
2. PPLだけでは説明できないrecall改善
3. false readの減少
4. stale readの減少
5. load-factor崩壊点の改善
6. old-fact intrusionの減少
7. false expirationの許容範囲内維持
8. duplicate slotの減少
9. 一部slotまたはmemory operationのcausal contribution
10. 複数seedで方向が再現
11. loss-vs-wall-clockが許容範囲
12. 追加peak memoryが許容範囲

## 24.4 Systems Acceptance

1. headline backendがdevice-nativeで検証されている
2. raw timing sampleが公開される
3. compileとexecutionが分離される
4. full-step throughputが報告される
5. peak memoryが実測される
6. multi-device correctnessとscaling efficiencyが分離される
7. unvalidated backendが暗黙にheadline pathへ選ばれない

---

# 25. Falsification Criteria

次の場合、本設計の中心仮説は支持されない。

```text
PPLは改善するがmemory taskが改善しない

retrievalは改善するが
parameter-matched controlで同等になる

slot数増加だけで改善が説明される

write頻度増加だけで改善が説明される

all-irrelevant readがsoftmax controlと変わらない

novelty calibrationがM変更に対して安定しない

revision accuracy改善と同時に
persistent fact recallが大幅悪化する

adaptive retentionがfixed half-lifeを上回らない

STE temperatureまたはseedで結果が反転する

kernel speedupがfull-stepへ反映されない

memory branchをzeroingしてもlogitが変化しない
```

negative resultも報告対象とする。

---

# 26. Risks

1. absolute thresholdが学習初期にmemory starvationを起こす
2. threshold warm-upが最終的にも低thresholdを固定する
3. capacity calibrationのnull modelが学習後のkey分布と一致しない
4. occupancy導入で初期memory utilizationが遅れる
5. admissionがalways-writeまたはnever-writeへcollapseする
6. duplicate slotが曖昧routeを自己強化する
7. ambiguity penaltyが複数有用slotへのwriteを過剰抑制する
8. empty-first allocationがbank imbalanceを悪化させる
9. age/usage victim policyが将来重要なslotを削除する
10. eviction criticが短いtraining segmentへ過適合する
11. capacity-conserving updateが表現力を制限する
12. free-edit updateがslot normを発散させる
13. hard-forward routingのbiased gradientが不安定になる
14. multi-read tileがduplicate retrieval subspaceを形成する
15. fixed bank partitionがtask分布に合わない
16. vitalityが全slotで0または1へ飽和する
17. token-clock retentionが無関係な長文で過剰忘却する
18. event-clock retentionが一時情報を保持しすぎる
19. grouped projectionがcross-group表現力を低下させる
20. reset journalが高novel-rate taskで肥大化する
21. Pallas kernelの複雑化がwall-clock benefitを消す
22. reference、GPU、TPUの意味論が時間とともにdriftする
23. 複数screened layerが同一情報を重複保存する
24. slot observabilityがinterpretabilityと誤認される
25. occupied slot数の離散変化でcapacity-calibrated thresholdが不連続に変わる

---

# 27. Implementation Priority

## Phase 0: Semantics and Reproducibility

```text
semantics_version
explicit benchmark mode
provenance schema
golden vectors
backend validation status
```

この段階はmodel behaviorを変えない。

## Phase 1: v5 Core Memory Semantics

```text
occupancy state
empty-first allocation
safe threshold parameterization
capacity-calibrated read/write threshold
bounded read aggregation
ambiguity-aware matched confidence
```

## Phase 1 Decision Gate

Phase 1はportable referenceを先に実装し、accelerator kernel、checkpoint redesign、
adaptive retentionへ直ちに展開しない。v4 predecessor、no-screening baseline、
parameter-matched controlと比較し、synthetic memory taskは最低5 seed、
matched small language-model runは最低3 seedで評価する。

次を満たす場合のみPhase 2以降へ進む。

```text
semantic acceptanceとchunk invarianceが成立する

false read、load-factor collapse、revisionまたはstale readの
少なくとも一つでv4を再現可能に上回る

改善がparameter数、compute増加、write頻度だけでは説明されない

seed、threshold、slot countを変えても改善方向が反転しない

事前に定めたloss、throughput、peak-memory budget内に収まる
```

このgateを満たさない場合はv5 coreをproduction pathへ昇格せず、
negative resultとablationを保存して設計を凍結または棄却する。

## Phase 2: Continuous Memory Editing

```text
hard address / continuous amplitude
erase/write separation
capacity-conserving mode
new accounting semantics
```

## Phase 3: Capacity Management

```text
redundancy metric
revised victim score
optional future-use critic
write-budget regularization
```

## Phase 4: Checkpoint Redesign

```text
interval-aware stability budget
erase/write tape
irreversible-event handling
reset journal or recompute path
```

## Phase 5: Adaptive Retention

```text
vitality
half-life policy
retention clock
retention causal interventions
```

## Phase 6: Grouped Retention

```text
grouped latent state
block-separable projections
group-specific decay
new Pallas kernels
```

Phase 5以降は、v5 coreがmatched experimentsで有効と確認された場合にのみproduction candidateとする。

---

# 28. Claim Boundary

実装完了時に主張できるのは次である。

> State-Level Screeningを、明示的occupancy、容量校正されたabsolute relevance、曖昧性を考慮したcompetitive routing、hard addressとcontinuous memory editingの分離、および数値安定性を明示した固定容量online memoryとして再設計した。

実験なしに次を主張してはならない。

```text
long-context recallを改善した
memory hygieneを改善した
KDAより優れている
softmax attentionを置換できる
human-readable memoryを獲得した
hard eviction問題を解決した
production trainingに利用可能である
```

v5 retentionを実装した場合でも、主張は次に限定する。

> KDAおよびGated Delta系研究に着想を得たadaptive retentionとerase/write分離を、projected-state consistencyを保つslot memoryへ統合した。

KDAの直接再実装とは呼ばない。

---

# 29. References

1. Ken M. Nakanishi, *Screening Is Enough*, [arXiv:2604.01178](https://arxiv.org/abs/2604.01178), 2026.
2. Bo Peng et al., *RWKV-7 “Goose” with Expressive Dynamic State Evolution*, [arXiv:2503.14456](https://arxiv.org/abs/2503.14456), 2025.
3. Bo Peng et al., *RWKV: Reinventing RNNs for the Transformer Era*, [arXiv:2305.13048](https://arxiv.org/abs/2305.13048), 2023.
4. Ashish Vaswani et al., *Attention Is All You Need*, [arXiv:1706.03762](https://arxiv.org/abs/1706.03762), 2017.
5. Albert Gu and Tri Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, [arXiv:2312.00752](https://arxiv.org/abs/2312.00752), 2023.
6. Yutao Sun et al., *Retentive Network: A Successor to Transformer for Large Language Models*, [arXiv:2307.08621](https://arxiv.org/abs/2307.08621), 2023.
7. Songlin Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*, [arXiv:2406.06484](https://arxiv.org/abs/2406.06484), 2024.
8. Songlin Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule*, [arXiv:2412.06464](https://arxiv.org/abs/2412.06464), 2024.
9. Kimi Team et al., *Kimi Linear: An Expressive, Efficient Attention Architecture*, [arXiv:2510.26692](https://arxiv.org/abs/2510.26692), 2025.
10. Ali Hatamizadeh, Yejin Choi, Jan Kautz, *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention*, [arXiv:2605.22791](https://arxiv.org/abs/2605.22791), 2026.
11. Ali Behrouz et al., *It’s All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization*, [arXiv:2504.13173](https://arxiv.org/abs/2504.13173), 2025.
12. Ali Behrouz et al., *Titans: Learning to Memorize at Test Time*, [arXiv:2501.00663](https://arxiv.org/abs/2501.00663), 2025.
13. Simran Arora et al., *Zoology: Measuring and Improving Recall in Efficient Language Models*, [arXiv:2312.04927](https://arxiv.org/abs/2312.04927), 2023.
14. Nelson F. Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, [arXiv:2307.03172](https://arxiv.org/abs/2307.03172), 2023.
15. Cheng-Ping Hsieh et al., *RULER: What’s the Real Context Size of Your Long-Context Language Models?*, [arXiv:2404.06654](https://arxiv.org/abs/2404.06654), 2024.
16. Alex Graves et al., *Neural Turing Machines*, [arXiv:1410.5401](https://arxiv.org/abs/1410.5401), 2014.
17. Alex Graves et al., *Hybrid Computing Using a Neural Network with Dynamic External Memory*, [arXiv:1605.06065](https://arxiv.org/abs/1605.06065), 2016.
18. Francesco Locatello et al., *Object-Centric Learning with Slot Attention*, [arXiv:2006.15055](https://arxiv.org/abs/2006.15055), 2020.
19. Yoshua Bengio et al., *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation*, [arXiv:1308.3432](https://arxiv.org/abs/1308.3432), 2013.
20. L. A. Belady, *A Study of Replacement Algorithms for a Virtual-Storage Computer*, IBM Systems Journal 5(2), 1966.
