# RWKV7M 全ブランチ技術調査

実装構造 数値互換性 実験証拠と検証計画

調査日 2026年10月3日  対象 Rumia-Channel/RWKV7M

本稿のブランチ先端・ahead／behind・実装仕様は、調査時点の固定コミットを基準に記述する。mainは004d7b39、最新研究版は591b1ac9を対象とし、本レポート追加後のブランチ先端とは区別する。

**結論**　RWKV7Mは、RWKV-7の再帰コアに固定容量の副メモリを接続する研究実装である。JAX／Flax NNXへの移植、数値境界の明示、GPU／TPU向け実行基盤には具体的な成果がある。一方、追加メモリが長文検索や言語モデル品質を改善するという中心仮説は、公開された証拠では確立していない。

調査対象の全4ブランチのうち、最新の研究はmainから66コミット進んだ未統合ブランチにある。mainは閾値付きslot readとEMA write、最新v5は占有状態・新規登録・置換を持つオンライン辞書へ発展している。両者を同じモデル仕様として扱うと、実装の意味も評価結果も取り違える。[M1](https://github.com/Rumia-Channel/RWKV7M/commit/004d7b39b5006ba2c6ce93f92c68a9c826605ba1) [M2](https://github.com/Rumia-Channel/RWKV7M/branches) [E1](https://github.com/Rumia-Channel/RWKV7M/commit/591b1ac9f6340bae672ecc85899beb8a064702b3) [E2](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/docs/state_level_screening_v5_design.md)

### 判断に直結する所見

| 項目 | 判断 |
| --- | --- |
| 採用できる根拠 | 固定容量の状態、NNX学習経路、限定条件の数値parity、実機記録は確認できる。 |
| 最優先の検査 | v5のwindow単位admission quotaは後続tokenで過去の採否が変わる。分割方法でも候補数が変わる。 |
| 品質上の現状 | 7月22日のv5試験はメモリ利用の縮退と極小のmemory-off効果を報告。7月26日のquota変更の成功証拠ではない。 |
| 推奨する進め方 | 因果性と再開整合性を先に固め、quota無効の基準系で同一予算・複数seedの比較を行う。 |

**検証範囲**　本調査ではコード・履歴・公開記録の照合と、依存の小さいローカル再現を行った。完全なJAXモデルの学習、GPU／TPU性能、v5の最終logit差は再測定していない。以下では「実装確認」「作者報告」「今回の再現」「提案」を区別する。

## 1 対象ブランチと実装の発展

2026年10月3日に公開されている4ブランチを確認した。mainの先端は7月14日15時13分UTCの統合コミットであり、日本時間では7月15日となる。新しい研究ブランチは7月26日11時03分UTCまで進んでいる。[M1](https://github.com/Rumia-Channel/RWKV7M/commit/004d7b39b5006ba2c6ce93f92c68a9c826605ba1) [M2](https://github.com/Rumia-Channel/RWKV7M/branches) [E1](https://github.com/Rumia-Channel/RWKV7M/commit/591b1ac9f6340bae672ecc85899beb8a064702b3)

| ブランチ | 先端SHA | mainとの関係 |
| --- | --- | --- |
| main | 004d7b39 | 基準となる統合版 |
| agent/explicit-dtype-contract | 37594422 | ahead 0 / behind 1<br>ファイルツリーはmainと同一 |
| agent/blackwell-parity-repro | c798ce94 | ahead 0 / behind 10<br>PR5で統合済み |
| agent/screening-pallas-benchmarks | 591b1ac9 | ahead 66 / behind 0<br>未統合の研究版 |

古い2ブランチは独立した最新候補ではない。explicit-dtype-contractとmainのGit treeは同じc5bffddfで、差はmerge commitの履歴だけである。blackwell-parity-reproは数値互換性とNNX実機検証の途中段階を残す。

### mainから最新研究版へ

7月13〜14日にはupstream RWKV-7とのforward／backward／optimizer比較、NNX学習、モデル並列、dtype契約、Pallas WKVが統合された。その後の研究版ではScreeningのPallas化と投影済みメモリの保持、v5 core、AMD上の検証、勾配診断、counterfactual評価へ進んだ。7月26日の変更はwrite admissionをwindow内quotaで制御する。

命名も区別が必要である。工学的な「v2」実装が論文上のv4 predecessorを指す場合があり、ファイル名の数字だけで意味論を判断できない。v5-coreは実装済みだが、v5-retentionは設計段階として拒否される。[E2](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/docs/state_level_screening_v5_design.md) [E9](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/src/rwkv7m/model/nnx_model.py#L1720-L1850)

### 公開状況の限界

公開APIで確認したPR1〜5は7月13〜14日に全てmerge済みで、issue検索、release一覧、GitHub Actions run一覧は空だった。外部CIや未公開実験がないという証明ではない。README、研究草稿、コードには更新時期の差があり、「Linen参照のみ」「custom kernel未実装」といった古い説明を現行仕様へ転記しない。

## 2 RWKVの再帰コアと副メモリの接続

公開APIの主経路はFlax NNXである。token列はembedding、複数のRWKV block、最終LayerNorm、語彙headを通る。embeddingと語彙headは別パラメータで、token間のT×T attention行列や位置embeddingはこの経路にない。[M3](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/model/nnx_model.py#L1162-L1212)

### RWKV行列状態の更新

1 headの状態行列Aは行がvalue、列がkeyである。dは入力依存decay、k̂は正規化key、aは学習率gate、kは調整済みkey、vはvalue、rは読み出しベクトルとする。実装の向きに合わせると更新の中心は次式で表せる。[M4](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/model/rwkv_core.py#L54-L85) [P1](https://arxiv.org/html/2503.14456v2)

$$
d_t = \exp(-\exp(w_t))
$$

$$
A_t = A_{t-1}\operatorname{diag}(d_t) + \bigl(A_{t-1}(-\hat{k}_t)\bigr)(\hat{k}_t\odot a_t)^{\mathsf T} + v_t k_t^{\mathsf T}
$$

$$
y_t = A_t r_t
$$

decayに加えて、状態依存のrank-one修正とvalue-keyのouter productが入る。RWKV-7の一般化delta ruleとその計算能力の理論は原論文の貢献であり、RWKV7Mのslot追加による新たな証明ではない。TimeMixの後にはheadごとの正規化、補正、gate、出力射影が続く。ChannelMixは1-token shift、線形射影、squared ReLU、線形射影である。

### Screeningの挿入点

対象層では、まず通常RWKV blockが入力xからh_baseとネイティブ状態を計算する。Screeningはx、h_base、独立したslot状態から残差を作り、h_baseへ加える。slotは更新前にreadされ、その後同じtokenからwriteされる。[M3](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/model/nnx_model.py#L1162-L1212) [M5](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/model/nnx_model.py#L1005-L1159)

ScreeningはWKV行列を引数に受けず、WKVの成分を直接マスクする機構ではない。後段層には補正後のhiddenが入るため、後段のRWKV状態へは間接的に影響する。第0層valueを後段へ渡すv_firstは層間のvalue residualであり、別の会話履歴キャッシュではない。

### 学習と逐次生成

prefillとdecode_oneは同じ種類の状態を受け渡す。mainの標準学習ではbatch間のcarryは既定で無効だが、同一系列内のrecurrent chunk境界でstop-gradientを入れない。語彙headも分割できるものの、全activationが一定量になるわけではない。[M11](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/train/nnx_train.py#L165-L357)

## 3 mainのScreening読み書き

各対象層にはM個の幅Sのslotがある。slot内容は系列ごとの実行時状態で、別の学習可能slot_embedがslotの識別性を与える。readは正規化したqueryとslot keyのcosine類似度から独立したrelevanceを計算する。[M5](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/model/nnx_model.py#L1005-L1159) [M6](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/model/screening.py#L21-L139)

$$
q_t = \operatorname{unit}\!\left(W_q\operatorname{LayerNorm}(x_t)\right),\qquad k_m = \operatorname{unit}(W_k s_m)
$$

$$
\tau = 2\sigma(\theta)-1,\qquad \operatorname{rel}_m = \left[\operatorname{ReLU}\!\left(\frac{q_t\cdot k_m-\tau}{1-\tau+\varepsilon}\right)\right]^2
$$

slot間softmaxを使わず、全slotを読まない状態を表せる。集約zはvalueの重み付き和で、TanhNormにより集約のノルムを抑え、gateと出力射影を通してh_baseへ足す。TanhNormのcapは出力射影や学習可能scaleの後の最終残差を一律に上限保証しない。全slotを射影してから係数を掛けるため、relevanceが0でも計算を自動で省略するわけではない。

### writeはphaseで意味が変わる

$$
\delta_m = \tanh\!\left([\operatorname{LayerNorm}(x_t),\,h_{\mathrm{base}},\,e_m]W_\delta+b\right)
$$

$$
s_m \leftarrow s_m + \mu_{\mathrm{bank}}\,g_m\,(\delta_m-s_m)
$$

read_screening_onlyではg=1の通常EMA更新を行う。名前にread-onlyとあっても書き込み停止ではない。read_writeかつwrite screening有効では、別query/keyで得るwrite relevanceとwrite_rel_floorの大きい方をgに使う。既定floor=10⁻³なので、関連度0でも微量更新が残る。候補δに入るeは現在slotではなくslot identityである。

| bank | 初期μ | 基礎EMAの半減期 |
| --- | --- | --- |
| short | 0.025 | 約27 tokens |
| mid | 0.010 | 約69 tokens |
| long | 0.0025 | 約277 tokens |

この半減期はwrite relevanceを掛ける前の値であり、意味情報の保持寿命ではない。half-lifeを指定したbankでは対応するμを直接計算する。ageはwrite screening時だけ更新され、mainのusageは主に観測用である。leaky warmupやage maskを使う場合も、標準設定と分けて解釈する。

## 4 最新v5のオンライン辞書設計

研究版のscreening-v5-coreは、明示的に占有されたslotから読む固定容量のオンライン辞書へ再設計されている。主な変更は保存先の選択と入れ替えを定義した点にある。mainの全slotに小さく書くEMAと同じ更新則ではない。[E2](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/docs/state_level_screening_v5_design.md) [E3](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/src/rwkv7m/model/screening_v5.py#L506-L859)

| 構成 | v5での役割 |
| --- | --- |
| occupancy | 空slotをreadから除外する。占有情報のない旧状態を暗黙変換しない。 |
| 容量較正threshold | 占有数、tile数、key幅、目標偶然一致率から閾値を作り、学習offsetを加える。 |
| matched write | 既存slotへの一致度と集中度で、曖昧なmatchの更新を弱める。 |
| novel admission | 既存slotに合わない候補を保存対象にするか判断する。 |
| allocationとeviction | bankを選び、空きがあれば割り当て、なければage・usageなどからvictimを選ぶ。 |
| eraseとwrite | 保持・消去と新規候補の書き込みを別係数で表す。提供例はtied更新。 |

$$
s^{\prime} = (1-e)s+w\delta
$$

提供0.185B例は4 read tiles、rank64のcandidate、value幅でのgateなどを使う。hard-forward／soft-backwardのroutingは離散選択を近似勾配で学習する方式であり、hard objectiveの厳密な勾配ではない。容量較正もGaussian近似に基づき、学習後に相関したkey/queryのfalse-positive率を保証するものではない。[E6](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/configs/rwkv7m-0.185b-screening-v5-core.json.example) [P7](https://arxiv.org/abs/1308.3432)

### 実装経路と状態互換性

v5-coreはportable reference recurrenceへ進み、v4 Pallas bodyをそのまま使わない。branch名にPallasが含まれていても、v5全体が専用accelerator kernelで動くとの解釈は誤りである。phaseはread_writeを要求し、v4 checkpointのoccupancy欠落を拒否する。v5-retentionは未実装として明示的に停止する。[E8](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/src/rwkv7m/model/nnx_model.py#L1420-L1445) [E9](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/src/rwkv7m/model/nnx_model.py#L1720-L1850)

slotの投影済みread key／value／write keyをrecurrence内で保持するため、永続stateだけから作業メモリは見積もれない。redundancy計算を有効にした場合はslot間の二乗項も入る。標準例では最新quotaが有効、PI admission controllerは無効で、両者の同時使用は拒否される。

## 5 admission quotaの因果性と分割依存

**最優先で確認すべき性質**　最新v5設定例のadmission quotaは、128-token window全体のscoreをstable sortし、上位ceil(128×0.05)=7件を候補にする。後続tokenのscoreが過去tokenの採否を変えられる。適用は学習時だけに限定されていない。[E4](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/src/rwkv7m/model/nnx_model.py#L249-L277) [E5](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/src/rwkv7m/model/nnx_model.py#L1618-L1662) [E6](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/configs/rwkv7m-0.185b-screening-v5-core.json.example)

### 今回のローカル再現

リポジトリのquota helperをNumPyの代替演算で実行し、同じ先頭121件のscoreを保ったまま末尾7件だけを高くすると、先頭tokenのadmission maskが1から0へ変わることを確認した。また同一128件を異なる呼び出し単位に分けると次の差が出た。

| 呼び出し方 | admission候補数 | 候補率 |
| --- | --- | --- |
| 128 tokenを一度に処理 | 7 / 128 | 5.47% |
| 64 tokenずつ2回 | 8 / 128 | 6.25% |
| 1 tokenずつ128回 | 128 / 128 | 100% |

これは最終的なwrite件数ではない。候補maskの後にnovel判定、allocation、edit係数などが作用する。それでも、入力prefixだけでは決まらない候補選択が存在し、prefill／chunk／逐次decodeで同じ再帰計算になるとの前提は成立を確認できない。

**再現の限界**　この検査はhelperの選別意味論を確認したもので、完全なJAXモデルのlogit差、学習lossへの影響量、最終品質低下を測ったものではない。「全v5が非因果」「全branchが使えない」とも結論しない。対象はquotaを有効にした特定経路である。

### 品質評価前に置く合格条件

1. 同じprefixに異なるsuffixを付けても、prefix時点のstateとlogitが変わらないことを確認する。
2. full、複数chunk幅、1-token scanで、logit・各状態・勾配を比較する。
3. teacher-forced評価と自己回帰生成で同じ意味のモデルを評価していることを確認する。

まずquota無効の因果的な基準系を作る。必要なら、過去の累積量だけを参照するtoken budgetや逐次threshold制御を別方式として評価する。window全体の上位選択を維持したまま、同じ計算を因果的に再現できると仮定してはいけない。

## 6 数値互換性と実機性能の証拠

**本節の測定値は作者報告であり、今回の独立再測定ではない。** BF16の重み・計算、FP32のWKV状態・update・optimizer moments・gradient accumulationを分ける契約は、移植の比較条件を明確にしている。全処理をBF16化した実装ではない。[M9](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/model/wkv.py#L16-L133)

### 数値parityが示す範囲

Blackwellの7月13日記録は、RWKV-LM-V7の固定SHA 665472da、固定BF16重み、3 seedでlogit最大絶対差0.00581〜0.00670、gradient cosine約0.999996、69 mapped tensorの通過を報告する。optimizerは1 stepでupdate cosine約0.998、relative L2約0.059〜0.062である。長期の学習軌道や追加Screeningの品質を保証する試験ではない。[M10](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/docs/gpu_blackwell_validation.md) [P8](https://github.com/RWKV-Vibe/RWKV-LM-V7)

同機のLightning／DeepSpeed launchは失敗しており、公式training stack全体の成功とは読めない。L40Sの別記録はupstream direct trainingを3〜4 step、localモデルを1 stepからrestore後2 stepまで確認する。これもcheckpoint実行性の証拠である。

### 計測境界を揃えて性能を読む

| 測定対象 | 作者報告 | 条件と解釈 |
| --- | --- | --- |
| L40S WKV<br>7月14日 | 0.537対8.109 ms<br>15.09倍 | forward＋backward<br>T128 B1 H12 N64 |
| L40S 0.185B<br>同日 | 3,181対764 tok/s<br>4.17倍 | full step中央値<br>WKV倍率と分ける |
| TPU v5e WKV<br>同日 | 0.538対0.766 ms<br>1.42倍 | WKVのみ<br>7B実行の証拠ではない |
| 研究版L40S<br>7月16日 | 11,561 / 9,708 /<br>10,365 tok/s | no-screen / read / write<br>legacy compute-only |

main側のcanonical FFN2688では、no-screening 9,378、read-only 3,875、read-write 3,747 tok/sで、追加Screeningの負担が大きい。一方、7月16日の高速化は投影済みlegacy recurrenceの結果であり、portable v5の速度へ転用できない。後者はcompile、転送、checkpoint、loggingなどを除く。[M14](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/docs/gpu_l40s_pallas_performance.md) [M15](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/docs/tpu_pallas_performance.md) [E10](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/docs/gpu_l40s_pallas_performance.md)

backend名の対応だけで実機検証済みとは言えない。CPU interpret、単体kernel、full graph、モデル全体、multi-hostを別gateにする。逆再構成からstate tapeへの変更後は、旧速度と旧メモリ値を再測定する必要がある。

## 7 メモリ品質に関する正と負の証拠

**finiteなlossやgradientは、メモリが使われていることを示さない。** 最新研究版はnegative resultを記録しており、数値動作とmemory-quality gateを分けて読むことができる。[E2](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/docs/state_level_screening_v5_design.md) [E11](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/docs/gpu_mi300x_training_validation.md)

| 作者報告の試験 | 観測 | 解釈 |
| --- | --- | --- |
| v5 0.185B<br>7月22日旧profile | 2,000 step / 32.768M tokens<br>train loss 20.891→4.954<br>423/423 gradient leaf finite | 1/16 slotへ縮退<br>memory-off差は約+9×10⁻⁶ |
| v5 0.3B<br>同時期 | 400 step / 3.2768M tokens<br>480/480 leaf finite | 集約利用率3.125%<br>memory-off差は約−4×10⁻⁶ |
| delayed retrieval<br>checkpoint205 | held-out streaming<br>accuracy 1/256<br>loss差 +1.04×10⁻⁴ | chance水準<br>logitへの効果だけでは<br>retrieval改善を示さない |

最初の2行のmemory-offは、同じMiniPile評価batchでメモリを消すcounterfactualであり、独立したheld-out generalization比較ではない。residual/base RMSも0.185Bで9.42×10⁻⁶、0.3Bで5.37×10⁻⁷と小さい。単にlossが下がったことを追加メモリの寄与へ帰属できない。

### 7月26日の変更との時間関係

旧profileではall-write、empty、all-writeへの遷移、memory-on／offでloss・accuracy・logitが一致する試験、極大gradientも報告された。その後のPI controllerなどの回復策は実機検証待ちと記録され、最終HEADではさらにPIを無効にしquotaを有効にしている。旧collapseを最新quota構成で再現済みとは言えず、quotaで解決したとも言えない。[E6](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/configs/rwkv7m-0.185b-screening-v5-core.json.example) [E11](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/docs/gpu_mi300x_training_validation.md)

### AMDの安定性は別の未完了項目

固定checkpointとbatchを使う切り分けでは、Pallas backward単体はfiniteでも、複数backwardが同居するfull graphで非有限値が出た。ROCm／Triton loweringやaliasing、scheduling相互作用が候補で、原因を単一kernel数式へ限定していない。autoはPallas forward＋reference VJPへ退避したが、新hybridの実機full-step確認は残る。CPU interpret成功で代替できない。[E11](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/docs/gpu_mi300x_training_validation.md)

mainには長文専用評価や予算整合比較の不足がある。研究版にはdelayed retrieval、answer-only loss、document reset、memory-off評価経路があるため、「評価harnessが一切ない」と一般化しない。

## 8 再現した問題と検証範囲

quotaの検査に加え、依存の小さい実moduleまたは原文helperの抽出で次を確認した。いずれも完全なtrainerやモデル生成を走らせた再現ではない。

| 問題 | 対象 | 確認できた挙動と対処 |
| --- | --- | --- |
| 再開後のbest保護 | mainと研究版 | best_evalを復元せず初期化。古いbestがrotation対象になり得る。best metadata復元と保護テストを追加。 |
| EOS decode | mainと研究版 | encodeのEOS=0を含めてdecodeするとKeyError 0。特殊tokenと停止条件を定義。 |
| optimizer override | main | model-config／preset経路で明示lrが元値のまま。研究版では修正済み。 |
| quotaの未来依存 | 研究版の有効設定 | 選別maskがsuffixと分割幅で変わる。品質評価前に因果性gateを置く。 |

resume問題は偽checkpointを用いたhelper試験で再現し、呼び出し配線を照合した。重要なbest weightsはrotation対象と別に保全する。EOS問題は実tokenizerで再現したが、実モデルが0を生成する頻度は未測定である。optimizer overrideはmainのhelperと代替configで0.123指定が0.001に残ることを確認した。[M16](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/cli/train_binidx_distributed.py#L550-L560) [M17](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/distributed/checkpoint.py#L279-L302) [M18](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/tokenizer/rwkv_tokenizer.py#L98-L123) [M19](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/cli/config.py#L5-L46) [E14](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/src/rwkv7m/cli/train_binidx_distributed.py#L752-L761)

### 今回の実行範囲

全4ブランチのPython AST parseはmain 106、dtype 106、Blackwell 95、研究版143ファイルで成功。JSON exampleは順に8、8、8、13件でparseできた。これは構文検査で、実行成功やschemaの正しさを保証しない。mainの比較shellはbash -nを通過し、tokenizerのASCII・日本語・emoji・空文字roundtrip、binidxの小型fixtureも確認した。

実行環境はPython 3.12.14、NumPy 2.3.5で、project要求のPython 3.13以上やJAX／Flax／Optax／pytestを備えていない。pytest collect-onlyはNo module named pytestで起動できなかった。これはテストの失敗件数ではなく、実行環境の不足である。数値full suite、GPU／TPU、学習は未実行。

### 未再現の実機欠陥を区別する

mainのPallas backwardはdecayで割る逆再構成を使い、decayが0へ丸められると未定義になる。研究版はFP32のtoken別state tapeへ変更した。これは作者が発見・修正を記録した欠陥であり、今回のGPU独立再現ではない。[M20](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/kernels/wkv_pallas_gpu.py#L221-L226) [E13](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/src/rwkv7m/kernels/wkv_pallas_gpu.py)

テスト関数の静的定義数はmain146、研究版283。pytest実行数ではない。文書の140、134、91、279 passed等は版や時点が異なる。通常のupstream parity unit testも毎回公式CUDAを動かすものではなく、synthetic archiveを使う検査を含む。[M23](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/tests/test_upstream_rwkv7_parity.py)

## 9 状態量とハードウェアの見積もり

mainでB=batch、L=層数、H=head数、N=head幅、C=HN、Lₛ=Screening層数、M=slot数、S=slot幅とする。FP32のWKV状態、time／channel shift、slot、age、usageだけなら次式になる。[M7](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/model/presets.py#L9-L160) [M8](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/model/state.py#L5-L60)

$$
\text{state bytes} = 4B\left[L(HN^2+2C)+L_s(MS+2M)\right]
$$

| main preset | L / C | 総parameter | Screening分 | 状態MiB<br>1 sample |
| --- | --- | --- | --- | --- |
| 0.185B | 12 / 768 | 184,985,222 | 1,028,486<br>0.56% | 2.328 |
| 0.3B | 13 / 1024 | 297,738,764 | 4,078,092<br>1.37% | 3.383 |
| 1B | 28 / 1536 | 985,479,192 | 18,329,112<br>1.86% | 10.922 |
| 3B | 34 / 2560 | 2,943,319,064 | 49,717,784<br>1.69% | 22.071 |
| 7B | 32 / 4096 | 6,994,788,376 | 122,802,200<br>1.76% | 33.250 |

全main presetはvocab 65,536、head幅64、slot数16、bank配分8/4/4。総parameterは定義値、Screening分と状態量は形状からの算術であり、ピークメモリ実測ではない。v5設定には流用しない。

### 固定状態と学習メモリは別

状態式は系列長Tに依存しないが、重み、logit、activation、backward保存、optimizer、通信bufferを含まない。7BのBF16重みだけで約13.03 GiB、repoの18 bytes／parameter見積もりでは全体約117.26 GiBになる。shardingで割った値も実機でのfitを保証しない。[M21](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/distributed/scaling.py#L104-L157)

v5の永続Screening状態はoccupancyを加え4BM(S+3) bytesだが、recurrence内の投影済みkey／value、time方向の候補、state tapeが追加される。redundancyを有効にしたslot間計算にはMの二乗項もある。

### 計算量と実時間

通常blockはtokenあたり概ねC²＋CF＋HN²に低rank項を加えた規模で、語彙headにはC×vocabの仕事がある。Screeningもdense gateや候補射影を追加する。Tに対する線形性は、Transformerより常に速いことや、Tを伸ばしてもレイテンシが一定という意味ではない。

小型single-deviceから始め、compile時間とsteady-state、prefillとdecode、tokens/sとpeak memory、学習と推論を別に測る。7Bやmulti-hostは、同じrevisionのcheckpoint再開と有限性を確認した後の拡張対象とする。

## 10 独自性を評価するための比較設計

独自性の候補は、RWKV-7と固定容量のScreening副メモリを組み合わせ、read拒否・write・新規登録・置換を具体的に設計した点にある。外部メモリ、delta rule、eraseとwriteの分離、straight-through routingそのものには先行研究がある。[P1](https://arxiv.org/html/2503.14456v2) [P5](https://arxiv.org/abs/2605.22791) [P6](https://www.nature.com/articles/nature20101) [P7](https://arxiv.org/abs/1308.3432)

### Multiscreenから何を受け継ぐか

直接の着想元Screening Is Enoughはtoken文脈へのquery/key選別を行う。RWKV7Mは圧縮済みslotを読む。調査時点のMultiscreen v4ではthresholdは0〜1の範囲だが、mainのτは−1〜1で、距離SoftmaskやMiPEとも異なる。9月29日版v4との比較は最新文献との照合であり、7月の実装がその版を使ったという歴史的主張ではない。[P2](https://arxiv.org/html/2604.01178v4)

標準RWKV-7は元々slot-softmaxを行わない。そのためsoftmaxの一般的な欠点を挙げるだけでは、元モデルより良くなる根拠にならない。slot容量や追加parameterによる効果と、絶対thresholdによる効果を分離する。

### 決定力のある対照実験

| 切り分ける要因 | 比較する条件 |
| --- | --- |
| 追加容量と機構 | core-only、read、read/write、同等parameterのMLP／EMA memory |
| 選別則 | hard threshold、sigmoid、softmax、null-option softmax、random read |
| 保存と置換 | write floor 0と既定値、bank凍結、write抑制、slot削除／shuffle |
| 予算差 | 同token数に加えparameter、FLOPs、state bytes、実時間を揃えた対照 |
| 長文能力 | MQAR、複数needle、delayed key-value retrieval、類似distractor |

長さ、キー数、情報位置、遅延、distractorの類似度を独立に動かし、複数seedの平均と分散を報告する。単一needleやlossだけでなく、答えの正確さ、無関連時の誤read、必要情報の未保存、memory-offによる変化を測る。slot可視化だけでは因果的な寄与を示せない。[P3](https://arxiv.org/abs/2312.04927) [P4](https://arxiv.org/abs/2404.06654)

### 引用と配布上の確認

研究版草稿のDNC引用に付くarXiv:1605.06065は別論文で、正しいDNC原著はNature DOI 10.1038/nature20101である。本体はBSD 2-Clauseだが、比較upstreamはApache-2.0である。vocabulary、派生コード、データ、重みを一括してBSD扱いせず、配布前に由来と必要表示を確認する。[E7](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/RWKV7M.paper.md) [P6](https://www.nature.com/articles/nature20101) [M13](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/LICENSE) [P9](https://www.apache.org/licenses/LICENSE-2.0)

## 11 優先順位と合格条件

次の順序なら、長時間学習を増やす前に、評価の意味を変える問題と運用上の損失を先に取り除ける。以下は提案する検証計画であり、完了済みの成果ではない。

| 優先 | 作業 | 合格の証拠 |
| --- | --- | --- |
| P0 | quota有効時の因果性<br>と実行分割の同値性 | prefix不変性、full／chunk／tokenのlogit・state・gradient。失敗経路は無効化または因果的方式へ変更。 |
| P0 | checkpoint再開の保護 | 過去bestのmetadataが復元され、keep_lastが小さくても保護される。継続runとの選定一致。 |
| P1 | EOSと実効config | 特殊tokenのdecode／停止、指定lrなどの反映、保存configと起動引数の一致。 |
| P1 | AMD full graphの有限性 | 固定checkpoint／batchでhybridのmulti-step finite、reference parity、restore後の再確認。 |
| P2 | memory-quality gate | 独立held-out retrievalでchanceを超え、複数seedとmemory-off・予算対照で一貫する増分。 |
| P3 | 最新revisionの性能 | 品質gateを通る設定のend-to-end速度、peak memory、安定性。小型からmulti-hostへ拡張。 |

### 評価データと再開条件を固定する

small CLIのperiodic evalはtrain fileを使い、distributed CLIもeval-data-file未指定ならtrain dataへfallbackする。held-out splitの明示が必要である。mainのstandalone PPLはexp(min(loss,20))なので、極端なlossでcapされる表示も記録する。[M22](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/cli/train_binidx.py#L226-L269)

memoryの長距離保持を評価するなら、documentごとにresetし、同じ文書のchunk間は正しくcarryする。通常のchunkごとzero-stateのlossでは、境界を越えた有用性を測れない。resumeではscheduler horizonも照合し、stepとoptimizerの復元だけで同じ学習計画へ戻ると仮定しない。

### 採用判断

現状は、再帰モデルの実装・数値境界・accelerator挙動を調べる研究基盤として有用である。長文記憶の改善を目的に採用する場合は、まず上のP0とP1を通し、その後にP2で追加メモリの寄与を判断する。kernel速度や学習lossの下降を、モデル品質の代わりに合格条件へ置かない。

## 12 再現手順と記録すべき成果物

以下は実行準備の手順であり、本調査で全手順を実行したという意味ではない。依存導入、データ取得、GPU／TPU実行は環境と費用を確認してから行う。

### ソースと設定を固定する

```bash
git clone https://github.com/Rumia-Channel/RWKV7M.git
cd RWKV7M
git checkout 004d7b39b5006ba2c6ce93f92c68a9c826605ba1
git status --porcelain
```

研究版は別checkout／worktreeでSHA 591b1ac9f6340bae672ecc85899beb8a064702b3へ固定する。古い2ブランチはc798ce949292c5ec68eb3416b7a9773e2927963cと375944224787fc858a20e7b9fe8de77815f0fc36で比較する。branch名だけを保存しない。

### 実行環境を分離する

local JAX側はPython 3.13以上、uv.lockと対応するJAX 0.10.x／Flax 0.12.7系を固定する。CUDA、ROCm、TPUは異なるextraとdriver条件を持つ。公式PyTorch baselineは別のPython 3.12環境に置き、PyTorch、DeepSpeed FusedAdam、CUDA toolkit、C++ toolchain、Python headers、upstream SHAとpatchを記録する。[M12](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/README.md) [M10](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/docs/gpu_blackwell_validation.md)

比較scriptの--no-runは完全なdry-runではなく、fetch、データ取得、環境準備、依存同期を伴う。読み取り検査と実行準備を分ける。

### 小さい検査から進める

1. 構文・config・tokenizer・状態shape・checkpointを確認し、pytestの収集数とskip理由を保存する。
2. reference、NNX、各backendのforward／backward／updateを固定入力で比較する。
3. quotaを切った基準系と有効系を分け、prefixとchunkの同値性を検査する。
4. 小型モデルでrestoreを含むmulti-stepと独立held-out評価を行う。
5. 合格したrevisionだけで性能と大規模化を測定する。

### 必ず残す情報

commit、実効config、seed、tokenizer／data hash、split、token数、batch／chunk／carry／reset規則、dtype、hardware／driver、warmup・同期・計時範囲、loss定義、評価結果、best／latest checkpoint、resume条件を一緒に保存する。単体kernelとfull trainingの数値を同じ列へ混在させない。

公開コードには追跡文書があるが、生ログ、checkpoint、ROCm capture等の一部は一時ホストやgitignore対象で、全てをコードだけから検算できる状態ではない。学習済みweightやmodel card、releaseも本調査では確認できない。研究実装を動かすことと、有用な学習済みモデルを取得することは分けて計画する。

## 出典 実装と原論文

本文の参照番号はクリック可能。Mはmain固定版、Eは研究版固定版、Pは原論文または比較対象。アクセス確認日 2026年10月3日。

- **M1**: [main 固定コミット](https://github.com/Rumia-Channel/RWKV7M/commit/004d7b39b5006ba2c6ce93f92c68a9c826605ba1)

- **M2**: [公開ブランチ一覧](https://github.com/Rumia-Channel/RWKV7M/branches)

- **M3**: [main APIとNNXモデル接続](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/model/nnx_model.py#L1162-L1212)

- **M4**: [WKVのFP32参照更新](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/model/rwkv_core.py#L54-L85)

- **M5**: [main Screeningのreadとwrite](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/model/nnx_model.py#L1005-L1159)

- **M6**: [Screening数式と設定](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/model/screening.py#L21-L139)

- **M7**: [モデルpresetとパラメータ定義](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/model/presets.py#L9-L160)

- **M8**: [状態定義と初期化](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/model/state.py#L5-L60)

- **M9**: [dtypeとWKV API境界](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/model/wkv.py#L16-L133)

- **M10**: [Blackwell検証記録](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/docs/gpu_blackwell_validation.md)

- **M11**: [NNX chunk学習](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/train/nnx_train.py#L165-L357)

- **M12**: [READMEの再現と評価条件](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/README.md)

- **M13**: [本体ライセンス](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/LICENSE)

- **M14**: [L40S Pallas性能記録](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/docs/gpu_l40s_pallas_performance.md)

- **M15**: [TPU Pallas性能記録](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/docs/tpu_pallas_performance.md)

- **M16**: [resumeとsummaryの初期化](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/cli/train_binidx_distributed.py#L550-L560)

- **M17**: [checkpoint rotation](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/distributed/checkpoint.py#L279-L302)

- **M18**: [TokenizerとEOS](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/tokenizer/rwkv_tokenizer.py#L98-L123)

- **M19**: [mainのCLI override](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/cli/config.py#L5-L46)

- **M20**: [mainのGPU backward逆再構成](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/kernels/wkv_pallas_gpu.py#L221-L226)

- **M21**: [学習メモリ見積もり](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/distributed/scaling.py#L104-L157)

- **M22**: [学習中評価のデータ選択](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/src/rwkv7m/cli/train_binidx.py#L226-L269)

- **M23**: [upstream parityのunit test](https://github.com/Rumia-Channel/RWKV7M/blob/004d7b39b5006ba2c6ce93f92c68a9c826605ba1/tests/test_upstream_rwkv7_parity.py)

- **E1**: [最新実験ブランチ固定コミット](https://github.com/Rumia-Channel/RWKV7M/commit/591b1ac9f6340bae672ecc85899beb8a064702b3)

- **E2**: [v5の設計契約と実験記録](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/docs/state_level_screening_v5_design.md)

- **E3**: [v5のreadとallocationとwrite](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/src/rwkv7m/model/screening_v5.py#L506-L859)

- **E4**: [admission quota helper](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/src/rwkv7m/model/nnx_model.py#L249-L277)

- **E5**: [admission quotaの適用箇所](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/src/rwkv7m/model/nnx_model.py#L1618-L1662)

- **E6**: [v5 core 0.185B設定例](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/configs/rwkv7m-0.185b-screening-v5-core.json.example)

- **E7**: [実験ブランチ研究草稿](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/RWKV7M.paper.md)

- **E8**: [v5状態とcheckpoint契約](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/src/rwkv7m/model/nnx_model.py#L1420-L1445)

- **E9**: [v5のportable reference経路](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/src/rwkv7m/model/nnx_model.py#L1720-L1850)

- **E10**: [研究版L40S性能記録](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/docs/gpu_l40s_pallas_performance.md)

- **E11**: [MI300X学習と失敗の記録](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/docs/gpu_mi300x_training_validation.md)

- **E12**: [quota検査とchunk検査](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/tests/test_screening_v5.py#L730-L952)

- **E13**: [研究版GPU backend](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/src/rwkv7m/kernels/wkv_pallas_gpu.py)

- **E14**: [研究版resume初期化](https://github.com/Rumia-Channel/RWKV7M/blob/591b1ac9f6340bae672ecc85899beb8a064702b3/src/rwkv7m/cli/train_binidx_distributed.py#L752-L761)

- **P1**: [RWKV-7 Goose 原論文 v2](https://arxiv.org/html/2503.14456v2)

- **P2**: [Screening Is Enough v4](https://arxiv.org/html/2604.01178v4) — 2026年9月29日版

- **P3**: [Zoology](https://arxiv.org/abs/2312.04927)

- **P4**: [RULER](https://arxiv.org/abs/2404.06654)

- **P5**: [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)

- **P6**: [Differentiable Neural Computer 原著](https://www.nature.com/articles/nature20101)

- **P7**: [Straight through estimator](https://arxiv.org/abs/1308.3432)

- **P8**: [RWKV-LM-V7比較対象](https://github.com/RWKV-Vibe/RWKV-LM-V7)

- **P9**: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
