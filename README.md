Domain: Support

# Swiggy Support Decision Intelligence

Capstone project, Minor in Artificial Intelligence and Data Science, IIT Mandi.

All data in this repository is synthetic. It comes from `generate_data.py` and does not reflect Swiggy's actual data, products, policies or performance. The brand is only the setting for the decisions.

Everything is written in plain Python with NumPy. Every reported number is computed by code in this repository, and the committed files in `outputs/` are the exact output of the scripts.

## How to run

From a clean clone, in this order:

```bash
git clone https://github.com/SrishtiKesh2705/Swiggy-Support-Decision-Intelligence
cd Swiggy-Support-Decision-Intelligence
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

- `generate_data.py` must print `800 10 170 7cb48ef89914049d76954530b6482e04`.
- `part4.py` reads `knn_best_k` from `outputs/part2_results.txt`, so `part2.py` must run before `part4.py`.
- `common.py` holds shared helpers (data loading, the 560/240 split, standardisation, logistic regression, metrics, file writers). Importing it does not run any Part.
- All results are written to `outputs/`.

## Part 1 findings

**R1.1 Hypothesis test.** The null hypothesis (mean resolution time = 8.0 minutes) is rejected: `z_stat` = 2.8930, which is larger than the critical value 1.96. A Type I error here would mean concluding that average resolution time differs from the planned 8 minutes when the staffing assumption is in fact right. The gap is `mean_value` - 8.0 = 8.9382 - 8.0 = 0.9382 minutes, about 12 percent above plan; the 95 percent interval is 8.3026 to 9.5738 minutes, so it excludes 8.0. Statistical significance is not the same as practical size: whether about one extra minute per ticket matters depends on ticket volume, where it adds up to many staffing hours. The spread is also large (`sd_value` = 9.1723), so individual tickets vary a lot around the average.

**R1.2 Sample size.** `n_required` = 1293 tickets to estimate the mean to within 0.5 minutes at 95 percent confidence, but only 800 rows are available. The current margin of error is about 0.64 minutes (1.96 x 0.3243), so the planning team would need roughly 500 more tickets to reach the target precision.

**R1.3 Gradient descent.** `gd_slope_gap` = 0.0000. Squared-error cost for a straight line has a single minimum, which is exactly the least-squares line, so gradient descent walks down to it (the cost is unchanged between update 100 and update 500: 40.3698).

**R1.4 Funnel check.** `resid_sd_ratio_raw` = 3.7525 and `resid_sd_ratio_log` = 1.0396. The raw model shows a strong funnel: residual spread in the upper half of fitted values is about 3.75 times that of the lower half. The log model removes it, since a ratio near 1 means roughly constant spread.

**R1.5 Multiple regression.** The predictors with an absolute coefficient above 0.05 are ln(X4) (1.1576), X2 (0.0940), X3 (0.1067) and X6 (0.1518). X1 (-0.0074), X5 (0.0028) and X7 (0.0172) fall below the threshold. The 0.05 rule is mechanical and is not a ranking of importance, because the coefficients are in different units. `mr_adj_r2` = 0.7403 against `loglog_adj_r2` = 0.6561, an improvement of about 0.084, so the extra predictors earn their place.

## Part 2 findings

**R2.1 Learning rates.** After 200 updates the losses, from lowest to highest, are eta = 1.0 (0.3703), eta = 0.1 (0.3776) and eta = 0.01 (0.5091). A learning rate that is too small is still far from the minimum after the same number of updates, while a large one converges fastest here (it was stable on this standardised data, though a large rate can overshoot on other problems).

**R2.2 Weights and odds ratios.** The three largest absolute weights are X3 (0.9013), X1 (-0.5554) and X2 (0.5505). The largest odds ratio is X3 at 2.4629: each extra training standard deviation of SLA already consumed multiplies the odds that a ticket escalates by about 2.5, other predictors held fixed.

**R2.3 Accuracy and thresholds.** `blind_accuracy` = 0.7875 against 0.8042 at tau = 0.5, so the model is only 1.7 points better than one that never flags anything; accuracy hides how many escalations are missed when only one record in five is positive. Lowering tau from 0.5 to 0.3 raised recall from 0.4706 to 0.7451 and lowered precision from 0.5455 to 0.4634. Accuracy at tau = 0.3 (0.7625) is below the blind accuracy, which again shows accuracy is a weak score here.

**R2.4 Logistic vs KNN.** The logistic model at tau = 0.3 has F1 = 0.5714 and recall = 0.7451. K-nearest neighbours (best K = 5 by cross-validation) has F1 = 0.3696 and recall = 0.3333. Logistic regression is clearly better on both.

**R2.5 Deployment.** I would deploy the logistic model at tau = 0.15, the threshold with the lowest cost on the training rows (training cost INR 12,150). On the test rows this costs INR 6,050 (`cost_test_at_tau`), against INR 11,800 at tau = 0.5 (`cost_test_at_0.5`) and INR 20,400 for flagging nobody (`cost_test_flag_nobody`). The threshold was chosen on training rows because choosing it on the test rows would tune it to the very data used to judge it and make the reported cost optimistic.

## Part 3 findings

**R3.1 K-means and the elbow.** The four successive WCSS drops are WCSS(2) - WCSS(3) = 2262.6753 - 1602.8471 = 659.8282, WCSS(3) - WCSS(4) = 1602.8471 - 1494.7732 = 108.0739, WCSS(4) - WCSS(5) = 1494.7732 - 1323.7157 = 171.0575 and WCSS(5) - WCSS(6) = 1323.7157 - 1278.1691 = 45.5466. The elbow rule gives `elbow_k` = 3: the drop from K = 3 to K = 4 (108.0739) is far below half of the drop from K = 2 to K = 3 (329.9141), so a fourth cluster adds little. The drops do not shrink steadily (the drop from 4 to 5 is larger than the drop from 3 to 4), because the pinned starting rows give one reproducible solution, not necessarily the global optimum. The remaining analysis uses K = 3.

**R3.2 Segments.** Segments are ordered by ascending mean SLA consumed (X3). Rows are from `part3_segments.csv`.

- **Segment A, "Quick, low-friction tickets"** (322 tickets): SLA consumed 0.2250, 1.95 earlier tickets, mean resolution 3.5976 minutes, `outcome_rate` 0.0186. Action: keep these in the self-service or auto-resolution lane and skip supervisor review, since almost none escalate.
- **Segment B, "Repeat-contact, high-spend, slow tickets"** (255 tickets): 6.07 earlier tickets, the highest account value (about INR 28,719 per year), the slowest mean resolution (17.1248 minutes), `outcome_rate` 0.2353. Action: route to experienced agents and aim to resolve at first contact, reading the customer's earlier tickets before replying.
- **Segment C, "Already-waiting tickets"** (223 tickets): SLA consumed 0.7118 when the ticket reaches an agent, the lowest satisfaction (38.5215), `outcome_rate` 0.4664. Action: move them to the front of the queue and trigger supervisor review early, before the SLA runs out.

The segment with the highest escalation rate is segment C, with `outcome_rate` = 0.4664.

**R3.3 Hierarchical clustering.** Merge heights are single 1.3, 1.4, 2.8, 2.9, 3.4; complete 1.3, 2.0, 2.8, 2.9, 7.9; average 1.3, 1.7, 2.8, 2.9, 5.6556.

- Single linkage, cut in its largest gap (1.4, between merges 2 and 3), gives 4 clusters: {E1}, {E2, E3, E6}, {E4} and {E5}.
- Complete linkage, cut in its largest gap (5.0, between merges 4 and 5), gives 2 clusters: {E1, E4, E5} and {E2, E3, E6}. Average linkage gives the same 2 clusters.

The two rules disagree because single linkage uses the closest pair, so it keeps chaining cohorts together through nearest neighbours and its biggest jump comes early, right after E2, E3 and E6 join. Complete linkage uses the farthest pair, so it keeps clusters compact and only shows a large jump at the very end, when the two natural groups are forced together.

**R3.4 PCA.** The mean vector of the standardised data is all zeros, so centring changes nothing. The first component is driven by X1 tenure (+0.5317), X3 SLA consumed (-0.5187), X5 satisfaction (+0.4418) and X2 earlier tickets (-0.4100), with X4 account value contributing less (-0.2916): it contrasts long-tenure, satisfied customers on one side with tickets that have used up their SLA and have repeat contacts on the other. The first component alone explains 0.5227 of the variance, and `components_for_80pct` = 3 (cumulative 0.7593 with two components, 0.8616 with three). `mean_pc1_segment_A/B/C` = 1.7773, -0.7831 and -1.6709, so the segments are ordered along the first component in the same order as their escalation rates (0.0186, 0.2353, 0.4664): segment A sits at the calm end and segment C at the high-risk end.

## Part 4 findings

**R4.1 Splits and overfitting.** The best split by both information gain and Gini gain is S4 (X1 <= 24 months of tenure), with information gain 0.1252 and Gini gain 0.0606 on the 560 training rows (root entropy 0.7462, root Gini 0.3347). It is best because it cuts impurity the most: 188 short-tenure tickets escalate 45.7 percent of the time (`stump_rate_yes` = 0.4574) against 8.9 percent for the other 372 (`stump_rate_no` = 0.0887). A tree grown until every leaf is pure would memorise noise in the training rows and fail on new tickets, so it overfits. One rule that stops a tree growing too deep is a maximum depth, or a minimum number of rows required in a leaf.

**R4.2 Model choice.** Test results at tau = 0.3 (from `part4_voting.csv`):

| Model | F1 | Recall |
|---|---|---|
| logistic | 0.5714 | 0.7451 |
| knn | 0.3696 | 0.3333 |
| stump | 0.5532 | 0.7647 |
| hard_vote | 0.5344 | 0.6863 |
| soft_vote | 0.5507 | 0.7451 |

I would ship the logistic model: it has the highest F1 (0.5714) and the highest precision (0.4634), and it is the model whose threshold was chosen with the INR cost rule. Combining models did not help: hard vote (F1 0.5344) and soft vote (F1 0.5507) both score below the logistic model alone, because the weak KNN model (F1 0.3696) drags the votes down. The stump has the highest recall (0.7647), but at the price of lower precision and F1. It is a one-question model using tenure only, so it is worth noting that it comes close to the logistic model.

**R4.3 AdaBoost.** After round 1 the two misclassified points (points 3 and 7) hold a total normalised weight of 0.5, up from 0.25. Round 2 trusts h2 more than h1 because its weighted error (0.1667) is lower than h1's (0.2500), which gives it a larger alpha (0.8047 against 0.5493). The ensemble classifies 6 of the 8 points correctly (`ensemble_accuracy` = 0.7500). The out-of-bag share is `oob_fraction` = 0.3676, close to 1/e.

**R4.4 Architecture sizes.**

Model A, option 2 (convolutional network), input 28 x 28 x 1:

| Layer | Output shape (height x width x depth) | Parameters |
|---|---|---|
| Conv 5 x 5, 6 filters, stride 1, padding 2 | 28 x 28 x 6 | 156 |
| ReLU | 28 x 28 x 6 | 0 |
| Max-pool 2 x 2, stride 2 | 14 x 14 x 6 | 0 |
| Conv 5 x 5, 16 filters, stride 1, padding 0 | 10 x 10 x 16 | 2,416 |
| ReLU | 10 x 10 x 16 | 0 |
| Max-pool 2 x 2, stride 2 | 5 x 5 x 16 | 0 |
| Flatten | 400 | 0 |
| Dense, 32 neurons, ReLU | 32 | 12,832 |
| Dense, 10 neurons (output) | 10 | 330 |
| **Total** | | **15,734** |

The fully connected alternative (784 to 16 to 16 to 10) has 12,560 + 272 + 170 = 13,002 parameters (`mlp_total`). Convolution scales because filters are shared across positions: one fully connected layer on a 128 x 128 x 3 input with 1,024 neurons needs 50,332,672 parameters (`fc_layer_params`), while a convolutional layer with 64 filters of size 3 x 3 on the same input needs only 1,792 (`conv_layer_params`).

Model B (recurrent network, 64-dimensional inputs, 128-dimensional hidden state, 16 classes): W_xh is 128 x 64, W_hh is 128 x 128 and W_hy is 16 x 128, giving `rnn_total` = 26,768 parameters. This number does not grow with longer sequences because the same weights are reused at every time step.

For 10 classes I would use ReLU in the hidden layers and softmax in the output layer with categorical cross-entropy loss.

A very deep plain network can stop learning in its early layers because the gradient is multiplied by many small numbers on its way back, so it shrinks towards zero (the vanishing gradient). A skip connection adds the layer input to its output, giving the gradient a direct path to earlier layers.

## Model-choice memo

**To:** Head of Customer-Support Operations
**From:** Support Analytics

**Recommendation:** ship the logistic-regression model, and flag a ticket for supervisor review when its predicted escalation probability is 0.15 or higher.

**Why this threshold:** a missed escalation costs about INR 400 and a needless review about INR 50, so we chose the cutoff that minimises that cost on training tickets only (0.15, training cost INR 12,150). On 240 held-out test tickets it costs INR 6,050. The default cutoff of 0.5 costs INR 11,800 and flagging nobody costs INR 20,400, so the model roughly halves the cost of the default and cuts about 70 percent from doing nothing. At a cutoff of 0.3 it catches 74.51 percent of escalations (recall 0.7451). K-nearest neighbours was clearly weaker (F1 0.3696 against 0.5714 for logistic), and combining models by voting did not beat the logistic model alone. Only the logistic model was costed with our INR rule; the other models were not costed that way.

**Risk:** the evidence rests on one train/test split with only 51 escalations in the test set, and the data is synthetic, so the INR figures are indicative and the threshold should be re-checked on live tickets before it is relied on.

**Next build:** Model B, the ticket-text sorter, is worth building before Model A because it is the main use case, and the photo model can wait.

## Limitations

1. The data is synthetic and was generated from known formulas, so real Swiggy tickets would be noisier and the INR costs and thresholds would need to be re-estimated.
2. Everything rests on one fixed 560/240 split with only 51 positive test records, so test metrics and the chosen threshold could change with a different split, and the best K in cross-validation is valid for that split only.
3. K-means uses pinned starting rows, which gives a reproducible but not necessarily globally optimal solution, and the hierarchical clustering was run on a small supplied table of six cohorts.
4. The cost rule (INR 50 per false alarm, INR 400 per miss) is an assumed average, and the logistic model assumes a linear effect of each predictor on the log odds.