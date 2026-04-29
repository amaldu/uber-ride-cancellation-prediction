# Project Walkthrough

This documents explains the step-by-step logic and workflow following CRISP-DM adapted. The goal of the project is to predict Uber ride cancellations using machine learning to enable proactive intervention by creating strategies to reduce the overall cancellation rate.

---

## Index
- [Project Walkthrough](#project-walkthrough)
  - [Index](#index)
- [Problem Framing](#problem-framing)
  - [1.1 Defininition of the objective in business terms](#11-defininition-of-the-objective-in-business-terms)
  - [1.2 How will the solution be used?](#12-how-will-the-solution-be-used)
  - [1.3 What are the current solutions/workarounds (if any)? There are none](#13-what-are-the-current-solutionsworkarounds-if-any-there-are-none)
  - [1.4 How should the problem be framed](#14-how-should-the-problem-be-framed)
  - [1.5 How should performance be measured?](#15-how-should-performance-be-measured)
    - [Primary objective](#primary-objective)
    - [Per-prediction expected value](#per-prediction-expected-value)
    - [Operational capacity constrain \& its strategy](#operational-capacity-constrain--its-strategy)
    - [Business constraints to accept (or reject) the trained model](#business-constraints-to-accept-or-reject-the-trained-model)
    - [Final metrics:](#final-metrics)
  - [1.6 What are comparable problems? Can you reuse experience or tools?](#16-what-are-comparable-problems-can-you-reuse-experience-or-tools)
  - [1.7 Is human expertise available?](#17-is-human-expertise-available)
  - [1.8 How would you solve the problem manually?](#18-how-would-you-solve-the-problem-manually)
  - [1.9 List the assumptions made so far. Verify assumptions if possible](#19-list-the-assumptions-made-so-far-verify-assumptions-if-possible)
- [2. Get the Data](#2-get-the-data)
  - [2.1 List the data](#21-list-the-data)
  - [2.2 Storage requirements for the dataset:](#22-storage-requirements-for-the-dataset)
  - [2.3 Check legal obligations](#23-check-legal-obligations)
  - [2.4 Access authorizations](#24-access-authorizations)
  - [2.5 Sensitive data analysis (PPI):](#25-sensitive-data-analysis-ppi)
- [3. EDA insights](#3-eda-insights)
  - [3.1 Size and type of data](#31-size-and-type-of-data)
  - [3.2 Cleaning and preprocessing steps](#32-cleaning-and-preprocessing-steps)
  - [3.3 Train/test split strategy](#33-traintest-split-strategy)
  - [3.4 Conclusions of the EDA](#34-conclusions-of-the-eda)
    - [Temporal features](#temporal-features)
    - [Identifier columns](#identifier-columns)
    - [vehicle\_type](#vehicle_type)
    - [pickup\_location and drop\_location](#pickup_location-and-drop_location)
    - [Route interaction (pickup × drop)](#route-interaction-pickup--drop)
    - [avg\_vtat](#avg_vtat)
    - [Missingness mechanism](#missingness-mechanism)
    - [Deterministic rules](#deterministic-rules)
- [4. Pipeline Architecture](#4-pipeline-architecture)
    - [Pipeline 1: Logistic Regression](#pipeline-1-logistic-regression)
    - [Pipeline 2: LightGBM or XGBoost](#pipeline-2-lightgbm-or-xgboost)
  - [4.1 Modelling strategy options](#41-modelling-strategy-options)
  - [4.2 Baseline](#42-baseline)
  - [4.3 Class imbalance](#43-class-imbalance)
  - [4.4 Calibration](#44-calibration)
  - [4.5 Cross-validation strategy](#45-cross-validation-strategy)
  - [4.6 Feature availability at inference](#46-feature-availability-at-inference)
- [5. Model Tuning](#5-model-tuning)
  - [5.1 Search strategy](#51-search-strategy)
  - [5.2 Hyperparameter ranges](#52-hyperparameter-ranges)
  - [5.3 Reproducibility](#53-reproducibility)
  - [5.4 Model selection rule](#54-model-selection-rule)
- [6. Evaluation](#6-evaluation)
  - [6.1 Metrics on the test set](#61-metrics-on-the-test-set)
  - [6.2 Comparison against baselines](#62-comparison-against-baselines)
  - [6.3 Error analysis](#63-error-analysis)
  - [6.4 Subgroup performance](#64-subgroup-performance)
  - [6.5 Stability of the profit estimate](#65-stability-of-the-profit-estimate)
  - [6.6 Sanity checks before declaring the model done](#66-sanity-checks-before-declaring-the-model-done)
- [7. Deployment](#7-deployment)
  - [7.1 Serving architecture](#71-serving-architecture)
  - [7.2 Feature availability at inference](#72-feature-availability-at-inference)
  - [7.3 Rollout strategy](#73-rollout-strategy)
  - [7.4 Rollback criteria](#74-rollback-criteria)
  - [7.5 Causal uplift — open assumption](#75-causal-uplift--open-assumption)
- [8. Monitoring](#8-monitoring)
  - [8.1 Health of the data](#81-health-of-the-data)
  - [8.2 Model performance](#82-model-performance)
  - [8.3 Business impact](#83-business-impact)
  - [8.4 Concept drift](#84-concept-drift)
  - [8.5 Retraining trigger](#85-retraining-trigger)
---

# Problem Framing
## 1.1 Defininition of the objective in business terms
Uber offers a booking service in an Indian metropolitan area and provided me with data from all the bookings of 2024. I identified that the dataset contains 170,000 bookings of which the 32% never reached completion status, that includes different cancellation type based on reason of cancellation or lack of driver available.

**[After EDA]** The dataset actually contains 150.000 rows. This means almost 1 cancellation for every 3 rides

In terms of business impact, I can think of the following areas:

1. Financial impact:
   1. Revenue loss
   2. Hidden operational costs like the time and fuel wasted by drivers while the ride is not cancelled yet, the costs from the processing platform and the potential churn of frustrated customers
   3. Opportunity cost from resources within the company like customer support, refunds, managing complains, etc.

2. Operational impact:
   1. Imbalance in supply & demand. While drivers are booked and removed from the pool, this can artificially increase the price of the available services and potentially damaging the image of the company
   2. Driver insatisfaction if the customer is cancelling their rides too often

Let's imagine that Uber wants to reduce the cancellation rate by 10%, this means 3.700 more rides completed and almost 75000$ recovered so building a predictive model that identifies bookings with high cancellation risk at the time of booking could help achieving it.

## 1.2 How will the solution be used?

The model will be deployed as a real-time prediction system integrated into Uber's booking workflow so when a customer requests a ride, the model will score the cancellation probability.
This will help other departments of the company to develop and implement long-term fixes like:
- Increase customer engagement on high-risk rides by sending booking confirmation messages, provide more frequent ETA updates, or offering loyalty points
- Optimization of the algorithm for driver allocation, if driver rating is a strong indicator we can redistribute or prioritize better drivers more often

**[After EDA]** Waiting time is a strong cancellation indicator so a good idea would be to redistribute drivers to keep a low waiting time in every area

## 1.3 What are the current solutions/workarounds (if any)? There are none

## 1.4 How should the problem be framed

I'm going to start framing it as a supervised binary classification problem: cancelled vs. completed. In future approaches it would be interesting to try a multi-class approach using columns that show the reason for cancelling

The training will be offline on historical batch data, re-trained daily and monitor concept drift to check if it requires faster adaptation and we have to change to near-online or online training

The inference will be in real time

**[After EDA]** The distribution of cancellation rate and total rides over the year did not show clear patterns so I would strongly to advocate for offline training + daily re-train as the first option

## 1.5 How should performance be measured?

After having a chat with Product and Ops teams I define a cost matrix:

| Outcome | What happens | Cost/Benefit | Business Meaning |
|------------|----------------|----------------|------------------|
| True Positive (TP) | Ride is cancelled & system intervenes | +$15 | Prevented cancellation saves $20 revenue, minus $5 intervention cost |
| False Positive (FP) | Ride is NOT cancelled & system intervenes | -$5 | Unnecessary intervention cost |
| True Negative (TN) | Ride is NOT cancelled & system does NOT intervene | $0 | Ride completes normally, no model contribution |
| False Negative (FN) | Ride is cancelled & system does NOT intervene | -$20 | Lost booking revenue + driver idle time + customer dissatisfaction |

A missing a cancellation is 4x more costly than a false alarm!

### Primary objective

What stakeholders care about is to maximize the annual aggregate profit:

```
Annual profit = TP x $15 - FP x $5
```

It's important to remember that this formula only evaluates what the model did, not what it failed to do so FNs are not counted here

### Per-prediction expected value

In every single ride with a predicted cancellation probability P. I compare the expected value of both actions:

```
EV(intervene) = P x (+$15) + (1 - P) x (-$5) = 20P - 5
EV(don't intervene) = P x (-$20) + (1 - P) x ($0) = -20P

Intervene when EV(intervene) > EV(don't intervene):
20P - 5 > -20P
40P > 5 -> P > 0.125
```
So I am interested in an intervention on any ride with if P(cancellation) > 12.5%

### Operational capacity constrain & its strategy

The Ops team said that the system can handle a maximum of 70K interventions/year and based on the historical data it's clear that I should expect more. Also the deployment would be in real time so a top-k global ranking is not possible because not all rides are available at once

To calibrate the deployment threshold, I sort the validation set by predicted probability from highest to lowest and find the predicted score at the 70Kth position, that will be the model probability score I will use at serving time. As a rough guide 70K / 150K = 46.7% of rides would be flagged

### Business constraints to accept (or reject) the trained model

I have oversimplified the calculations to see the min recalls, not the target because FP = 0 until I train the model

1. System viability (model building + maintenance cost: $50K/year)

```
Required TP x $15 >= 50.000 -> TP >= 3.334
Recall >= 3.334 / 48.000 = 7%
```

2. Target ROI (defined above 10% cancellation reduction)

```
0.1 x 48.000 = 4.800 rides to prevent
4.800 x $20 revenue = $96.000 target

Required TP x $15 >= 96.000 TP >= 6.400
Recall >= 6.400 / 48.000 = 13%
```

3. Min precision
```
P x $15 - (1 - P) x $5 >= 0 -> 20P >= 5
Precision >= 25%
```

4. Final Expected profit
```
Net profit = TP x $15 - FP x $5 >= $96.000
```

### Final metrics:

Based on the analysis above, the full set of (business and technical) metrics is:

1. Decision threshold = 12.5%
2. Recall >= 13%
3. Precision >= 25%
4. F2-score to model ranking - > β² = FN/FP = $20/$5 = 4 weights recall 4x over precision
5. PR-AUC to compare models because I have class imbalance
6. Expected profit = TP x $15 - FP x $5 >= $96.000

## 1.6 What are comparable problems? Can you reuse experience or tools?

No because it's the first project in this company :)

## 1.7 Is human expertise available?

There should be but this is just a ML checklist so I'm going to guess a lot:

The dataset provides domain knowledge, including documented cancellation reasons (wrong address, driver-related issues), vehicle-type performance metrics, and observable patterns related to payment methods.

From the company I would probably be able to obtain general industry knowledge of ride-sharing operations like common cancellation triggers in transportation services, established operational best practices, and typical customer behavior patterns in on-demand mobility platforms, etc.

Data science expertise is also available in the topics of binary classification modeling, imbalanced datasets, engineering temporal and behavioral features, etc.

However, certain knowledge gaps that come to my mind would need to be addressed like Uber operational constraints, regional factors affecting rides, clear definitions of peak hours and the impact of surge pricing...

## 1.8 How would you solve the problem manually?

Heuristic approach with rules I can think of:

1. Time-Based Rules:
   - Late night bookings (11 PM - 5 AM): higher cancellation risk
   - Rush hour bookings: the driver availability issues
   - Weekend vs. weekday patterns

2. Location-Based Rules:
   - Known problematic pickup locations (poor GPS, restricted access)
   - Long-distance routes: higher driver cancellation
   - Airport/station pickups: the customer plan changes

3. Vehicle Type Rules:
   - Premium vehicles (Premier Sedan): lower cancellation
   - Budget options (Auto, eBike): customer price sensitivity

4. Customer History (if available):
   - Previous cancellaton history
   - Rating patterns
   - Payment method reliability

5. Real-Time Factors:
   - High VTAT (Vehicle Time to Arrival): Customer impatience
   - Surge pricing active: the customer may cancel after seeing final price

But this approach has limitations because its rules are static and they cannot capture complext interactions

**[After EDA]** Most of these heuristic rules could not be validated but there are three deterministic patterns that work as hard rules inside a two-stage system:
- `avg_vtat ≤ 2.9 min` → never cancels
- `avg_vtat > 15 min` → always cancels (likely a system timeout)
- `vtat_missing = 1` → always cancels (vehicle never arrived)

My approach would be to flag these three cases deterministically and apply the ML model only on the remaining rides

## 1.9 List the assumptions made so far. Verify assumptions if possible

**[After EDA]** Assumptions marked as **confirmed**, **refuted** & **unverified** (could not be tested)

Data Assumptions:
1. The 2024 dataset is representative of typical booking patterns **refuted** 31st of Dec is missing
2. Cancellation reasons are accurately recorded **unverified** dataset is from kaggle
3. All relevant data is captured in the dataset **refuted** columns and rows don't match the document
4. Patterns in 2024 will persist into future periods **unverified** dataset is from kaggle
5. Features available at booking time don't include post-booking information **refuted** and dropped
6. Class distribution reflects realistic cancellation rate **unverified** dataset is from kaggle
7. Target variable is correctly defined and labelled **refuted** more labels than needed, collapsed into binary
8. Grouping `Incomplete` rides with `Completed` as class 0 is a valid modelling choice **unverified** driver no-shows arguably belong on the cancelled side, would need a product/ops conversation to confirm

Business Assumptions:
1. Proactive measures can actually prevent cancellations **unverified** this project does not contain modelling phase yet
2. Intervention costs are lower than cancellation costs **confirmed**
3. Customers will respond positively to interventions **unverified** dataset is from kaggle

Technical Assumptions:
1.  All features used in training will be available at inference time **unverified** dataset is from kaggle
2.  Model can score bookings within acceptable time (<100ms) **unverified** but any ride booking service can hire me and change this
3.  Deployment infrastructure exists or can be built **unverified** but any ride booking service can hire me and change this

Model Assumptions:
1.  Some degree of separability exists between cancelled and completed rides **confirmed**
2.  Available features contain signal for prediction **confirmed**
3.  Model trained on historical data will generalize to new bookings **unverified** this project does not contain modelling phase yet

# 2. Get the Data
## 2.1 List the data

The dataset used is the Uber Ride Analytics Dashboard downloaded from Kaggle

URL: https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard

## 2.2 Storage requirements for the dataset:

Storage Requirements:
- Processed datasets: ~30-50 MB
- Model artifacts: to be defined
- Notebooks & outputs: to be defined
- Total Workspace: to be defined

Memory Requirements:
- DataFrame in memory: 111 MB
- Training with full dataset: to be defined

## 2.3 Check legal obligations

- Data License: CC BY-SA 4.0
- Personal Data: Customer/Driver IDs are anonymized
- GDPR/Privacy: No personally identifiable information (PII)

## 2.4 Access authorizations
The dataset comes from Kaggle so it's free to use via API access

## 2.5 Sensitive data analysis (PPI):
There are no sensitive names or contact information, location data or financial details from customers

# 3. EDA insights

The full content of this section can be found in the notebooks 04_univar_eda.ipynb, 05_bivar_eda.ipynb and 06_multivar_eda.ipynb

## 3.1 Size and type of data

The dataset contains 150.000 rows, 21 columns and covers the range between the 1/1-30/12 and the 29th of February is present.
The sample size is adequate for ML modeling, the 32% of the rides are cancellations so it's moderately imbalanced (same regime as many churn problems).

## 3.2 Cleaning and preprocessing steps

1. All columns got snake_cased (spaces removed, lowercased, underscores added)
2. The target column is_cancelled was remapped to binary: Completed and Incomplete = 0, everything else = 1. Note: putting `Incomplete` rides (driver no-show and similar) on the "not cancelled" side is a modelling choice, not obvious. Flagged as an assumption in 1.9
3. Quotes were stripped from booking_id and customer_id
4. All columns were reviewed for leakage and redundancy. The list of dropped columns and reason to be dropped:

| Column | Reason |
|---------|--------|
| Cancelled Rides by Driver | Information from the future |
| Cancelled Rides by Customer| Information from the future |
| Reason for cancelling by Customer | Information from the future
| Driver Cancellation Reason| Information from the future
| Incomplete Rides| Redundant information
| Incomplete Rides Reason | Information from the future
| Driver Ratings| Redundant information
| Customer Rating| Redundant information

**[After EDA]** The final dataset contains 150.000 rows ans 13 columns

## 3.3 Train/test split strategy

The data has temporal ordering (date & time) which means it can be used to analyze trends, seasonality, and patterns. The best strategy will probably be a stratified temporal split to avoid data leakage, with a 70/15/15 train/validation/test ratio.

**[After EDA]** The cancellation rate is stationary over the year, there's no drift or seasonality in the target. So a random stratified split doesn't leak anything time-dependent and is simpler than a temporal split. Volume has some mild time-of-day structure but the target doesn't, so that's fine

| Set | Percentage | Records | Purpose |
|-----|------------|---------|---------|
| Training | 70% | ~105,000 | Model training + cross-validation |
| Validation | 15% | ~22,500 | Threshold calibration + model selection |
| Test | 15% | ~22,500 | Final evaluation |

## 3.4 Conclusions of the EDA

### Temporal features
date, hour, weekday, month, is_weekend, quarter are all flat at 32% cancellation. Interesting detail though: ride volume is not flat over the day (clear morning and evening peaks) but the cancellation rate is. So the population changes over the day but behaviour doesn't.

### Identifier columns
booking_id and customer_id were dropped because they are just IDs (high cardinality, no predictive meaning) and they even have duplicates, which is a data-quality issue worth flagging but not useful for modelling

### vehicle_type
7 categories all sitting at the overall mean. Same conclusion holds when crossed against pickup/drop locations and against avg_vtat, so this variable carries no signal anywhere

### pickup_location and drop_location
176 categories each without signal individually, only noise due to their high cardinality. A next step would be to find more geographical information to cluster them

### Route interaction (pickup × drop)
Individually each location is basically noise (Cramer's V around 0.037 with the target), but the concatenation pickup + drop jumps to Cramer's V = 0.4485. That's a huge jump and suggests specific origin-destination pairs do carry signal. Caveat: with so many route combinations the value is probably inflated by high cardinality, so before using it I want to validate with target encoding + CV and compare against a model without routes

### avg_vtat
The dominant predictor with five behavioural zones from 2 minutes to 20 minutes and contains 7% of NaNs. vtat_zone has the same NaNs as avg_vtat (pd.cut leaves them alone), and vtat_missing is the flag that marks exactly those rows. Both derived features are kept because they are the strongest predictors in the whole analysis. In the real world I would conduct a further investigation about the storage of Arrival Times to understand the missing values (MNAR)

### Missingness mechanism
Used a logistic regression to see if anything other than is_cancelled predicts vtat_missing. Baseline (is_cancelled only) got AUC = 0.866, full model with all other features got AUC = 0.867. So missingness is essentially MNAR driven by the target, no hidden confounders. This means imputing avg_vtat or using vtat_missing as a feature won't pull in bias from other variables

### Deterministic rules
There are basically three "free" rules in the data that don't need a model:
- if avg_vtat ≤ 2.9 min, the ride never cancels (0% in 7.693 rows)
- if avg_vtat > 15 min, the ride always cancels (100% in 3.521 rows)
- if avg_vtat is missing, the ride always cancels (100% in 10.500 rows)

Together they cover a decent chunk of the dataset and the model only really needs to work on the messy middle

# 4. Pipeline Architecture

Two separate sklearn pipelines because logistic regression and tree-based models need different handling of avg_vtat (non-monotonic) and locations (high cardinality)

The deterministic rules from section 3.4 are the first wall. If the ride hits one of the three hard rules, skip the model, otherwise continue with two approaches

### Pipeline 1: Logistic Regression
- vtat_zone one-hot encoded
- vtat_missing used as is
- route with target encoding and out-of-fold CV
- numerical features standardised

### Pipeline 2: LightGBM or XGBoost
- avg_vtat as it is with a -1 sentinel for NaN
- route with target encoding and out-of-fold CV
- as a secondary variant, try pickup_location and drop_location passed as native categoricals to LightGBM and see if the tree finds the interaction on its own. If it does, I would save the target-encoding step at serving time

## 4.1 Modelling strategy options

Three approaches to experiment with, in order:

1. Pure ML solution:
One model sees all rides. vtat_missing is a feature alongside vtat_zone and locations. The model learns the missing-VTAT pattern on its own. Simplest to build and maintain, and definitely my starting point

2. Heuristic + ML solution:
Rides where vtat_missing = 1 are flagged as cancelled deterministically (22%) and the ML model only runs on the remaining rides (78%). The advantage is that the easy rides are easily classified, the disadvantage is that 2 systems are more complicated to maintain over time than one

## 4.2 Baseline

Before training anything fancy I want three reference points so I can tell if the ML layer is actually adding value:

1. Majority class baseline: predict "not cancelled" for everything. Recall = 0, profit = $0. Sanity floor
2. Deterministic rules only (from section 3.4): flag any ride where `avg_vtat > 15` or `vtat_missing = 1` as cancelled. That's ~14k guaranteed TPs on the full dataset (3.521 + 10.500 from section 3.4) before any model runs. Already ~$210k on the profit formula if precision stays high, which is worth verifying on the validation set
3. Logistic regression with only avg_vtat and vtat_missing: the simplest model that can exist. If LightGBM with 13 features doesn't beat this clearly, something is wrong

The ML models in 4.1 have to beat baseline 2 by a meaningful margin on F2 and expected profit. If they don't, the honest answer is to ship the rules and skip the model

## 4.3 Class imbalance

32% positive rate is moderate, same regime as churn. I don't plan to use SMOTE or random undersampling because both distort the probability calibration that the 0.125 threshold depends on. Instead:
- Logistic regression: `class_weight='balanced'`
- LightGBM / XGBoost: `scale_pos_weight = n_negatives / n_positives ≈ 2.1`

These reweight the loss without touching the sample distribution so calibrated probabilities remain meaningful

## 4.4 Calibration

The decision threshold (0.125) is derived from expected value, which assumes the predicted score is a real probability. Tree-based models are usually miscalibrated so I will:
- Fit the model on the training set
- Fit a calibrator (sigmoid first, isotonic as a second option) on the validation set
- Check a reliability diagram on the test set

If the reliability curve is already close to the diagonal after class-weighting, I skip the calibrator. Otherwise it becomes part of the serialised pipeline

## 4.5 Cross-validation strategy

`customer_id` has duplicates in the data (section 3.4), so the same customer can appear in multiple rows. A plain `StratifiedKFold` would leak customer-level signal from train into validation. The honest choice is `StratifiedGroupKFold` with `groups=customer_id`, 5 folds. Target encoding for `route` is fitted inside each fold to avoid target leakage too

## 4.6 Feature availability at inference

Flagged as unverified in 1.9 and worth revisiting here because the whole real-time story depends on it. `avg_vtat` is described in the dataset as "average time for driver to reach pickup location," which reads as a historical aggregate per area/driver rather than a post-booking measurement. If that reading is correct, it's available at booking time. If it's actually measured per-ride, the feature leaks and the model is useless in production. Before any serious training I would confirm this with whoever owns the data pipeline

---

# 5. Model Tuning

Hyperparameter search using cross-validated F2-score (β² = 4, weights recall 4x over precision) on the training set, with the `StratifiedGroupKFold` described in 4.5. The operational capacity constraint (70K interventions/year) means the deployment threshold is calibrated separately after training — sort the validation set by predicted probability, find the score at position 70K, and use that as the serving threshold. This decouples model training from threshold selection

## 5.1 Search strategy

Optuna with TPE sampler, 50 trials per model, pruning with `MedianPruner`. Random search would also work but Optuna gives me the pruning for free which matters with LightGBM where bad configs take forever. Random state is fixed per trial for reproducibility

## 5.2 Hyperparameter ranges

Logistic regression:
- `C`: log-uniform [1e-3, 1e2]
- `penalty`: {l1, l2, elasticnet}
- `l1_ratio`: [0, 1] if elasticnet
- `solver`: saga (handles all three penalties)

LightGBM:
- `num_leaves`: [15, 255]
- `max_depth`: [3, 12]
- `learning_rate`: log-uniform [0.005, 0.2]
- `min_child_samples`: [5, 200]
- `reg_alpha`, `reg_lambda`: log-uniform [1e-4, 10]
- `n_estimators`: up to 2000 with early stopping at 50 rounds on the inner validation fold

XGBoost: similar ranges, adapted to its parameter names

## 5.3 Reproducibility

- One fixed `random_state = 42` at the top of every notebook / script
- `requirements.txt` pinned to exact versions
- MLflow tracks every trial: params, metrics on each fold, confusion matrix at 0.125, threshold at the 70Kth position, git commit hash
- Dataset version (kaggle download date + md5) logged with every run so I can match a model back to the exact data it saw

## 5.4 Model selection rule

Pick the model with the highest mean F2 across folds, tie-break by expected profit on the validation set at the 70K threshold. Reject any model that fails the business constraints from 1.5 (recall ≥ 13%, precision ≥ 25%, profit ≥ $96K) even if it has the best F2. Not meeting the business constraints means the model should not ship regardless of how pretty the CV numbers are

---
# 6. Evaluation

Evaluate every trained model against the business constraints defined in section 1.5: PR-AUC, F2-score, recall, precision and expected profit. Everything here is run on the held-out test set, which is only touched once per candidate model

## 6.1 Metrics on the test set

Report at three threshold points:
- At 0.125 (the EV-derived threshold from 1.5)
- At the 70K operational threshold (from the validation set ranking)
- At 0.5 as a reference point

For each: confusion matrix, precision, recall, F2, expected profit

PR-AUC and a calibration plot go alongside so I can tell if the score is a meaningful probability or just a ranking

## 6.2 Comparison against baselines

Side-by-side table vs the three baselines in 4.2 (majority class, deterministic rules only, logreg with only 2 features). The ML model has to beat the deterministic rules on expected profit by enough to justify its maintenance cost ($50K/year from 1.5). If the gap is small, the rules ship and the model doesn't

## 6.3 Error analysis

The bulk of the insight usually comes from looking at *where* the model is wrong, not at aggregate numbers. I would slice errors by:
- Vehicle type (7 categories)
- Hour of day and day of week
- Pickup / drop location top-20 (where most errors concentrate)
- VTAT zone (including the messy middle between 2.9 and 15 minutes, which is where the model actually has to work)

For each slice: recall, precision and false negative rate. If one slice is systematically worse, either it needs a dedicated feature or a product-side call (e.g. "don't intervene on these rides at all")

## 6.4 Subgroup performance

Even without demographic attributes, pickup location clusters can proxy for neighbourhoods. I would check whether any cluster gets flagged materially more often than the overall rate, and whether precision is stable across clusters. Not a full fairness audit but the honest minimum when the intervention affects service quality

## 6.5 Stability of the profit estimate

Expected profit is a single number that stakeholders will quote. It deserves an uncertainty range. Bootstrap the test set 1000 times, recompute TP, FP and profit each time, report the 95% interval. Something like `$96k profit` becomes `$96k [89k, 103k]` which is much more honest

## 6.6 Sanity checks before declaring the model done

- Reliability diagram close to the diagonal after calibration
- No single feature dominates gain importance by more than 80% (would be a fragility signal)
- Removing `route` and retraining: how much profit is lost? Tells me how dependent the model is on that high-cardinality feature
- Performance on the last month of the test set vs the first month: even though the target is stationary in 2024, this catches any late drift I missed

---
# 7. Deployment

Hypothetical deployment plan since there's no real Uber integration on the other end. The model scores rides in real time at booking time (I ensured inference < 100ms when validating assumptions). When the predicted probability exceeds the deployment threshold, the ride is flagged for intervention

## 7.1 Serving architecture

- Single serialised artifact containing preprocessing + calibrator + model, loaded once per service instance. This is the main guarantee that training and serving transformations are identical
- REST endpoint behind the booking service, one request per booking, returns `{probability, flag, model_version}`
- Feature lookup for aggregated features (like `avg_vtat` per area) from a feature store or a cached table refreshed daily. These are the features that would bite the hardest if computed inconsistently between training and serving
- Fallback: if the model service times out or returns an error, fall back to the deterministic rules from 3.4. Never block a booking on a model failure

## 7.2 Feature availability at inference

Revisit of the concern raised in 4.6. `avg_vtat`, `pickup_location`, `drop_location`, `vehicle_type`, `time`, `date` all have to be resolvable at the moment the booking request hits the service, before the driver is assigned. If any of them aren't, the model is retrained without that feature or the project doesn't ship. This gets validated with the data pipeline owner before any real training run

## 7.3 Rollout strategy

- Shadow mode first: the model scores every ride but no interventions are triggered. Compare predicted flags against actual outcomes for 2-4 weeks to confirm production performance matches the test set
- Canary: route 5% of traffic through the active model (interventions triggered), the rest keeps running without interventions. Monitor daily
- Full rollout only after canary hits the business constraints from 1.5 on live data
- Blue/green for the model service itself so rollback is instant

## 7.4 Rollback criteria

Automatic rollback triggers:
- Weekly precision < 20% for 2 consecutive weeks (below the 25% minimum from 1.5 with a small margin)
- Weekly recall < 10% for 2 consecutive weeks
- Expected profit negative for any single week
- Prediction latency p95 > 100ms
- Error rate on the model endpoint > 1%

Manual rollback for anything else that looks wrong in monitoring. Better to revert to rules-only than serve a broken model

## 7.5 Causal uplift — open assumption

The economics in 1.5 assume that an intervention on a high-risk ride actually reduces the cancellation probability. That's a product assumption, not something this model can prove. Treating it as unverified and out of scope for the current project. The honest way to measure it later would be an A/B test where flagged rides are randomly split into "intervene" vs "hold out," comparing realised cancellation rates. Without that, the $15 TP benefit is an estimate, not a measurement

---
# 8. Monitoring

Cancellations are labelled within minutes to hours of the booking, so I can monitor true performance on live data without long delays. That's a luxury not every ML problem has and it shapes the whole monitoring plan

## 8.1 Health of the data

Track the `vtat_missing` rate daily because it's where the heuristic part of the hybrid solution could break. Any sudden shift from the 7% observed initially can invalidate the best feature. Track class distribution shifts because that means the population has changed and may trigger retraining

Alerting thresholds:
- `vtat_missing` rate outside [4%, 12%] for 2 consecutive days → warn
- `vtat_missing` rate outside [2%, 15%] for any single day → alert
- Daily cancellation rate outside [25%, 40%] → alert

## 8.2 Model performance

Log predicted probability and actual outcome for every scored ride. Compute PR-AUC, F2-score, precision and recall on a rolling weekly window and compare against training baselines. Check that neither recall nor precision drop under thresholds defined in section 1.5

Alerting thresholds (2 consecutive weeks unless noted):
- Weekly F2 below 90% of validation F2 → warn
- Weekly precision below 25% → alert (violates business constraint)
- Weekly recall below 13% → alert (violates business constraint)
- Weekly PR-AUC drop greater than 0.05 vs training baseline → warn

## 8.3 Business impact

Most meaningful signal for stakeholders. Track:
- Weekly interventions vs the 70K/year capacity (≈1.350/week). Overshoot means the threshold needs retuning
- Monthly expected profit, with bootstrap CI from 6.5 updated monthly
- Quarterly cumulative profit vs the $96K/year target

## 8.4 Concept drift

Something that surprised me is that the 2024 data was stationary, so drift is the thing most likely to break this model first. Track:
- Cramér's V between cancellation rate and temporal features (weekday, hour) weekly. Baseline is near zero, any jump above 0.05 means the stationarity assumption is breaking
- PSI on `vtat_zone` weekly: PSI < 0.1 fine, 0.1–0.25 warn, > 0.25 alert (industry convention)
- PSI on top-20 pickup and top-20 drop locations weekly, same thresholds
- `vtat_missing` is already covered by 8.1

## 8.5 Retraining trigger

Section 1.4 mentions daily retraining as a default. That's fine as a starting posture but it should be revisited once the pipeline is live. Concretely:
- Scheduled daily retraining as the baseline
- Performance-triggered retraining if any PSI goes critical or any business constraint is violated for 2 consecutive weeks, whichever comes first
- Every retrain runs through the same evaluation gate as the initial model (section 6.2 baselines, business constraints from 1.5). A retrained model that fails the gate doesn't get promoted, the previous version keeps running
