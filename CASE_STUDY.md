# Case Study

This documents explains the step-by-step logic & workflow following CRISP-DM adapted

---

## Index
- [Case Study](#case-study)
  - [Index](#index)
- [Problem Framing](#problem-framing)
  - [1.2 How will the solution be used?](#12-how-will-the-solution-be-used)
  - [1.4 How should the problem be framed](#14-how-should-the-problem-be-framed)
  - [1.5 How should performance be measured?](#15-how-should-performance-be-measured)
    - [Primary objective](#primary-objective)
    - [Per-prediction expected value](#per-prediction-expected-value)
    - [Operational capacity constrain \& its strategy](#operational-capacity-constrain--its-strategy)
    - [Business constraints to accept (or reject) the trained model](#business-constraints-to-accept-or-reject-the-trained-model)
    - [Final metrics:](#final-metrics)
  - [1.8 How would I solve the problem manually?](#18-how-would-i-solve-the-problem-manually)
  - [1.9 List the assumptions made so far. Verify assumptions if possible](#19-list-the-assumptions-made-so-far-verify-assumptions-if-possible)
- [2. Get the Data](#2-get-the-data)
  - [2.2 Storage requirements for the dataset:](#22-storage-requirements-for-the-dataset)
  - [2.5 Sensitive data analysis (PPI):](#25-sensitive-data-analysis-ppi)
- [3. EDA insights](#3-eda-insights)
  - [3.2 Cleaning and preprocessing steps](#32-cleaning-and-preprocessing-steps)
  - [3.3 Train/test split strategy](#33-traintest-split-strategy)
  - [3.4 Conclusions of the EDA](#34-conclusions-of-the-eda)
    - [Temporal features](#temporal-features)
    - [Identifier columns](#identifier-columns)
    - [vehicle\_type](#vehicle_type)
    - [pickup\_location and drop\_location](#pickup_location-and-drop_location)
    - [route interaction (pickup × drop)](#route-interaction-pickup--drop)
    - [avg\_vtat](#avg_vtat)
    - [Missingness mechanism on avg\_vtat](#missingness-mechanism-on-avg_vtat)
    - [Deterministic rules](#deterministic-rules)
- [4. Modelling](#4-modelling)
  - [4.1 Class imbalance](#41-class-imbalance)
  - [4.2 Baseline](#42-baseline)
  - [4.1 Modelling strategy options](#41-modelling-strategy-options)
    - [1. Pure ML solution](#1-pure-ml-solution)
      - [Pipeline 1: Logistic Regression](#pipeline-1-logistic-regression)
      - [Pipeline 2: LightGBM or XGBoost](#pipeline-2-lightgbm-or-xgboost)
    - [2. Heuristic + pure ML solution](#2-heuristic--pure-ml-solution)
  - [4.4 Calibration](#44-calibration)
  - [4.5 Cross-validation strategy](#45-cross-validation-strategy)
- [5. Model Tuning](#5-model-tuning)
- [6. Evaluation](#6-evaluation)
- [7. Deployment](#7-deployment)
- [8. Monitoring](#8-monitoring)
---

# Problem Framing
Uber offers a booking service in an Indian metropolitan area and provided me with data from all the bookings of 2024. I identified that the dataset contains 170,000 bookings of which the 32% never reached completion status, that includes different cancellation type based on reason of cancellation or lack of available driver

**[After EDA]** The dataset actually contains 150.000 rows with 32% cancellations (1/3 of rides)

In terms of business impact, I can think of the following areas:

1. Financial impact:
   1. Revenue loss
   2. Hidden operational costs like the time and fuel wasted by drivers while the ride is not cancelled yet, the costs from the processing platform and the potential churn of frustrated customers
   3. Opportunity cost from resources within the company like customer support, refunds, managing complains, etc

2. Operational impact:
   1. Imbalance in supply & demand. While drivers are booked and removed from the pool, this can artificially increase the price of the available services and potentially damaging the image of the company
   2. Driver insatisfaction if the customer is cancelling their rides too often

Let's imagine that Uber wants to reduce the cancellation rate by 10%, this means 3.700 more rides completed and almost 75000$ recovered so building a predictive model that identifies bookings with high cancellation risk at the time of booking could help achieving it

## 1.2 How will the solution be used?

The model will be deployed as a real-time prediction system integrated into Uber's booking workflow 

This will help other departments of the company to develop and implement long-term fixes like:
- Increase customer engagement on high-risk rides by sending booking confirmation messages, provide more frequent ETA updates, or offering loyalty points
- Optimization of the algorithm for driver allocation, if driver rating is a strong indicator we can redistribute or prioritize better drivers more often

**[After EDA]** Waiting time is a strong cancellation indicator so a good idea would be to redistribute drivers to keep a low waiting time in every area

## 1.4 How should the problem be framed

I'm going to start framing it as a supervised binary classification problem: cancelled vs. completed. In future approaches it would be interesting to try a multi-class approach using columns that show the reason for cancelling

The training will be offline on historical batch data, re-trained daily and monitor concept drift to check if it requires faster adaptation and we have to change to near-online or online training

The inference will be in real time

**[After EDA]** The distribution of cancellation rate and total rides over the year did not show clear patterns so I would strongly to advocate for offline training + daily re-train as the first option

## 1.5 How should performance be measured?

After having a chat with Product and Ops teams I define a cost matrix:

| Outcome | What happens | Cost/Benefit | Business Meaning |
|------------|----------------|----------------|------------------|
| True Positive (TP) | Ride is cancelled & system intervenes | +15$ | Prevented cancellation saves 20$ revenue, minus 5$ intervention cost |
| False Positive (FP) | Ride is NOT cancelled & system intervenes | -5$ | Unnecessary intervention cost |
| True Negative (TN) | Ride is NOT cancelled & system does NOT intervene | 0$ | Ride completes normally, no model contribution |
| False Negative (FN) | Ride is cancelled & system does NOT intervene | -20$ | Lost booking revenue + driver idle time + customer dissatisfaction |

A missing a cancellation is 4x more costly than a false alarm!

### Primary objective

What stakeholders care about is to maximize the annual aggregate profit:

```
Annual profit = TP x 15$ - FP x 5$
```

It's important to remember that this formula only evaluates what the model did, not what it failed to do so FNs are not counted here

### Per-prediction expected value

In every single ride with a predicted cancellation probability P. I compare the expected value of both actions:

```
EV(intervene) = P x (+15$) + (1 - P) x (-5$) = 20P - 5
EV(don't intervene) = P x (-20$) + (1 - P) x (0$) = -20P

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

1. System viability (model building + maintenance cost: 50K$/year)

```
Required TP x 15$ >= 50.000 -> TP >= 3.334
Recall >= 3.334 / 48.000 = 7%
```

2. Target ROI (defined above 10% cancellation reduction)

```
0.1 x 48.000 = 4.800 rides to prevent
4.800 x 20$ revenue = 96.000$ target

Required TP x 15$ >= 96.000 TP >= 6.400
Recall >= 6.400 / 48.000 = 13%
```

3. Min precision
```
P x 15$ - (1 - P) x 5$ >= 0 -> 20P >= 5
Precision >= 25%
```

4. Final Expected profit
```
Net profit = TP x 15$ - FP x 5$ >= 96.000$
```

### Final metrics:

Based on the analysis above, the full set of (business and technical) metrics is:

1. Decision threshold = 12.5%
2. Recall >= 13%
3. Precision >= 25%
4. F2-score to model ranking - > β² = FN/FP = 20$/5$ = 4 weights recall 4x over precision
5. PR-AUC to compare models because I have class imbalance
6. Expected profit = TP x 15$ - FP x 5$ >= 96.000$


## 1.8 How would I solve the problem manually?

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

**[After EDA]** Most of these heuristic rules could not be validated but there are three deterministic patterns (see EDA conclusions below). My approach would be to flag these three cases deterministically and apply the ML model on the remaining rides

## 1.9 List the assumptions made so far. Verify assumptions if possible

**[After EDA]** Assumptions marked as **confirmed**, **refuted** & **unverified** (could not be tested)

Data Assumptions:
1. The 2024 dataset is representative of typical booking patterns **refuted** 31st of Dec is missing and it's a leap year
2. Cancellation reasons are accurately recorded **unverified** dataset is from kaggle
3. All relevant data is captured in the dataset **refuted** columns and rows don't match the document
4. Patterns in 2024 will persist into future periods **unverified** dataset is from kaggle
5. Features available at booking time don't include post-booking information **refuted** and dropped
6. Class distribution reflects realistic cancellation rate **unverified** dataset is from kaggle
7. Target variable is correctly defined and labelled **refuted** more labels than needed, collapsed into binary


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

The dataset used is the Uber Ride Analytics Dashboard downloaded from Kaggle

URL: https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard

- Data License: CC BY-SA 4.0
- Personal Data: Customer/Driver IDs are anonymized
- GDPR/Privacy: No personally identifiable information (PII)


## 2.2 Storage requirements for the dataset:

Storage Requirements 
- Processed datasets: ~30-50 MB
- Model artifacts: to be defined
- Notebooks & outputs: to be defined
- Total Workspace: to be defined

Memory Requirements:
- DataFrame in memory: 111 MB
- Training with full dataset: to be defined

## 2.5 Sensitive data analysis (PPI):
There are no sensitive names or contact information, location data or financial details from customers

# 3. EDA insights

The full content of this section can be found in the notebooks:
```
04_univar_eda.ipynb
05_bivar_eda.ipynb
06_multivar_eda.ipynb
```

## 3.2 Cleaning and preprocessing steps

The raw dataset was cleans following the next steps: snake_case cols, target mapping to binary, stray quotes stripped from ID cols, and recast of data types. No duplicated rows were found

The more interesting part of this step was the review for leakage and redundancy. The following columns were dropped:

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

The final dataset contains 150.000 rows and 13 columns

## 3.3 Train/test split strategy

The data has date and time, so a stratified temporal split with 70/15/15

**[After EDA]** cancellation rate is flat across the year, no drift or seasonality in the target. A random stratified split leaks nothing time-dependent and is simpler so that's what I'll use


## 3.4 Conclusions of the EDA

### Temporal features
Date, hour, weekday, month, is_weekend and quarter are all flat at the overall cancellation rate. Ride volume has clear morning and evening peaks but the cancellation rate doesn't move with them

### Identifier columns
booking_id and customer_id were dropped. Also they have duplicates which is a data-quality issue worth flagging 

### vehicle_type
All 7 categories sit at the overall mean, including when crossed against locations and against avg_vtat. No signal anywhere

### pickup_location and drop_location
176 categories each, only noise individually because of the high cardinality. A next step would be to bring in geographical information to cluster them somehow

### route interaction (pickup × drop)
It shows a clear association with the target but it's probably a signal inflated by high cardinality. I left the analysis with target encoding + CV for future iterations

### avg_vtat
The dominant predictor, with behavioural zones from 2 to 20 minutes and around 7% NaNs. Derived two features: vtat_zone (binned) keeps the NaNs as-is and vtat_missing (bool) flags exactly those rows

### Missingness mechanism on avg_vtat
Tested whether anything other than is_cancelled predicts vtat_missing and missingness is essentially MNAR driven by the target, with no hidden confounders. Imputing missing vals won't create bias :)

### Deterministic rules
There are three rules:
- if avg_vtat <= 2.9 min, the ride never cancels (0% in 7.693 rows)
- if avg_vtat > 15 min, the ride always cancels (100% in 3.521 rows)
- if avg_vtat is missing, the ride always cancels (100% in 10.500 rows)


# 4. Modelling

## 4.1 Class imbalance

32% positive rate is moderate and I don't plan to resample the data, instead I'll change the class weights in LogR and the scale_pos_weight in trees

## 4.2 Baseline

These are the 2 references to check:

1. Majority class is 68%
2. Deterministic rules only, already cover approx 2.1k rides in the test set at near 100% precision

My decision is to fit a logistic regression only with avg_vtat

## 4.1 Modelling strategy options

### 1. Pure ML solution
Two pipeline variants:

#### Pipeline 1: Logistic Regression
- vtat_zone one-hot encoded
- vtat_missing used as is
- route with target encoding and out-of-fold CV
- numerical features standardised

#### Pipeline 2: LightGBM or XGBoost
- avg_vtat as it is with a -1 sentinel for NaN
- route with target encoding and out-of-fold CV
- as a secondary variant, try pickup_location and drop_location passed as native categoricals to LightGBM and see if the tree finds the interaction 
  
### 2. Heuristic + pure ML solution

Good bc easy rides are classified, but it needs maintenance of two systems 

## 4.4 Calibration

Plan:
- Use CalibratedClassifierCV 
- Check the reliability diagram
- Verify the final reliability diagram


## 4.5 Cross-validation strategy

Since most customers appear once and only a few have 2-3 rides I commit leakage so I would use StratifiedGroupKFold with groups=customer_id. Either way, 5 folds. Also the target encoding for route is fitted inside each fold to avoid target leakage 

# 5. Model Tuning

Hyperparameter search using cross-validated F2-score and Optuna

Things to take into account:

- Reuse the StratifiedGroupKFold splits from 4.5 so tuning sees the same CV as training
- Optimise F2 on the out-of-fold predictions
- Tune the decision threshold that maximises expected profit 

# 6. Evaluation

Besides scoring on the metrics in section 1.5, I have to:

- Slice errors and flag those worse than the overall model
- Check proxy groups and confirm precision stays > 25% in each
- Sanity checks before shipping: reliability diagram on the diagonal, no single feature dominating importance, profit drop if route is removed


# 7. Deployment

The model scores in real time and flags it when the probability passes the threshold

The steps I would take would be:

1.  Ship one artifact with preprocessing, calibrator and model & serve it as a REST endpoint under 100ms
- Confirm that all model features are available at booking time
- Roll out in shadow mode, then 5% canary, then full all with blue/green for instant rollback
- Set auto-rollback rules based on business, technical and infrastructure metrics 
- Set A/B tests

# 8. Monitoring

Labels come in within minutes to hours, so live performance can be tracked without delay

Things to look at:

- Log probability and outcome for every scored ride
- Track data health and drift of present features daily
- Track weekly metrics and set threshold for recall and precision
- Retrain on drift or business constraint thresholds 
