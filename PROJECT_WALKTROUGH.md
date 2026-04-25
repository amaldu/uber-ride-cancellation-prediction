# Project Walkthrough

This documents explains the step-by-step logic and workflow following CRISP-DM adapted. The goal of the project is to predict Uber ride cancellations using machine learning to enable proactive intervention by creating strategies to reduce the overall cancellation rate.

---

## Index
- [Project Walkthrough](#project-walkthrough)
  - [Index](#index)
- [Problem Framing](#problem-framing)
  - [1.1 Defininition of the objective in business terms](#11-defininition-of-the-objective-in-business-terms)
  - [1.2 How will the solution be used?](#12-how-will-the-solution-be-used)
  - [1.3 What are the current solutions/workarounds (if any)?](#13-what-are-the-current-solutionsworkarounds-if-any)
  - [1.4 How should the problem be framed](#14-how-should-the-problem-be-framed)
  - [1.5 How should performance be measured?](#15-how-should-performance-be-measured)
    - [Recall: deriving the target from business constraints](#recall-deriving-the-target-from-business-constraints)
    - [Precision: sanity check](#precision-sanity-check)
    - [Validation → Precision = 50% and Recall = 40%:](#validation--precision--50-and-recall--40)
    - [Final metrics:](#final-metrics)
  - [1.6 Is the performance measure aligned with the business objective?](#16-is-the-performance-measure-aligned-with-the-business-objective)
  - [1.7 What would be the minimum performance needed to reach the business objective?](#17-what-would-be-the-minimum-performance-needed-to-reach-the-business-objective)
  - [1.8 What are comparable problems? Can you reuse experience or tools?](#18-what-are-comparable-problems-can-you-reuse-experience-or-tools)
  - [1.9 Is human expertise available?](#19-is-human-expertise-available)
  - [1.10 How would you solve the problem manually?](#110-how-would-you-solve-the-problem-manually)
  - [1.11 List the assumptions made so far](#111-list-the-assumptions-made-so-far)
  - [1.12 Verify assumptions if possible](#112-verify-assumptions-if-possible)
- [2. Get the Data](#2-get-the-data)
  - [2.1 List the data](#21-list-the-data)
  - [2.2 Origin of the dataset](#22-origin-of-the-dataset)
  - [2.3 Storage requirements for the dataset:](#23-storage-requirements-for-the-dataset)
  - [2.4 Check legal obligations](#24-check-legal-obligations)
  - [2.5 Access authorizations](#25-access-authorizations)
  - [2.6 Data ingestion and overview](#26-data-ingestion-and-overview)
  - [2.7 Format of the data](#27-format-of-the-data)
  - [2.8 Sensitive data analysis (PPI):](#28-sensitive-data-analysis-ppi)
- [3. EDA insights](#3-eda-insights)
  - [3.1 Size and type of data](#31-size-and-type-of-data)
  - [3.2 Cleaning and preprocessing steps](#32-cleaning-and-preprocessing-steps)
    - [3.2.1 Data Leakage Analysis](#321-data-leakage-analysis)
      - [Resulting dataset with data types optimized](#resulting-dataset-with-data-types-optimized)
  - [3.3 Train/test split strategy](#33-traintest-split-strategy)
  - [Basic Analysis](#basic-analysis)
  - [Univariate Analysis](#univariate-analysis)
    - [TODO - Is there a relationship between customers booking more than once and cancelling? Same vehicle/location each time? Are repeat-bookers more likely to cancel?](#todo---is-there-a-relationship-between-customers-booking-more-than-once-and-cancelling-same-vehiclelocation-each-time-are-repeat-bookers-more-likely-to-cancel)
    - [TODO - No description of vehicle types in the dataset. Understanding their characteristics (size, price tier) could reveal patterns.](#todo---no-description-of-vehicle-types-in-the-dataset-understanding-their-characteristics-size-price-tier-could-reveal-patterns)
    - [TODO - Find geographical info to compute actual distances and check if pickup == drop.](#todo---find-geographical-info-to-compute-actual-distances-and-check-if-pickup--drop)
  - [Bivariate Analysis (Feature vs Target)](#bivariate-analysis-feature-vs-target)
    - [Engineered features](#engineered-features)
- [4. Multivariate EDA](#4-multivariate-eda)
  - [Analyses Performed](#analyses-performed)
  - [Key Findings](#key-findings)
    - [Missingness is not confounded](#missingness-is-not-confounded)
    - [Feature redundancy resolved](#feature-redundancy-resolved)
    - [Correlation structure](#correlation-structure)
  - [Final Feature Set for Modeling](#final-feature-set-for-modeling)
  - [Encoding Recommendations](#encoding-recommendations)
  - [Realistic Expectations](#realistic-expectations)
  - [Deferred to Modeling Phase](#deferred-to-modeling-phase)
- [5. Feature Engineering](#5-feature-engineering)
  - [Pipeline Architecture](#pipeline-architecture)
    - [Logistic Regression Pipeline](#logistic-regression-pipeline)
    - [Tree Models Pipeline](#tree-models-pipeline)
  - [Key Design Decisions](#key-design-decisions)
  - [Artifacts Saved](#artifacts-saved)

---

# Problem Framing
## 1.1 Defininition of the objective in business terms

Uber offers a booking service in an Indian metropolitan area and provided me with data from all the bookings of 2024. I identified that the dataset contains 150,000 bookings of which the 32% never reached completion status, that includes different cancellation type based on reason of cancellation or lack of driver available. 

This means almos 1 cancellation for every 3 rides.  

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


## 1.3 What are the current solutions/workarounds (if any)?

There are none

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

The cost asymmetry is **3:1** (net TP benefit $15 vs FP cost $5), so catching one cancellation is worth tolerating up to 3 false alarms. This makes **recall the primary metric** — the goal is to catch as many cancellations as possible while keeping false alarms manageable.

### Recall: deriving the target from business constraints

**Step 1 — Lower bound from financial targets**

Working assumption: precision = 50% (for every cancellation caught, we generate one false alarm). Net savings per TP caught:

```
Net savings per TP = $15 (TP benefit) - $5 (one FP generated at 50% precision) = $10 per TP
```

The financial team sets two targets:

Minimum viability ($50K/year):
```
TP × 10 ≥ 50.000 → TP ≥ 5.000 → Recall ≥ 5.000 / 48.000 = 10%
```

Target ROI ($100K/year):
```
TP × 10 ≥ 100.000 → TP ≥ 10.000 → Recall ≥ 10.000 / 48.000 = 21%
```

**Step 2 — Upper bound from operational capacity**

The system can handle 50.000 interventions/year. At 50% precision (FP = TP), total interventions = 2 × TP:

```
2 × TP ≤ 50.000 → TP ≤ 25.000 → Recall ≤ 25.000 / 48.000 = 52%
```

**Step 3 — Choose recall target**

| Constraint | Recall bound | Reason |
|------------|-------------|--------|
| Minimum viability | ≥ 10% | Below this, savings don't cover costs |
| Target ROI | ≥ 21% | Minimum to hit the $100K business target |
| Operational capacity | ≤ 52% | Max catchable cancellations within 50K intervention budget |

Valid window: **21% – 52%**. Choosing **Recall = 40%** as a conservative starting target with headroom on both sides.

### Precision: sanity check

With Recall = 40% and the 50% precision working assumption:

```
TP = 0.4 × 48.000 = 19.200
FP = 19.200 (at 50% precision)
Total interventions = 38.400 → within 50K operational cap ✓
```

Break-even precision — the minimum precision where every positive prediction has positive expected value:

```
Precision × $15 - (1 - Precision) × $5 ≥ 0
15P - 5 + 5P ≥ 0 → 20P ≥ 5 → P ≥ 0.25
```

50% precision > 25% break-even ✓

### Validation → Precision = 50% and Recall = 40%:

```
TP = 0.4 × 48.000 = 19.200
FP = 19.200 (at 50% precision)
Total interactions = 38.400 → within operational capacity ✓

&

Net savings = 19.200 × 15 - 19.200 × 5 = 288.000 - 96.000 = 192.000 → above target ROI ✓

&

Precision (50%) > break-even precision (25%) ✓
```

### Final metrics:

Based on the analysis above I decided to choose metrics that:

1. Reflect the cost asymmetry of 3:1
2. Work for moderated class imbalance


| Metric | Value | Reasoning |
|------------|---------------------|------------------|
| Recall | ≥ 40% | Derived from the $100K ROI target and operational capacity constraints |
| Precision | ≥ 50% | Working assumption used throughout the analysis; above 25% break-even |
| F2-score | ≥ 0.42 | β² = $15/$5 = 3 → β ≈ 1.73, rounded to F2 = 5×P×R / (4P + R); value at P=50%, R=40% |
| PR-AUC | The best possible | Model comparison metric for imbalanced datasets |
| Expected profit | TP×$15 - FP×$5 ≥ $100K | Final sanity check: validates the model delivers actual business value |

## 1.6 Is the performance measure aligned with the business objective?

Yes, the selected performance measures are aligned with the business objective and cost structure of the cancellation prediction problem.

Given the asymmetric cost of errors, where recall is weighted four times more heavily than precision, I need to use of the F2-score as the primary optimization metric so I can reflect the higher business impact of missed cancellations.

Recall is constrained to be at least 70% to ensure that the majority of cancellation events are detected and precision is constrained to be at least 60% to limit unnecessary intervention and extra operational costs.

Since the dataset is imbalanced, the Precision–Recall (PR) curve is used for model comparison.

Expected profit is used as a post-selection validation metric to make sure that these metrics have a translation into business value.

## 1.7 What would be the minimum performance needed to reach the business objective?

| Metric | Minimum Threshold | Rationale |
|--------|-------------------|-----------|
| F2-Score | ≥ 0.55 | Calculated from minimum P=50%, R=60% |
| Recall | 60% | From ROI analysis |
| Precision | 50% | From break-even analysis |
| F1-Score | 0.55 | Baseline for balanced performance |
| AUC-ROC | 0.32 | Extracted from ositive class proportion |


## 1.8 What are comparable problems? Can you reuse experience or tools?

No because it's the first project in this company :)

## 1.9 Is human expertise available?

There should be but this is just a ML checklist so I'm going to guess a lot: 

The dataset provides domain knowledge, including documented cancellation reasons (wrong address, driver-related issues), vehicle-type performance metrics, and observable patterns related to payment methods.

From the company I would probably be able to obtain general industry knowledge of ride-sharing operations like common cancellation triggers in transportation services, established operational best practices, and typical customer behavior patterns in on-demand mobility platforms, etc.

Data science expertise is also available in the topics of binary classification modeling, imbalanced datasets, engineering temporal and behavioral features, etc. 

However, certain knowledge gaps that come to my mind would need to be addressed like Uber operational constraints, regional factors affecting rides, clear definitions of peak hours and the impact of surge pricing...

## 1.10 How would you solve the problem manually?

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


## 1.11 List the assumptions made so far

Data Assumptions:
1. The 2024 dataset is representative of typical booking patterns
2. Cancellation reasons are accurately recorded
3. All relevant features are captured in the dataset
4. Patterns in 2024 will persist into future periods
5. Features available at booking time don't include post-booking information

Business Assumptions:
1. Proactive measures can actually prevent cancellations
2. Intervention costs are lower than cancellation costs
3. Customers will respond positively to interventions

Technical Assumptions:
1.  All features used in training will be available at inference time
2.  Model can score bookings within acceptable time (<100ms)
3.  Deployment infrastructure exists or can be built

Model Assumptions:
1.  Some degree of separability exists between cancelled and completed rides
2.  Available features contain signal for prediction
3.  Model trained on historical data will generalize to new bookings

## 1.12 Verify assumptions if possible

At this stage, I assume that the dataset contains all relevant features required to model the target variable, and that no critical predictive information is systematically missing.

Since this is the first version of the project those will be the project I will only verify the following assumptions in the `02_assumptions.ipynb` notebook:

| Assumption | Verification Method | Status | Action taken |
|------------|---------------------|--------|--------------|
| Data completeness | Number of columns and rows match the document | Amount of columns and rows is different | Updated document |
| Time coverage | The dataset represents data from the whole year 2024 | 31st of December is missing | Contact Ops team to obtain the data |
| Class balance | Verify that the imbalance is 25% | Verified: based on the criteria used to label cancellations, the final value is 32% cancellation rate | None |
| Valid target | Target column exists and it's in the format that the business needs to predict | More labels than needed | Define what is cancelled and classify the labels into 2 categories |
| No obvious data leakage | Post-cancellation columns are removed | Verified | 7 columns have to be removed by leakage or redundance |

These assumptions require business input and post-deplopyment monitorization that can be verified after the model is deployed: 

| Assumption | Verification method |
|------------|---------------------|
| Intervention effectiveness | A/B test results or pilot data |
| Cost structure | Finance or Ops team input |
| Operational capacity | Operations team assessment |
| Pattern stationarity | Model performance drift detection |
| Feature availability | Data pipeline monitoring |
| Customer response | Intervention success rate tracking |


# 2. Get the Data
## 2.1 List the data 

**Characteristics of the available Data**:
- **150.000 booking records** from 2024
- **21 columns** covering all required feature categories (according to what is assumed in the previous section)
- **Full year coverage** with daily granularity

**Data Sufficiency Assessment**:
- The sample size is adequate for ML modeling 
- The positive class (cancellations) has sufficient examples (32% of that class)
- Temporal coverage spans full year with time-stamp granularity

## 2.2 Origin of the dataset

The dataset used is the **Uber Ride Analytics Dashboard** downloaded from Kaggle

URL: https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard 

There are two other files:

- `Uber.pbix` - Power BI dashboard file
- `Dashboard.gif` - Visualization preview

## 2.3 Storage requirements for the dataset:

**Storage Requirements**:

- Processed datasets: ~30-50 MB 
- Model artifacts: to be defined
- Notebooks & outputs: to be defined
- **Total Workspace**: to be defined

**Memory Requirements**:
- DataFrame in memory: 111 MB 
- Training with full dataset: to be defined


## 2.4 Check legal obligations

**Legal Assessment**:
- Data License: `CC BY-SA 4.0`
- Personal Data: Customer/Driver IDs are anonymized
- Commercial Use: It is allowed to:
    - Share — copy and redistribute the material in any medium or format for any purpose, even commercially.
    - Adapt — remix, transform, and build upon the material for any purpose, even commercially. 
- GDPR/Privacy: No personally identifiable information (PII)
- About the creator: [Yash Dev Laddha](https://www.kaggle.com/yashdevladdha)


**Data Privacy Compliance**:
- No real names or contact information
- Location data is categorical 
- Booking IDs are synthetic identifiers
- No sensitive financial details (only payment method type)

## 2.5 Access authorizations

**Access Requirements**:

1. **Kaggle Account**: Required for API access
2. **API Credentials**: `kaggle.json` file with username and key
3. **Environment Setup**: 
   ```bash
   export KAGGLE_USERNAME="your_username"
   export KAGGLE_KEY="your_api_key"
   ```

## 2.6 Data ingestion and overview

The data has been downloaded using the Kaggle API as documented in `notebooks/03_eda.ipynb`
:

- File location: `uber-analysis/data/raw/ncr_ride_bookings.csv`
- Download date: As per notebook execution
- Ingestion: CSV file loads successfully with pandas

## 2.7 Format of the data

| Stage | Format | Purpose |
|-------|--------|---------|
| Raw | CSV | Original data preservation |
| Working | Pandas DataFrame | Analysis and transformation |
| Processed | Parquet | Efficient storage with types preserved |

## 2.8 Sensitive data analysis (PPI):

The following columns are suspected to contain PII. After performing a quick overview in 03_initial_inspection.ipynb, I can summarize their characteristics and the actions to take:

| Column | Status | Action |
|--------|-------------|--------|
| Customer ID | anonymized | Keep as-is |
| Booking ID | synthetic codes | Keep as-is |
| Pickup Location | area names | Keep as-is |
| Drop Location | area names | Keep as-is |
| Payment Method | type only | Keep as-is |


# 3. EDA insights

## 3.1 Size and type of data

| Attribute | Value |
|-----------|-------|
| Total Records | 150.000 |
| Total Features | 21 columns |
| Data Type | Tabular (structured) |
| Temporal Scope | Year 2024 |
| Geographic Scope | NCR (National Capital Region) |
| Update Frequency | Static dataset |

```
 #   Column                             Non-Null Count   Dtype  
---  ------                             --------------   -----  
 0   Date                               150000 non-null  object 
 1   Time                               150000 non-null  object 
 2   Booking ID                         150000 non-null  object 
 3   Booking Status                     150000 non-null  object 
 4   Customer ID                        150000 non-null  object 
 5   Vehicle Type                       150000 non-null  object 
 6   Pickup Location                    150000 non-null  object 
 7   Drop Location                      150000 non-null  object 
 8   Avg VTAT                           139500 non-null  float64
 9   Avg CTAT                           102000 non-null  float64
 10  Cancelled Rides by Customer        10500 non-null   float64
 11  Reason for cancelling by Customer  10500 non-null   object 
 12  Cancelled Rides by Driver          27000 non-null   float64
 13  Driver Cancellation Reason         27000 non-null   object 
 14  Incomplete Rides                   9000 non-null    float64
 15  Incomplete Rides Reason            9000 non-null    object 
 16  Booking Value                      102000 non-null  float64
 17  Ride Distance                      102000 non-null  float64
 18  Driver Ratings                     93000 non-null   float64
 19  Customer Rating                    93000 non-null   float64
 20  Payment Method                     102000 non-null  object 
```
## 3.2 Cleaning and preprocessing steps 

| Transformation | Columns | Actions | 
|----------------|---------|---------|
| snake case | all dataset | remove spaces, lowercase and add underscores \
| map target | "is_cancelled" | "Completed": 0, "Cancelled by Driver": 1, "No Driver Found": 1, "Cancelled by Customer": 1,"Incomplete": 0 |
| format Identifiers | "booking_id", "customer_id" | remove quotes |
| data leakage analysis | all columns | remove potential leaking and redundant columns 

### 3.2.1 Data Leakage Analysis

The following columns have been removed:

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

#### Resulting dataset with data types optimized 
```
 0   date             150000 non-null  datetime64[ns]
 1   time             150000 non-null  object        
 2   booking_id       150000 non-null  string        
 3   customer_id      150000 non-null  string        
 4   vehicle_type     150000 non-null  category      
 5   pickup_location  150000 non-null  category      
 6   drop_location    150000 non-null  category      
 7   avg_vtat         139500 non-null  float32       
 8   avg_ctat         102000 non-null  float32       
 9   booking_value    102000 non-null  float32       
 10  ride_distance    102000 non-null  float32       
 11  payment_method   102000 non-null  category      
 12  is_cancelled     150000 non-null  float32   
 ```    


## 3.3 Train/test split strategy

The dataset was split using **stratified random sampling** rather than a temporal split. This decision is justified by the EDA findings:

1. **No temporal signal**: Bivariate analysis showed cancellation rate is flat across all temporal dimensions (daily, weekly, monthly, hourly). The 30-day rolling average is nearly constant with no drift.
2. **No seasonal patterns**: Chi-square tests on weekday, month, week_of_year, and quarter all returned negligible effect sizes (Cramér's V ≈ 0).
3. **Stationarity confirmed**: The statistical properties of the data do not change over time, so there's no concept drift to simulate.

Stratified sampling ensures balanced class distribution in both sets, which is more important given the 68/32 class imbalance than preserving temporal order that carries no predictive signal.

| Set | Percentage | Records | Purpose |
|-----|------------|---------|---------|
| Training | 80% | ~120,000 | Model training + cross-validation |
| Test | 20% | ~30,000 | Final evaluation |

## Basic Analysis

No duplicated rows.

## Univariate Analysis

Detailed in `04_univar_eda.ipynb`.

- **Temporal features** (date, time, hour, weekday, month, is_weekend): all weak. Cancellation rate flat at ~32% regardless of time. Cyclical encodings added for completeness.
- **booking_id / customer_id**: both have duplicates that shouldn't exist. Dropped (identifiers, not features).
- **vehicle_type**: 7 categories, all ~32% cancellation. Weak predictor.
- **pickup_location / drop_location**: 176 categories each, faint signal (V ≈ 0.037). Need target encoding.
- **avg_vtat**: range 2–20 min, 7% missing. **Strongest predictor** — non-linear relationship with cancellation.
- **Dropped** (`avg_ctat`, `booking_value`, `ride_distance`, `payment_method`): all share 48K NaNs aligned perfectly with cancellations. Data leakage.
- **is_cancelled** (target): 68/32 split.

### TODO - Is there a relationship between customers booking more than once and cancelling? Same vehicle/location each time? Are repeat-bookers more likely to cancel?
### TODO - No description of vehicle types in the dataset. Understanding their characteristics (size, price tier) could reveal patterns.
### TODO - Find geographical info to compute actual distances and check if pickup == drop.

---

## Bivariate Analysis (Feature vs Target)

Detailed in `05_bivar_eda.ipynb`.

- **All temporal features**: no signal. Every test (chi-square, Fisher's, Spearman) returns negligible effect sizes. Cancellation is time-independent.
- **vehicle_type**: no signal (V = 0.006).
- **pickup/drop_location**: faint signal (V ≈ 0.037). Might improve with target encoding or route-level features.
- **avg_vtat**: dominant predictor with a non-linear, non-monotonic relationship captured by five behavioral zones:
  - **Instant (2–2.9 min)**: 0% cancellation
  - **Low (3–5 min)**: ~26%
  - **Baseline (5.1–11.9 min)**: ~31% (overall mean)
  - **Dip (12–15 min)**: ~9% — sunk-cost effect
  - **Timeout (15.1–20 min)**: 100% — system auto-cancellation
- **vtat_missing**: every row with missing avg_vtat is cancelled (phi = +0.40). Not confounded by other features.
- `vtat_zone` (χ² = 16,819) is the single most powerful feature in the dataset.

### Engineered features

`vtat_zone`, `is_instant_arrival`, `is_timeout`, `is_long_wait`, `vtat_missing`, `weekday`, `month`, `hour`, `is_night`, `is_rush`, `is_weekend`, cyclical encodings.


---

# 4. Multivariate EDA

Detailed in `06_multivar_eda.ipynb`.

## Analyses Performed

1. **Multivariate Missingness Confounding Check**: logistic regression with all non-VTAT features vs is_cancelled-only baseline. AUC lift = 0.002 — missingness is MNAR driven purely by the target with no hidden confounding from feature combinations.

2. **Feature Block Effect Sizes**: grouped features into families (temporal_date, temporal_time, locations, vehicle, vtat) and ranked each by cancellation rate swing against the target.

3. **Family Redundancy Check**: quantified overlap within the vtat family using η² (avg_vtat ↔ vtat_zone = 0.82) and Cramér's V (vtat_zone ↔ vtat_missing). Determined which representative to keep per model family.

4. **Interaction Effects (pickup × drop → route)**: Cramér's V = 0.45 suggested strong route signal, but this was entirely a cardinality artefact (~30k unique routes, 64% with ≤5 observations). Cross-validated target encoding showed zero lift at all smoothing levels.

## Key Findings

### Missingness is not confounded

No combination of non-VTAT features predicts avg_vtat missingness beyond is_cancelled alone. Rides cancelled before vehicle assignment have no VTAT recorded — that's the entire mechanism. Imputation or using vtat_missing as a feature won't introduce bias.

### Feature redundancy resolved

| Feature | Decision | Rationale |
|---------|----------|-----------|
| avg_vtat | Keep for trees | Granular continuous signal; trees handle non-linearity |
| vtat_zone | Keep for LR | Categorical proxy for avg_vtat; avoids log-odds linearity violation (η² = 0.82 with avg_vtat) |
| vtat_missing | Keep for both | Complementary to avg_vtat; captures early cancellations with 100% precision |
| pickup_location | Keep for both | Moderate signal (~10% rate swing), no redundancy with other features |
| drop_location | Keep for both | Same as pickup |
| is_instant_arrival, is_timeout, is_long_wait | Drop | Deterministic subsets of avg_vtat/vtat_zone |
| vehicle_type | Drop | 1% rate swing across 7 types — negligible signal |
| Temporal features | Drop | No signal confirmed across bivariate and multivariate analyses |
| route | Drop | Zero cross-validated lift; Cramér's V was a cardinality artefact |

### Correlation structure

Feature space is largely orthogonal — family redundancy checks (η², Cramér's V) confirmed no hidden dependencies among the final features.

## Final Feature Set for Modeling

| Feature | Logistic Regression | Tree Models | Verdict |
|---------|---------------------|-------------|---------|
| vtat_zone | ✓ Use | ✗ Skip | Categorical, captures non-linear pattern |
| avg_vtat | ✗ Skip | ✓ Use | Violates log-odds linearity for LR |
| vtat_missing | ✓ Use | ✓ Use | Strong signal (100% precision on 22% of cancellations) |
| pickup_location | ✓ Use | ✓ Use | Moderate signal (~10% rate swing) |
| drop_location | ✓ Use | ✓ Use | Moderate signal (~10% rate swing) |

## Encoding Recommendations

- **Logistic Regression**: frequency-encode locations, use vtat_zone as categorical
- **Tree Models**: label-encode locations, use avg_vtat with sentinel imputation (-1), set min_samples_leaf ≥ 50

## Realistic Expectations

- vtat_missing does most of the predictive work (22% of cancellations caught deterministically)
- Remaining features (avg_vtat/vtat_zone, locations) have weak effect sizes (Cohen's d ≈ 0.16)
- This is a fundamentally hard prediction problem with limited signal

## Deferred to Modeling Phase

- Influential point analysis (Cook's distance) for logistic regression

---

# 5. Feature Engineering

Detailed in `07_feature_engineering.ipynb`.

## Pipeline Architecture

Two separate sklearn pipelines were built to handle model-specific feature requirements:

### Logistic Regression Pipeline
- **vtat_zone**: One-hot encoded (5 categories → 5 binary features)
- **vtat_missing**: Binary passthrough
- **pickup_location_freq**: Frequency encoded
- **drop_location_freq**: Frequency encoded

### Tree Models Pipeline
- **avg_vtat_imputed**: Numeric with sentinel -1 for missing
- **vtat_missing**: Binary passthrough
- **pickup_location_encoded**: Label encoded
- **drop_location_encoded**: Label encoded

## Key Design Decisions

1. **No temporal features**: EDA showed zero signal across all temporal dimensions
2. **No vehicle_type**: Only 1% rate swing, negligible predictive value
3. **No route feature**: Failed cross-validation with zero lift at all smoothing levels
4. **Model-specific VTAT handling**: 
   - Zone encoding for LR (captures non-linearity that violates log-odds assumption)
   - Numeric for trees (handle non-linearity naturally)
5. **Sentinel imputation for trees**: -1 allows trees to learn the missing pattern as a split point
6. **Frequency encoding for LR**: Captures location popularity without introducing high cardinality issues

## Artifacts Saved

| Artifact | Path | Purpose |
|----------|------|---------|
| lr_pipeline.joblib | data/silver/feature_engineering/ | Fitted LR preprocessing pipeline |
| tree_pipeline.joblib | data/silver/feature_engineering/ | Fitted Tree preprocessing pipeline |
| X_train_lr.parquet | data/silver/feature_engineering/ | Transformed training features for LR |
| X_test_lr.parquet | data/silver/feature_engineering/ | Transformed test features for LR |
| X_train_tree.parquet | data/silver/feature_engineering/ | Transformed training features for trees |
| X_test_tree.parquet | data/silver/feature_engineering/ | Transformed test features for trees |
| y_train.parquet | data/silver/feature_engineering/ | Training labels |
| y_test.parquet | data/silver/feature_engineering/ | Test labels |
| feature_info.json | data/silver/feature_engineering/ | Feature names metadata |
