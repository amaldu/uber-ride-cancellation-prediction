# Project Walkthrough

Here I explain the step-by-step logic and workflow followed in this machine learning project, from data collection to model evaluation. 

The goal of the project is to predict Uber ride cancellations using machine learning to enable proactive intervention by creating strategies to reduce the overall cancellation rate.

# Problem Framing
## 1.1 Defininition of the objective in business terms

Looking at the Uber ride data from 2024, I identified a critical operational challenge, 37.430 rides out of 148.770 the total bookings never reached completion. 

This means that the 25% (37.430 rides) of all bookings end in cancellation of which a 19.15% (27.000 rides) are made by customers and 7.45% (10.500 rides) are made by drivers. This as a result, means that for every 4 ride requests, 1 fails to complete. 

In terms of business impact, I can think of the following areas:

1. Financial impact:
   1. Revenue loss: the dataset does not provide the ride prices but from my experience, if I estimate an averange booking value of 20$ per ride without taking into consideration the type of vehicle, the 37.430 cancelled rides represent 748.600$ in lost revenue in 2024. 
   2. Hidden operational costs: there are other costs like the time and fuel wasted by drivers while the ride is not cancelled yet, the costs of the processing platform (servers, computations, payment platform usage) and the potential churn of those frustrated customers.
   3. Opportunity cost: every cancellation activates resources within the company that also have a cost, some examples can be contacting customer supports, making refunds, managing complains, etc. They cost money that could be invested in other areas to promote the growth of the company.

2. Operational impact:
   1. Imbalance in supply and demand: when drivers are removed from the pool of available services while they are on their way to pick up the customer. If the price per ride depends on the amount of available drivers, this can artificially increase the price of the available services and potentially damaging the image of the company. 
   2. Driver insatisfaction: drivers are also consuming Uber's service and they also become frustrated and potentially change their providers if the customer is cancelling their rides too often. 
   
Let's imagine that Uber wants to reduce the cancellation rate by 10%, this means 3.700 more rides completed and almost 75000$ recovered so building a predictive model that identifies bookings with high cancellation risk at the time of booking could help achieving it.

## 1.2 How will the solution be used?

The model will be deployed as a real-time prediction system integrated into Uber's booking workflow so when a customer requests a ride, the model will score the cancellation probability.

This will help other parts of the company to develop and implement long-term fixes like:

- Increase customer engagement in high-risk rides: send confirmation messages, increase ETA updates, create loyalty points, etc. 
- Cluster and route drivers: based on their experience in completion and ratings, Uber can send drivers to higher risk rides.

All those techniques can also be monitored accross time and regions to better understand issues in their implementation and evolution over time.

## 1.3 What are the current solutions/workarounds (if any)?

Neither the metadata nor the dataset provide any other information about different approaches previously taken by the company.

## 1.4 How should the problem be framed

I'm going to start framing it as a supervised binary classification problem: cancelled vs. completed even though in future approaches it would be interesting to try a multi-class approach: cancelled by customer / driver vs. completed vs. incompleted if business value is demonstrated. 

The training will be offline on historical batch data I would plan to re-train it daily and monitor concept drift to check if concept drift requires faster adaptation and we have to change to near-online or online training.

The inference will be in real time when booking happens. 

## 1.5 How should performance be measured?

The dataset shows that we have a cancellation rate of 25%, which is moderately imbalanced so I need metrics that go beyond simple accuracy. 

Here you can find a cost matrix applied to the concrete business scenarios that I defined after having conversations with product or the Ops team in order to measure the net business cost of every possible outcome:

| Outcome | What happens | Cost/Benefit | Business Meaning |
|------------|----------------|----------------|------------------|
| True Positive (TP) | Ride is Cancelled & System intervenes | +$15 benefit | Prevented cancellation saves ~$20 revenue, minus ~$5 intervention cost |
| False Positive (FP) | Ride is NOT Cancelled & System intervenes | -$5 cost | Unnecessary intervention (incentive + operational time) |
| True Negative (TN) | Ride is NOT Cancelled & System does NOT intervene | $0 | Normal operation, no action needed |
| False Negative (FN) | Ride is Cancelled & System does NOT intervene | -$20 cost | Lost booking revenue + customer dissatisfaction + driver wasted time |

**Key insight here: the cost asymmetry is approximately 4:1!**

A missed cancellation (FN) costs $20, while a false alarm (FP) costs $5. This means that I should be willing to accept up to 4 false alarms to catch 1 additional cancellation.

This problem requires of metrics that give importance to the balance between catching cancellations (recall) with not overwhelming operations with false alarms (precision).

- **Recall** = TP / (TP + FN) = "Of all actual cancellations, what % did I catch?"
- **Precision** = TP / (TP + FP) = "Of all my cancellation predictions, what % were correct?"

These metrics have trade-offs because:
- If I want to catch MORE cancellations (increase recall), I will end up catching more false alarms (precision)
- If I want FEWER false alarms (increase precision), I will end up missing more cancellations (lower recall)
  
I need to find the right balance, and that balance depends on the **net business costs** of each type of error.

**Calculations**

*Precision*: 

If I intervene on a predicted cancellation:
- Expected benefit if correct (probability = precision): Save $20
- Expected cost if wrong (probability = 1 - precision): Waste $5

```
Precision × $20 = (1 - Precision) × $5
20P = 5 - 5P → 25P = 5 → P = 0.20
```

So every decision above 20% generates positive value BUT the business value here is also important. For example, the operational capacity, the customer experice or the credibility are also factors that are important to take into consideration and are not measured here. In order to give a valuable increase that protects these factors I will increase Precision to the 50% (every second ride).

*Recall*:

Given the asymmetry in the cost (FN is way more expensive than FP), I will prioritize recall over precision. Having 37.430 cancellations and a defined 60% of precision, let's talk about the constrains applied to this metric:

1. How many interventions can I make?
   The operational capacity is very important to take into consideration. Let's say we start with a model that can classify 50.000 rides/year as a start point. 

```
TP = Total interventions x precision = 50.000 x 0.60 = 30.000
Recall = TP / Real Positives = 30.000 / 37.430 = 0.801 ~ 80%
```
With 50.000 interventions/year I can reach around 80% of recall

2. What is the minimum ROI required?
   The project needs to generate enough davings to justify its existence, let's make some calculations:

   If the costs of each possible outcome are:
   - Missed cancellation (FN): $20
   - False alarm (FP): $5
   - Success rate of every intevention was set at 50%

```Revenue at risk = 37.430 / 20 = 748.600 dollars
Expected savings = TP x 20 x 0.5 - FP x 5 

    To get FP from precision:
    FP = TP x (1 - precision)/precision = TP x 0.4/0.6 = TP x 0.667

Then expected savings are:
Expected savings = TP x 20 x 0.5 - (TP x 0.667) x 5 = TP x 6.67

```

Time to get back to the financial team and decide which is the minimum cost. They say the minimum target is $50.000/ year and the target is $100.000/year:

Minimum savings to be viable:
```
TP x 6.67 ≥ 50.000 → TP ≥ 50.000/6.67 = 7.496'25
Recall = 7.496/ 37.430 = 0.2
```

Target savings:
```
TP x 6.67 ≥ 100.000 → TP ≥ 100.000/6.67 = 14.992'5
Recall = 14.993/ 37.430 = 0.4
```

Let's gather all the constrains and their results:

| Constraint | Derived Recall Range | Limiting Factor |
|------------|---------------------|-----------------|
| Break-even (minimum useful) | ≥ 20% | Below this, savings don't cover costs |
| Operational capacity | ≤ 80% | Maximum for 50K interventions with 60% precision |
| Target ROI | ≥ 40% | Requirement decided by the business |

Based on these values I will choose a 70% of recall. 


### Validation of both metrics → Precision = 60% and Recall = 70%:

```
TP = caugth cancellations = 0.7 X 37.430 = 26201  
FP = false alarms = 26.201 x 0.4/0.6 = 17.467 
Total interactions = 26.201 + 17.467 = 43.668 →  less than operational capacity

&

Savings = 26.201 x 20 x 0.5 = 262.010
False alarm cost = 17.467 x 5 = 87.335
Net savings = 262.010 - 87.335 = 174.675 → more than target ROI 

&

Precision  > Break-even precision (20%)
```

### Final metrics:

Based on the analysis above I decided to choose metrics that:

1. Reflect the cost assymetry of 4:1
2. Work for moderated class imbalance


| Metric | Value | Reasoning |
|------------|---------------------|-----------------|
| F2-score  | Since β² =$20/$5 = 4 ≥ 0.68 F2 = 5×P×R / (4P + R) = 0.68 | Weights recall 4x more than precision |
| Recall | ≥ 70% | Catching FN is the real challenge due to their cost |
| Precision | ≥ 60% | Also important due to FP costs |
| PR-AUC Curve | The best possible | Useful because the dataset is imbalanced |
| Expected profit | ≥ $100K | Final sanity check: Profit = TP×$10 - FP×$5. Validates that model delivers actual business value. |

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
| AUC-ROC | 0.25 | Extracted from ositive class proportion |


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


## 1.11 List the assumptions you (or others) have made so far

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

First of all I assume that the dataset contains all relevant features required to model the target variable, and that no critical predictive information is systematically missing.

Since this is the first version of the project those will be the project I will only verify the following assumptions:

| Assumption | Verification Method | Status |
|------------|---------------------|--------|
| Data completeness | Check missing value rates | to verify in EDA |
| Feature distributions | perform a statistical analysis | To verify in EDA |
| Class balance | Count target classes | Verified: 25% cancellation rate |
| Temporal patterns | Time-series analysis | To vrify in EDA |
| Feature correlations | perform a correlation analysis | To verify in EDA |

These are the assumptions that require business input and post-deplopyment monitorization that can be verified after the model is deployed: 

| Assumption | Required Information |
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
- **148,770 booking records** from 2024
- **20 columns** covering all required feature categories (according to what is assumed in the previous section)
- **Full year coverage** with timestamp-level granularity

**Data Sufficiency Assessment**:
- The sample size is adequate for ML modeling 
- The positive class (cancellations) has sufficient examples (25% of that class)
- All the categories are represented and have enough positive class examples
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
