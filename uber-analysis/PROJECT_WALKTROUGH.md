# Project Walkthrough

Here I explain the step-by-step logic and workflow followed in this machine learning project, from data collection to model evaluation. 

The goal of the project is to predict Uber ride cancellations using machine learning to enable proactive intervention by creating strategies to reduce the overall cancellation rate.

# Problem Framing
## 1.1 Defininition of the objective in business terms

Looking at the Uber ride data from 2024, I identified a critical operational challenge, 37.500 rides out of 150.000 the total bookings never reached completion. 

#FIXME - recheck the values with the real dataset size
This means that the 25% (37.500 rides) of all bookings end in cancellation of which a 19.15% (27.000 rides) are made by customers and 7.45% (10.500 rides) are made by drivers. This as a result, means that for every 4 ride requests, 1 fails to complete. 

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
| Class balance | Verify that the imbalance is 25% | Verified: 25% cancellation rate | None |
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
- The positive class (cancellations) has sufficient examples (25% of that class)
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


#FIXME - move this to later
## 3.3 Train/test/val split strategy

The dataset was split temporally. Holiday periods such as December were intentionally kept in the test set in order to evaluate the model’s robustness under seasonal regime shifts and peak-demand conditions. Training and validation sets will be used to tune the model under regular operating conditions

| Set | Percentage | Records | Purpose |
|-----|------------|---------|---------|
| Training | 75% | ~112,000 | Model training |
| Validation | 15% | ~25,000 | Hyperparameter tuning |
| Test | 10% | ~12,000 | Final evaluation |

## Basic Analysis


There are no duplicated rows

## Univariate Analysis

  
### date
- Type: object, later will merge with time and convert to datetime
- Range: 2024-01-01 to 2024-09-30 with no missing days but I need to contact Ops Team to discover why we have no data from the 31st of Dec.
- Unique values = 365 because is leap year 
- NaN: no. Every day has between ~350 and ~450 records
- Observations: no clear patterns
  
### time	
- Type: object. Impossible to convert into datatime without a "date". Will merge after the analysis
- Range: 00:00:00 to 23:59:59
- Unique values: too many to see something
- NaN: no
- Observations: bimodal pattern. Activity is minimal during the early morning hours, increases sharply during the morning commute with a peak around 10 AM, dips around midday, and reaches its highest level in the early evening (around 6 PM). Demand then steadily declines toward the end of the day.

### datetime (merge of date and time)
- Type: datetime
- Range: full year. 2024 started on Monday
- NaN: no

### hour
- Type: int32
- Range: 0-23
- NaN: no
- Observations: Created because time already showed clear patterns.

### weekday
- Type: int32
- Range: 0-6
- NaN: no
- Observations: There is no clear pattern

### is_weekend
- Type: bool
- Range: True - False
- NaN: no
- Observations: There is no clear pattern

### hour_sin & hour_cos
Encoded data using sine and cosine transformations to capture the cyclical nature of time

------------------------------------

### booking_id	
- Type: string[pyarrow]
- Range: 1405409 - 2726142 not consecutive
- Unique: 148767
- NaN: no
- Duplicated: 2457 values. The column is described in the `DATASET_INFO.md` as unique identifier for each booking but there are duplicated values: 
   1. 2430 bookings_id duplicated twice
   2. 27 bookings_id duplicated thrice
   
- Observations: the code starts with CNR followed by 7 numbers. IDs do not increment sequentially because they are likely generated by different services (to confirm)

### customer_id	
- Type: string[pyarrow]
- Range: 1000119 - 4523979 not consecutive
- Unique: 148788
- NaN: no
- Duplicated: 2418 values. The column contains unique customer_id so the analysis assuming that duplicated are the same customer using the service more than one time. The duplicated rows can be divided into:
   1. 2400 customer_id duplicated twice
   2. 18 customer_id duplicated thrice
   3. 147582 customer_id appearing once
   
- Observations: the code starts with CID followed by 7 numbers. IDs do not increment sequentially. 
  
  # TODO - is there a relationship between customers booking more than once and cancelling? does it happen trying to book the same vehice or leaving from the same place? (error in the platform) are multiple booking user more likely to cancel? 


### vehicle_type	
- Type: category
- Range: 'Auto', 'Go Mini', 'Go Sedan', 'Bike', 'Premier Sedan', 'eBike', 'Uber XL'
- Unique: 7
- NaN: no
- Observations: all columns contain enough data to be analysed

# TODO - there is no explanation about the types of cars, it would be interesting to analyse and understand them because maybe we can find more patterns relating the characteristics of each car with the behaviour of the customers (ie. size of vehicle and distance with cancellation)



### pickup_location	
- Type: category
- Range: 790 - 949 
- Unique: 176
- NaN: no
- Observations: I cannot check the format of the entries due to lack of knowledge and time. 

# TODO - I would try to find more geographical information to analyse the correlation between distance and target. Check if pickup and drop locations are the same


### drop_location	
- Type: category
- Range: 774 - 936
- Unique: 176
- NaN: no
- Observations: I cannot check the format of the entries due to lack of knowledge and time. 

# TODO - I would try to find more geographical information to analyse the correlation between distance and target. Check if pickup and drop locations are the same

Why there are the same NaNs for pickup and drop locations? Some services are car rentals and you have to introduce the location and destination before booking. Others like Bike you pick it up where it is (current location) and drop it somewhere the app does not know if you cancel

### avg_vtat	
- Type: float32
- Range: 2 - 20. Both make sense. 
- Unique: 181. Representing time, so it's continuous numerical
- NaN: 10500 (7.0%)
- Skewness: 0.30 (right skewness, aprox symmetric -> negligible)
- Kurtosis: -0.59 (platykurtic -> negligible)
- Observations: 
   The distribution is unimodal and slightly right-skewed. Most observations are concentrated in the range of 2-15 minutes
   There's no need for transformation. 
- Outliers: no


### avg_ctat	
- Type: float32
- Range: 10 - 45. Both make sense. 
- Unique: 351. Representing time, so it's continuous numerical
- NaN: 48000 (32.0%)
- Skewness: 0.04 (right skewness, negligible)
- Kurtosis: -1.123 (platykurtic, moderate)
- Observations: 
   The distribution is unimodal and slightly right-skewed with potential outliers. Most observations are concentrated in the range of 2-15 minutes
   There is no need for transformation.

   this feature will not be utilized in the model because it contains missing values exclusively for cancelled bookings
- Outliers: no


### booking_value	
Total fare amount for the ride in Dollars
- Type: float32
- Range: 50 - 4277. Max make sense? Gotta check the outliers
- Unique: 2566. Representing fare amounts makes total sense
- NaN: 48000 (32.0%). 
- Skewness: 2.28 (right skewness)
- Kurtosis: 9.88 (leptokurtic)
- Observations: There's the same amount of NaNs as avg_ctat.  
  When Uber offers you a ride with a car it gives you a final price before booking but there is no such information. When booking a bike or ebike it makes sense. 

  The distribution is unimodal and more strongly right-skewed than the previous variables. It's important to check for outliers Most observations are concentrated in the range of 50-1000 dollars.
  
  Potential candidate for transformation
  this feature will not be utilized in the model because it contains missing values exclusively for cancelled bookings
- Outliers: 3435 (2.29%)
  

### ride_distance	
Distance covered during the ride (in km) 
- Type: float32
- Range: 1 - 50
- Unique: 4901
- NaN: 48000 (32.0%). 
- Skewness: 0.12 (right skewness, negligible)
- Kurtosis: -1.21 (platykurtic, negligible for now)
- Observations: There's the same amount of NaNs as avg_ctat. 
  When Uber offers you a ride with a car it gives you a final price before booking but there is no such information. When booking a bike or ebike it makes sense. 

  The distribution is unimodal and more strongly right-skewed than the previous variables. It's important to check for outliers Most observations are concentrated in the range of 50-1000 dollars.
  
  Potential candidate for transformation
  this feature will not be utilized in the model because it contains missing values exclusively for cancelled bookings

- Outliers: none

### payment_method	
- Type: category
- Range: 774 - 936
- Unique: 5
- NaN: 48000 (32%)
- Observations: this feature will not be utilized in the model because it contains missing values exclusively for cancelled bookings

### is_cancelled (target)
- Type: float32
- Range: 0-1 (0 not cancelled & 1 cancelled)
- Unique: 2
- NaN: no
- Observations: there is a clear imbalance
```
is_cancelled
0.0 --> 68%
1.0 --> 32%
```












### Data Types by Category

| Category | Columns | Old Type | New Type | Additional Explanation |
|---------------|---------|----------------|------------------|-------------|
| Temporal | date  | object | datetime |
| Temporal | time  | object | object | This variable contains only hours so it will be merged with "date" in future steps |
| Identifiers | booking_id, customer_id  | object  | string |
| Categorical| vehicle_type, booking_status, pickup_location, drop_location, payment_method | object | category |
| Numerical | avg_vtat, avg_ctat, booking_value, ride_distance, driver_ratings, customer_rating | float64, int64 | float32 |
| Boolean/Flag | cancelled, cancelled_rides_by_customer, cancelled_rides_by_driver, incomplete_rides | float64 | float32  |
| Text | reason_for_cancelling_by_customer, driver_cancellation_reason, incomplete_rides_reason | object | string