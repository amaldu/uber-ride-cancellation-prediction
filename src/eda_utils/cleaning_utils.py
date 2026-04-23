import re
import pandas as pd

############################################################
# config 
############################################################


BOOKING_STATUS_MAP = {
    "Completed": 0,
    "Cancelled by Driver": 1,
    "No Driver Found": 1,
    "Cancelled by Customer": 1,
    "Incomplete": 0,
}

LEAKY_COLUMNS = [
    "cancelled_rides_by_customer",
    "reason_for_cancelling_by_customer",
    "cancelled_rides_by_driver",
    "driver_cancellation_reason",
    "incomplete_rides",
    "incomplete_rides_reason",
    "driver_ratings",
    "customer_rating",
    "avg_ctat",
    "booking_value",
    "ride_distance",
    "payment_method"]

ID_COLUMNS = ["booking_id", "customer_id"]

CATEGORICAL_COLUMNS = ["vehicle_type",
                       "pickup_location",
                       "drop_location"]
NUMERIC_COLUMNS = ["avg_vtat"]


############################################################
# data cleaning notebook helpers
############################################################

def to_snake_case(df:pd.DataFrame) -> pd.DataFrame:
    new_columns = []
    for col in df.columns:
        col = col.strip()
        col = re.sub(r"[\s\-]+", "_", col)
        col = re.sub(r"[^\w_]", "", col)
        col = col.lower()
        new_columns.append(col)
    
    df.columns = new_columns
    return df


############################################################
# cleaning pipeline 
############################################################

def clean(df:pd.DataFrame) -> pd.DataFrame:
    df = to_snake_case(df)

    df["is_cancelled"] = df["booking_status"].map(BOOKING_STATUS_MAP)
    df = df.drop(columns=["booking_status"])

    df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns])
    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    df["is_cancelled"] = df["is_cancelled"].astype("bool")


    return df
