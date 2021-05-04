from typing import List
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd



PATH_TO_DATA = 'gs://churn_prediction_inputs/rawdata/curn_prediction_rawdata.csv'
DATE_COL = 'ActiveDate'
USER_ID_COL = 'ID'
TARGET_COL = 'target'
ALL_EQUITY_COLUMN = 'EOM_Equity'
DATES_COLUMNS = ['ActiveDate', 'FTDdate', 'LastPosOpenDate', 'LastLoggedIn']


def convert_objects_to_dates(rawdata:pd.DataFrame, date_columns: List[str] = DATES_COLUMNS) -> pd.DataFrame:
    """
    This function convert dates from object type to date type column inplae
    """
    for date_column in date_columns:
        rawdata[date_column] = pd.to_datetime(rawdata[date_column])
    
    return rawdata


def exclude_users_with_gaps_in_time_points(rawdata: pd.DataFrame) -> pd.DataFrame:
    """
    This function find the user with time gaps and and exclude them form
    the data
    """
    rawdata = rawdata.sort_values(by=[USER_ID_COL, DATE_COL], ascending=False)
    
    # filter useres that have no times gaps
    total_number_of_time_points = rawdata[DATE_COL].nunique()
    users_time_points = rawdata.groupby(USER_ID_COL)[DATE_COL].nunique()
    users_with_potential_gaps = users_time_points[
        users_time_points < total_number_of_time_points].index
    
    users_with_potential_gaps_df = rawdata[rawdata[USER_ID_COL].isin(users_with_potential_gaps)].groupby(USER_ID_COL)
    
    users_to_exclude = set()
    for user_id, user_data in users_with_potential_gaps_df:            
        month_first_record = None
        for index, temp_date in enumerate(user_data[DATE_COL].dt.strftime('%Y-%m')):
            temp_date = datetime.strptime(temp_date, "%Y-%m")

            if index==0:
                month_first_record = temp_date

            elif temp_date+relativedelta(months=index)!=month_first_record:
                users_to_exclude.add(user_id)
    
    return users_to_exclude

                    
def generte_target(rawdata: pd.DataFrame, months_ahead: int = 1) -> pd.DataFrame:
    """
    This function generate the super visor (the target feature)
    """
    # find the follwing next EOM Equity amount 
    rawdata = rawdata.sort_values(by=[USER_ID_COL, DATE_COL], ascending=False)
    rawdata[f'next_{ALL_EQUITY_COLUMN}'] = rawdata.groupby([USER_ID_COL])[ALL_EQUITY_COLUMN].shift(months_ahead)
    
    # filter the last observation because we cont have the lead value (target)
    rawdata = rawdata[~rawdata[f'next_{ALL_EQUITY_COLUMN}'].isna()]
    
    # defined the target variable
    rawdata[TARGET_COL] = np.where(
        (rawdata[f'next_{ALL_EQUITY_COLUMN}'] < 25) & 
        (rawdata[ALL_EQUITY_COLUMN]>=25), 1, 
        np.where(
            (rawdata[f'next_{ALL_EQUITY_COLUMN}'] < 25) & 
            (rawdata[ALL_EQUITY_COLUMN]<25), -1, 0)
    )
    
    return rawdata


def preprocess(path_to_data: str = PATH_TO_DATA):
    """
    """
    # read data
    rawdata = pd.read_csv(path_to_data)
    rawdata.dropna(axis=0, how="all", inplace=True)
    
    # change features types
    rawdata = convert_objects_to_dates(rawdata)
    
    # exclude users with missing time points
    exclude = exclude_users_with_gaps_in_time_points(rawdata)
    rawdata = rawdata[~rawdata[USER_ID_COL].isin(exclude)]
    
    # generate target variable
    rawdata = generte_target(rawdata)
    
    # filter all non relevant rows
    rawdata = rawdata[(~rawdata[TARGET_COL].isna()) &
                      (rawdata[TARGET_COL]>=0)]
    
    return rawdata
    
    
    
    
# def generate_lag_features(rawdata:pd.DataFrame, features_to_lag: List[str],
#                          lags_range: int = 3):
#     """
#     This function generate lag features for each of the features provided in the 
#     list. The number of lags detemined by the lags_range parameter
#     """
    
