#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# set low memory to false to avoid warning
pd.options.mode.chained_assignment = None

def filter_date_limits(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["DATE"] <= datetime.datetime(2016,1,1)) & (df["DATE"] >= datetime.datetime(2014,1,1))]

def filter_geo_limits(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df['GEO_LAT'] > 42.25) & (df['GEO_LAT'] < 42.47)]
    df = df[(df['GEO_LON'] > -83.3) & (df['GEO_LON'] < -82.9)]
    df = df.dropna(subset=['GEO_LAT', 'GEO_LON'])
    return df

def extract_geo_location(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df.dropna(subset=[label], inplace=True)
    df["GEO_LAT"] = df[label].str.extract(r"(\d+\.\d+),\s(-\d+\.\d+)", expand=True)[0].astype(float)
    df["GEO_LON"] = df[label].str.extract(r"(\d+\.\d+),\s(-\d+\.\d+)", expand=True)[1].astype(float)
    df.dropna(subset=['GEO_LAT', 'GEO_LON'], inplace=True)
    return df

def plot_by_geo_location(df: pd.DataFrame, label_data: str ="", color: str ="blue") -> None:
    plt.figure(figsize=(10,3))
    plt.scatter(df['GEO_LON'], df['GEO_LAT'], marker='x', alpha=0.1, label=label_data, color=color)
    plt.xlabel('Lat')
    plt.ylabel('Lon')
    plt.title(label_data)
    plt.show()


def create_geo_location_grid(tile_size: int):
    grid = {"lat":[42.25,42.47],"lon":[-83.3,-82.9],"x": 26300,"y": 26300}
    lat = (grid["lat"][1]-grid["lat"][0])*tile_size/grid["y"]
    lon = (grid["lon"][1]-grid["lon"][0])*tile_size/grid["x"]
    x = int(grid["x"]/tile_size) + 1
    y = int(grid["y"]/tile_size) + 1
    return {"lat":lat,"lon":lon,"x":x,"y":y,"factor":tile_size}


def convert_geo_location_to_grid(df: pd.DataFrame, grid: dict) -> pd.DataFrame:
    df["x"] = ((df["GEO_LON"]-grid["lon"])/grid["lon"])
    df["y"] = ((df["GEO_LAT"]-grid["lat"])/grid["lat"])
    df["x"] = df["x"].apply(lambda x: int(x) if not np.isnan(x) else np.nan)
    df["y"] = df["y"].apply(lambda x: int(x) if not np.isnan(x) else np.nan)
    df = df.dropna(subset=["x", "y"])
    df["x"] = df["x"].astype(int)
    df["y"] = df["y"].astype(int)
    df["GEO_INDEX"] = df["x"] + df["y"]*grid["x"]
    return df


def process_permits(grid_data: dict):
    permits = pd.read_csv("./data/detroit-demolition-permits.tsv", sep="\t")
    permits = permits[permits["PERMIT_APPLIED"].str.contains("^[0-9]{2}/[0-9]{2}/[0-9]{2}$")]
    permits["DATE"] = pd.to_datetime(permits["PERMIT_APPLIED"], format="%m/%d/%y")
    permits["PARCEL_SIZE"] = permits["PARCEL_SIZE"].astype(float).fillna(0)
    permits["PARCEL_GROUND_AREA"] = permits["PARCEL_GROUND_AREA"].astype(float).fillna(0)

    permits = extract_geo_location(permits, "site_location")
    permits = filter_geo_limits(permits)
    permits = filter_date_limits(permits)
    permits = convert_geo_location_to_grid(permits, grid_data)

    permits.to_csv("./data/demolition_permits-cleaned.csv", index=False)

def process_crimes(grid_data: dict):
    crime = pd.read_csv("./data/detroit-crime.csv")
    crime['GEO_LON'] = crime['LON'].astype(float)
    crime['GEO_LAT'] = crime['LAT'].astype(float)
    crime['DATE'] = pd.to_datetime(crime['INCIDENTDATE'])

    crime = filter_geo_limits(crime)
    crime = filter_date_limits(crime)
    crime = convert_geo_location_to_grid(crime, grid_data)

    crime.to_csv("./data/detroit-crime-cleaned.csv", index=False)

def convert_violations_fines(df: pd.DataFrame) -> pd.DataFrame:
    df['FineAmt'] = df['FineAmt'].str.replace('$', '').astype(float)
    df['AdminFee'] = df['AdminFee'].str.replace('$', '').astype(float)
    df['LateFee'] = df['LateFee'].str.replace('$', '').astype(float)
    df['StateFee'] = df['StateFee'].str.replace('$', '').astype(float)
    df['CleanUpCost'] = df['CleanUpCost'].str.replace('$', '').astype(float)
    df['JudgmentAmt'] = df['JudgmentAmt'].str.replace('$', '').astype(float)
    return df

def process_violations(grid_data: dict):
    violations = pd.read_csv('./data/detroit-blight-violations.csv')
    violations = violations[violations["TicketIssuedDT"].str.contains("^[0-9]{2}/[0-9]{2}/[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2}")]
    violations["DATE"] = pd.to_datetime(violations["TicketIssuedDT"])

    violations = extract_geo_location(violations, "ViolationAddress")
    violations = filter_geo_limits(violations)
    violations = filter_date_limits(violations)
    violations = convert_violations_fines(violations)

    violations['VIOLATIONS'] = violations['ViolationCode'].astype('category')
    violations = violations[['VIOLATIONS', 'DATE', 'GEO_LAT', 'GEO_LON', 'FineAmt', 'AdminFee', 'LateFee', 'StateFee', 'CleanUpCost', 'JudgmentAmt']]
    violations = violations.dropna()

    violations = convert_geo_location_to_grid(violations, grid_data)

    violations.to_csv('data/blight_violations_clean.csv', index=False)


def process_311(grid_data: dict):
    issues = pd.read_csv('./data/detroit-311.csv')
    issues['issue_type'] = issues['issue_type'].astype('category')
    issues['DATE'] = pd.to_datetime(issues['acknowledged_at'])
    issues['GEO_LAT'] = issues['lat'].astype('float')
    issues['GEO_LON'] = issues['lng'].astype('float')

    issues = filter_geo_limits(issues)
    issues = filter_date_limits(issues)
    issues = convert_geo_location_to_grid(issues, grid_data)
    issues.to_csv('data/d311_clean.csv', index=False)


def read_clean_data_to_model():
    permits = pd.read_csv("./data/demolition_permits-cleaned.csv")
    violations = pd.read_csv('./data/blight_violations_clean.csv')
    issues = pd.read_csv('./data/d311_clean.csv')
    crimes = pd.read_csv("./data/detroit-crime-cleaned.csv")
    return permits, crimes, violations, issues


def aggregate_by_geo_index(df : pd.DataFrame) -> pd.DataFrame:
    df = df.groupby("GEO_INDEX").agg({
        "GEO_INDEX": "count",
        "GEO_LAT": "mean",
        "GEO_LON": "mean",
        })
    return df


def aggregate_by_permit_type(df : pd.DataFrame) -> pd.DataFrame:
    df = df.groupby("GEO_INDEX").agg({
        "GEO_INDEX": "count",
        "GEO_LAT": "mean",
        "GEO_LON": "mean",
        "PARCEL_SIZE": "sum",
        "PARCEL_GROUND_AREA": "sum",
        })
    df.rename(columns={'GEO_INDEX': 'PERMITS'}, inplace=True)
    return df


def aggregate_by_geo_index_violations(df : pd.DataFrame) -> pd.DataFrame:
    df = df.groupby("GEO_INDEX").agg({
        "GEO_INDEX": "count",
        "GEO_LAT": "mean",
        "GEO_LON": "mean",
        "FineAmt": "sum",
        "AdminFee": "sum",
        "LateFee": "sum",
        "StateFee": "sum",
        "CleanUpCost": "sum",
        "JudgmentAmt": "sum",
        })
    df.rename(columns={'GEO_INDEX': 'VIOLATIONS'}, inplace=True)
    return df


def create_base_df_counter(df : pd.DataFrame) -> pd.DataFrame:
    base_df = pd.DataFrame(index=df.index)
    base_df["GEO_INDEX"] = df["GEO_INDEX"]
    base_df["GEO_LAT"] = df["GEO_LAT"]
    base_df["GEO_LON"] = df["GEO_LON"]
    return base_df


def create_violations_df_counter(df : pd.DataFrame) -> pd.DataFrame:
    base_df = pd.DataFrame(index=df.index)
    base_df["GEO_INDEX"] = df["GEO_INDEX"]
    base_df["GEO_LAT"] = df["GEO_LAT"]
    base_df["GEO_LON"] = df["GEO_LON"]
    base_df["FineAmt"] = df["FineAmt"]
    base_df["AdminFee"] = df["AdminFee"]
    base_df["LateFee"] = df["LateFee"]
    base_df["StateFee"] = df["StateFee"]
    base_df["CleanUpCost"] = df["CleanUpCost"]
    base_df["JudgmentAmt"] = df["JudgmentAmt"]
    return base_df

def create_permits_df_counter(df : pd.DataFrame) -> pd.DataFrame:
    base_df = pd.DataFrame(index=df.index)
    base_df["GEO_INDEX"] = df["GEO_INDEX"]
    base_df["GEO_LAT"] = df["GEO_LAT"]
    base_df["GEO_LON"] = df["GEO_LON"]
    base_df["PARCEL_SIZE"] = df["PARCEL_SIZE"]
    base_df["PARCEL_GROUND_AREA"] = df["PARCEL_GROUND_AREA"]
    return base_df


def process_data():
    grid = create_geo_location_grid(tile_size=25)
    process_permits(grid)
    process_crimes(grid)
    process_violations(grid)
    process_311(grid)

    permits, crimes, violations, issues = read_clean_data_to_model()

    permits_count = create_permits_df_counter(permits)
    agg_permits = aggregate_by_permit_type(permits_count)

    violations_count = create_violations_df_counter(violations)
    agg_violations = aggregate_by_geo_index_violations(violations_count)

    crimes_count = create_base_df_counter(crimes)
    agg_crimes = aggregate_by_geo_index(crimes_count)
    agg_crimes.rename(columns={'GEO_INDEX': 'CRIMES'}, inplace=True)

    issues_count = create_base_df_counter(issues)
    agg_issues = aggregate_by_geo_index(issues_count)
    agg_issues.rename(columns={'GEO_INDEX': 'ISSUES'}, inplace=True)


    def add_features(df, line, label, grid: dict, factor=25):
        x = grid["lat"]*(factor+0.5)
        y = grid["lon"]*(factor+0.5)
        lat_sel = np.logical_and(df.GEO_LAT < line["GEO_LAT"]+x,df.GEO_LAT > line["GEO_LAT"]-x)
        long_sel = np.logical_and(df.GEO_LON < line["GEO_LON"]+y,df.GEO_LON > line["GEO_LON"]-x)
        g_sel = np.logical_and(lat_sel,long_sel)
        return df.loc[g_sel,label].sum()

    def merge(df1, df2, label, grid: dict):
        a1 = df1.apply(lambda x: add_features(df2,x,label,grid),axis=1)
        a1.name = label
        return df1.merge(a1,left_index=True,right_index=True)

    out = merge(agg_permits, agg_crimes, "CRIMES",grid)
    out = merge(out, agg_issues, "ISSUES", grid)
    out = merge(out, agg_violations, "VIOLATIONS", grid)
    return out


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def predict_crimes(data: pd.DataFrame):
    # use random forest to predict the number of crimes in a given area

    train, test = train_test_split(data, test_size=0.2)

    # train the model
    model = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=0)
    model.fit(train.drop(columns=['CRIMES']), train['CRIMES'])

    # predict the test data
    predictions = model.predict(test.drop(columns=['CRIMES']))

    # evaluate the model
    print("Mean Squared Error: ", mean_squared_error(test['CRIMES'], predictions))
    print("Mean Absolute Error: ", mean_absolute_error(test['CRIMES'], predictions))
    print("Model Confidence: ", model.score(test.drop(columns=['CRIMES']), test['CRIMES']))

    # plot the results
    plt.scatter(test['CRIMES'], predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.show()


def main():
    data = process_data()
    data.to_csv("./data/combined-grid-agg.csv")
    data = pd.read_csv("./data/combined-grid-agg.csv")
    predict_crimes(data)

__init__ = main()