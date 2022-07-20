import sqlite3
import pandas as pd
import matplotlib.dates as md
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import time
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import sys

data = "hour";

def readData(bts_id="10-1"):
    con = sqlite3.connect("data/anomaly_latest.db")
    query = "SELECT * from testdata_hour where BTS_ID='%s'" % bts_id;
    df = pd.read_sql_query(query, con)
    return df

def printData(df):
    print(df.shape)
    print(df.head())
    print (df.info())
    print ( df.describe())

def checkForNull(df):
    print ( df.isnull().any().any())

def convertToDateTime(df):
    df1 = df.copy()
    df1['day'] =  pd.to_datetime(df1['day'])
    print("Day object type")
    print(df1[['day']].info())
    return df1

def dropCol(df):
    columns_to_drop = ['day', 'BTS_ID']
    df_day = df[['day']]
    df = df.drop(columns_to_drop, axis=1)
    return df

def scale(df):
    scaler = RobustScaler()
    start_time = time.time()
    np_scaled = scaler.fit_transform(df)
    end_time = time.time()
    print("Scale Time --- %s seconds ---" % (end_time - start_time))
    df2 = pd.DataFrame(np_scaled)
    df2.columns = df.columns
    return df2

        
def doPCA(dataset):
    start_time = time.time()
    print(componentList)
    pca = PCA(n_components=number_of_components)
    pca.fit(dataset)
    pca_dataset = pca.transform(dataset)
    #saveModel(LE, "PcaModel.ml")
    print ( pca_dataset.shape)
    #print ( final_dataset.shape)
    print ( dataset.shape)
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
    return pca_dataset;

def doOneClassSVM(dataset):
    outliers_fraction = 0.2
    start_time = time.time()
    #clf = OneClassSVM(gamma='auto').fit(dataset)
    clf = OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.01).fit(dataset)
    end_time = time.time()
    print("Fit Time --- %s seconds ---" % (end_time - start_time))
    df1_y = clf.predict(dataset)
    end_predict_time = time.time()
    print("Predict Time --- %s seconds ---" % (end_predict_time - end_time))
    return df1_y

def evaluateClustersKMeans(dataset):
    n_cluster = range(1, 10)
    kmeans = [KMeans(n_clusters=i).fit(dataset) for i in n_cluster]
    scores = [kmeans[i].score(dataset) for i in range(len(kmeans))]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(n_cluster, scores)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    #plt.show();
    plt.savefig("./images/KMeans_Cluster_"+data+".png")

componentList = []
number_of_components = 1
for i in range(0, number_of_components):
    j = i + 1
    componentList.append("comp"+str(j))
def doIsolationForest(dataset, df_plot):
    # Isolation forest 
    outliers_fraction = 0.1
    ifo = IsolationForest(contamination = outliers_fraction)
    ifo.fit(dataset)
    df_plot['anomaly_if'] = pd.Series(ifo.predict(dataset))
    fig, ax = plt.subplots(figsize = (10, 5))
    a = df_plot.loc[df_plot['anomaly_if'] == -1, ['day', 'comp1']]
    ax.plot(df_plot['day'], df_plot['comp1'], 
            color = 'orange', label = 'Normal')
    ax.scatter(a['day'], a['comp1'], 
               color = 'red', label = 'Anomaly')
    plt.legend()
    for index, row in a.iterrows():
        day = row['day']
        comp1 = row['comp1']
        plt.text(day, comp1 , day)
    plt.xlabel('Date Time')
    plt.ylabel('PCA Performance Indicator')
    plt.title('IsolationForest to detect Anomaly')
    plt.savefig("./images/IsolationForest_"+data+".png")
    #plt.show();

def doDBSCAN(dataset):
    dbscan = DBSCAN(eps=0.6, min_samples=2)
    dbscan_result = dbscan.fit_predict(dataset)
    return dbscan_result

def process(bts_id='10-1'):
    df = readData(bts_id)
    printData(df)
    checkForNull(df)
    df1 = convertToDateTime(df)
    df1 = dropCol(df1)
    df2 = scale(df1)
    plt.style.use("Solarize_Light2")
    pca_dataset = doPCA(df2)
    reduced_data_set = pca_dataset
    y_predict = doOneClassSVM(reduced_data_set)
    df_pca = pd.DataFrame(reduced_data_set, columns = componentList)
    df_pca['time'] =  pd.to_datetime(df['day'])
    df_pca = df_pca[['time','comp1']]
    df_pca.head()
    df_plot = df.copy()

    df_plot['day'] =  pd.to_datetime(df_plot['day'])
    df_plot['anomaly3'] = pd.Series(y_predict)
    reduced_data_set[:,0:1].shape
    df_plot['comp1'] = reduced_data_set[:,0:1];
    # df_plot['comp1'] = pd.Series(reduced_data_set[:,0:1])
    #df_plot.head()

    print ( np.unique(y_predict, return_counts=True))

    fig, ax = plt.subplots(figsize=(10,6))
    a = df_plot.loc[df_plot['anomaly3'] == -1, ['day', 'comp1']] #anomaly

    ax.plot(df_plot['day'], df_plot['comp1'], color='blue', label ='Normal')
    ax.scatter(a['day'],a['comp1'], color='red', label = 'Anomaly')
    for index, row in a.iterrows():
        day = row['day']
        comp1 = row['comp1']
        plt.text(day, comp1 , day)
    plt.legend()
    plt.xlabel('Date Time')
    plt.ylabel('PCA Performance Indicator')
    plt.title('One Class SVM Anomaly')
    plt.savefig("./images/oneClassSVMAnomaly_"+data+".png")
    #plt.show()
    evaluateClustersKMeans(reduced_data_set)
    km = KMeans(n_clusters=3)
    km.fit(reduced_data_set)
    y_predict_km=km.predict(reduced_data_set)
    df_plot['anomaly4'] = pd.Series(y_predict_km)
    print ( np.unique(y_predict_km, return_counts=True))
    fig, ax = plt.subplots(figsize=(10,6))
    a = df_plot.loc[(df_plot['anomaly4'] > 0), ['day', 'comp1']] #anomaly

    ax.plot(df_plot['day'], df_plot['comp1'], color='blue', label ='Normal')
    ax.scatter(a['day'],a['comp1'], color='red', label = 'Anomaly')
    plt.legend()
    for index, row in a.iterrows():
        day = row['day']
        comp1 = row['comp1']
        plt.text(day, comp1 , day)
    plt.xlabel('Date Time')
    plt.ylabel('PCA Performance Indicator')
    plt.title('K-Means with 3 Clusters Anomaly')
    plt.savefig("./images/kmeansAnomaly_"+data+".png")
    #plt.show();
    from kats.consts import TimeSeriesData
    df_pca1 = df_pca.copy()
    df_pca1['value'] = df_pca1['comp1']
    columns_to_drop = componentList
    df_pca1 = df_pca1.drop(columns_to_drop, axis=1)
    from kats.detectors.outlier import OutlierDetector
    tsd = TimeSeriesData(df_pca1)
    ts_outlierDetection = OutlierDetector(tsd)
    ts_outlierDetection.detector()

    df_plot['anomaly5'] = np.where(df_plot['day'].isin(ts_outlierDetection.outliers[0]), 0, 1)

    fig, ax = plt.subplots(figsize=(10,6))
    #df['hasimage'] = np.where(df['photos']!= '[]', True, False)
    # a = df_plot.loc[(df_plot['anomaly4'] > 0) |  (df_plot['anomaly4'] > 1), ['day', 'comp1']] #anomaly
    a = df_plot[df_plot['day'].isin(ts_outlierDetection.outliers[0])]
    ax.plot(df_plot['day'], df_plot['comp1'], color='blue', label ='Normal')
    ax.scatter(a['day'],a['comp1'], color='red', label = 'Anomaly')
    plt.legend()
    for index, row in a.iterrows():
        day = row['day']
        comp1 = row['comp1']
        plt.text(day, comp1 , day)
    plt.xlabel('Date Time')
    plt.ylabel('PCA Performance Indicator')
    plt.title('Outlier Detection to find Anomaly')
    plt.savefig("./images/outlierDetection_"+data+".png")
    #plt.show();
    doIsolationForest(reduced_data_set, df_plot)
    dbscan_result = doDBSCAN(reduced_data_set)
    df_plot['anomaly_dbscan'] = pd.Series(dbscan_result)
    print ( np.unique(dbscan_result, return_counts=True))
    fig, ax = plt.subplots(figsize = (10, 5))
    a = df_plot.loc[df_plot['anomaly_dbscan'] == -1, ['day', 'comp1']]
    ax.plot(df_plot['day'], df_plot['comp1'], 
            color = 'orange', label = 'Normal')
    ax.scatter(a['day'], a['comp1'], 
            color = 'red', label = 'Anomaly')

    plt.xlabel('Date Time')
    plt.ylabel('PCA Performance Indicator')
    plt.title('DBSCAN to detect Anomaly')
    for index, row in a.iterrows():
        day = row['day']
        comp1 = row['comp1']
        plt.text(day, comp1 , day)
    plt.savefig("./images/dbscan_"+data+".png")
    plt.legend()
    #plt.show()
    df_plot['anomaly_all'] = 0
    df_plot['anomaly_all'] = np.where(df_plot['anomaly3'] == -1, df_plot['anomaly_all']+1, df_plot['anomaly_all'])
    df_plot['anomaly_all'] = np.where(df_plot['anomaly4'] > 0, df_plot['anomaly_all']+1, df_plot['anomaly_all'])
    df_plot['anomaly_all'] = np.where(df_plot['anomaly5'] == 0, df_plot['anomaly_all']+1, df_plot['anomaly_all'])
    df_plot['anomaly_all'] = np.where(df_plot['anomaly_if'] == -1, df_plot['anomaly_all']+1, df_plot['anomaly_all'])
    df_plot['anomaly_all'] = np.where(df_plot['anomaly_dbscan'] == -1, df_plot['anomaly_all']+1, df_plot['anomaly_all'])
    df_plot.head()
    fig, ax = plt.subplots(figsize=(10,6))
    a = df_plot.loc[df_plot['anomaly_all'] > 2, ['day', 'comp1']] #anomaly
    ax.plot(df_plot['day'], df_plot['comp1'], color='blue', label ='Normal')
    ax.scatter(a['day'],a['comp1'], color='red', label = 'Anomaly')
    plt.legend()
    plt.xlabel('Date Time')
    plt.ylabel('PCA Performance Indicator')
    plt.title('Combine Model to detect Anomaly')
    for index, row in a.iterrows():
        day = row['day']
        comp1 = row['comp1']
        plt.text(day, comp1 , day)


    plt.savefig("./images/MaxAnomalyInAllModel_"+data+".png")
    #plt.show();

if __name__ == "__main__":
    try:
        arg_command = sys.argv[1]
    except IndexError:
        arg_command = "10-1"
    
    print ( arg_command)
    process(arg_command)
