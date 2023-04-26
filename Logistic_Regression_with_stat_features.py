# purpose: Classification of the rock drill machine faults using Logistic Regression
# author: Sanjay Hegde

# importing required libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
import datasource_config


# function for logistic regression classifier implementation
def predict_using_LR(data):

    # using the statistical features from the pre-processed data
    sensor_data = data

    # selecting input features and target class
    features = sensor_data.iloc[:, 1:]
    target = sensor_data.iloc[:, 0]

    # split the train and test data
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # define the multinomial logistic regression model
    model = LogisticRegression(multi_class='ovr', solver='liblinear')

    # training the model
    st_time = time.time()
    model.fit(x_train, y_train)
    end_tm = time.time()
    print("Training Time : " + str(round(end_tm - st_time)) + " seconds")

    train_pred = model.predict(x_train)
    print("Training Accuracy : " + str((accuracy_score(y_train, train_pred) * 100)) + " %")

    # testing the model with test data
    y_pred = model.predict(x_test)

    # performance evaluation using different metrics
    r2_val = r2_score(y_test, y_pred)
    print("R2 value : " + str(r2_val))
    print("Testing Accuracy : " + str((accuracy_score(y_test, y_pred) * 100)) + " %")

    return model


# main function from which classifier implementation is invoked
if __name__ == "__main__":
    # using config file for source data path
    path = datasource_config.CLASSIFICATION_SOURCE_DATA_PATH

    # read the data
    df = pd.read_parquet(path)
    df1 = df[['fault_class', 'individual', 'pdmp_skew', 'pin_skew', 'po_skew', 'pdmp_variance', 'pin_variance',
            'po_variance', 'pdmp_kurtosis', 'pin_kurtosis', 'po_kurtosis', 'pdmp_pin_po_kwht',
            'pdmp_pin_kwut', 'pdmp_po_kwut', 'pin_po_kwut']]

    # call the classifier function
    classifier_model = predict_using_LR(df1)
