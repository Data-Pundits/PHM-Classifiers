# purpose: Classification of the rock drill machine faults using Random Forest Classifier using statistical features
# author: Sanjay Hegde

# importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
import time
import  datasource_config


# function for random forest classifier implementation
def predict_using_RF(data):

    # using the statistical features from the pre-processed data
    sensor_data = data

    # selecting input features and target class
    features = sensor_data.iloc[:, 1:]
    classes = sensor_data.iloc[:, 0]

    # split the train and test data
    x_train, x_test, y_train, y_test = train_test_split(features, classes, test_size=0.2)

    # normalise the data to bring all the values into single scale range
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    number_models = 2
    random_forest_model = RandomForestClassifier()

    # parameters for finding the best model that fits the data
    parameter_grid = {'n_estimators': [5,10,15,20], 'min_samples_leaf': [15, 20, 25,30],
                      'max_features': ['auto', 'sqrt'], 'bootstrap': [True, False]}

    # searching the best set of parameters for classifier
    classifier = RandomizedSearchCV(
        estimator=random_forest_model,
        param_distributions=parameter_grid,
        n_iter=number_models,
        scoring='accuracy',
        n_jobs=2,
        cv=5,
        refit=True,
        return_train_score=True)

    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_train)

    # print("Training Accuracy", accuracy_score(y_train, predictions))
    # print("Best params", classifier.best_params_)
    # print("Best score", classifier.best_score_)

    # creating Random Forest classifier object with the best parameters received from above function
    classifier = RandomForestClassifier(n_estimators = classifier.best_params_['n_estimators'],
                                        criterion='entropy',
                                        min_samples_leaf = classifier.best_params_['min_samples_leaf'],
                                        max_features='auto', bootstrap=False)

    # training the model
    st_time = time.time()
    classifier.fit(x_train, y_train)
    end_tm = time.time()
    print("Training Time : " + str(round(end_tm - st_time)) + " seconds")

    train_pred = classifier.predict(x_train)
    print("Training Accuracy : " + str((accuracy_score(y_train, train_pred) * 100)) + " %")

    # testing the model on test data
    y_pred = classifier.predict(x_test)

    # performance evaluation using different metrics
    r2_val = r2_score(y_test, y_pred)
    print("R2 value : " + str(r2_val))
    print("Testing Accuracy : " + str((accuracy_score(y_test, y_pred) * 100)) + " %")

    return classifier


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
    classifier_model = predict_using_RF(df1)

