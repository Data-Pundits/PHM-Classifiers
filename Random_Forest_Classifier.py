import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, accuracy_score
# import joblib
import time


def predict_using_RF(data, sensor_name, all_sensor_flag):
    start_time = time.time()

    if all_sensor_flag:
        print("All sensor calculation")
        pdmp_df = pd.DataFrame(data.pdmp.tolist())
        pin_df = pd.DataFrame(data.pin.tolist())
        po_df = pd.DataFrame(data.po.tolist())
        intr_df = pd.merge(pdmp_df, pin_df, left_index=True, right_index=True)
        intr2_df = pd.merge(intr_df, po_df, left_index=True, right_index=True)
        final_df = pd.merge(data, intr2_df, left_index=True, right_index=True).drop(["po", "pdmp", "pin"], axis=1)
        sensor_data = final_df.add_prefix(sensor + '_')
        number_of_trees, min_leaf = 700, 150

    else:
        print(sensor_name + " sensor calculation")
        sensor_df = data[["fault_class", "individual", sensor_name]]
        split_df = pd.DataFrame(sensor_df[sensor_name].tolist())
        sensor_data = pd.merge(sensor_df, split_df, left_index=True, right_index=True).drop(sensor_name, axis=1)
        sensor_data = sensor_data.add_prefix(sensor + '_')
        number_of_trees, min_leaf = 100, 30

    features = sensor_data.iloc[:, 1:]
    classes = sensor_data.iloc[:, 0]

    # print("Features: " + str(features.shape))
    # print("Classes: " + str(classes.shape))

    x_train, x_test, y_train, y_test = train_test_split(features, classes, test_size=0.30)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # parameter_grid = {'n_estimators': [200, 400, 500, 600, 700], 'min_samples_leaf': [55, 85, 100],
    #                   'max_features': ['auto', 'sqrt'], 'bootstrap': [True, False]}
    #
    # number_models = 2
    # random_forest_model = RandomForestClassifier()
    #
    # classifier = RandomizedSearchCV(
    #     estimator=random_forest_model,
    #     param_distributions=parameter_grid,
    #     n_iter=number_models,
    #     scoring='accuracy',
    #     n_jobs=2,
    #     cv=10,
    #     refit=True,
    #     return_train_score=True)

    # classifier.fit(x_train, y_train)
    # predictions = classifier.predict(x_train)
    #
    # print("Training Accuracy", accuracy_score(y_train, predictions))
    # print("Best params", classifier.best_params_)
    # print("Best score", classifier.best_score_)

    classifier = RandomForestClassifier(n_estimators=number_of_trees,
                                        criterion='entropy',
                                        min_samples_leaf=min_leaf,
                                        max_features='sqrt', bootstrap=True)

    # training
    st_time = time.time()
    classifier.fit(x_train, y_train)
    end_tm = time.time()
    print("Training Time : " + str(round(end_tm - st_time)) + " seconds")

    train_pred = classifier.predict(x_train)
    print("Training Accuracy : " + str((accuracy_score(y_train, train_pred) * 100)) + " %")

    # testing
    y_pred = classifier.predict(x_test)

    end_time = time.time()
    print("Processing Time : " + str(round(end_time - start_time)) + " seconds")

    validation_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(validation_df)

    r2_val = r2_score(y_test, y_pred)
    print("R2 value : " + str(r2_val))
    print("Testing Accuracy : " + str((accuracy_score(y_test, y_pred) * 100)) + " %")

    return classifier


if __name__ == "__main__":
    output = "/Users/Sanjay.Hegde/Documents/UOB_MSc_DS/SEM 2/Group Project/Rock drill fault detection/output/"

    df = pd.read_parquet(output)
    df.drop('index', axis=1, inplace=True)

    sensor = "pdmp"
    sensor_flag = True
    classifier_model = predict_using_RF(df, sensor, sensor_flag)

    # joblib.dump(classifier_model, 'randomforestmodel_'+sensor + '.pkl')
