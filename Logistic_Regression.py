import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time


def predict_using_LR(data):

    pdmp_df = pd.DataFrame(data.pdmp.tolist())
    pin_df = pd.DataFrame(data.pin.tolist())
    po_df = pd.DataFrame(data.po.tolist())
    intr_df = pd.merge(pdmp_df, pin_df, left_index=True, right_index=True)
    intr2_df = pd.merge(intr_df, po_df, left_index=True, right_index=True)
    final_df = pd.merge(data, intr2_df, left_index=True, right_index=True).drop(["po", "pdmp", "pin"], axis=1)
    sensor_data = final_df.add_prefix('sensor_')

    features = sensor_data.iloc[:, 1:]
    target = sensor_data.iloc[:, 0]

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # define the multinomial logistic regression model
    # model = LogisticRegression(multi_class='ovr', solver='liblinear')

    model = LogisticRegressionCV(class_weight='balanced', multi_class='multinomial', solver='lbfgs')

    # training
    st_time = time.time()
    model.fit(x_train, y_train)
    end_tm = time.time()
    print("Training Time : " + str(round(end_tm - st_time)) + " seconds")

    train_pred = model.predict(x_train)
    print("Training Accuracy : " + str((accuracy_score(y_train, train_pred) * 100)) + " %")

    # testing
    y_pred = model.predict(x_test)

    validation_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(validation_df)

    r2_val = r2_score(y_test, y_pred)
    print("R2 value : " + str(r2_val))
    print("Testing Accuracy : " + str((accuracy_score(y_test, y_pred) * 100)) + " %")

    return model


if __name__ == "__main__":
    output = "/Users/Sanjay.Hegde/Documents/UOB_MSc_DS/SEM 2/Group Project/Rock drill fault detection/output/"

    df = pd.read_parquet(output)
    df.drop('index', axis=1, inplace=True)

    classifier_model = predict_using_LR(df)
