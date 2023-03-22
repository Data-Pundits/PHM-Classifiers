import pandas as pd
import matplotlib.pyplot as plt

training = "/Users/Sanjay.Hegde/Documents/UOB_MSc_DS/Group Project/Rock drill fault detection/Data_Challenge_PHM2022_training_data/"
output = "/Users/Sanjay.Hegde/Documents/UOB_MSc_DS/Group Project/Rock drill fault detection/output/"
stage = "/Users/Sanjay.Hegde/Documents/UOB_MSc_DS/Group Project/Rock drill fault detection/stage/"

def plot_the_pressure(sensor):
    sensor_lo = sensor.lower()
    path = training + "data_"+ sensor_lo + "1.csv"

    df = pd.read_fwf(path, header=None)
    df = df[0].str.split(',', expand=True)

    class_2 = df.loc[df[0]=='2'].iloc[0:1,1:].values[0].tolist()
    class_2 = [eval(i) for i in class_2 if i != None]

    class_5 = df.loc[df[0]=='5'].iloc[0:1,1:].values[0].tolist()
    class_5 = [eval(i) for i in class_5 if i != None]

    class_1 = df.loc[df[0]=='1'].iloc[0:1,1:].values[0].tolist()
    class_1 = [eval(i) for i in class_1 if i != None]

    class_11 = df.loc[df[0]=='11'].iloc[0:1,1:].values[0].tolist()
    class_11 = [eval(i) for i in class_11 if i != None]

    plt.figure(figsize=(20,12))
    plt.plot(class_2,'-', label ='Thick drill steel')
    plt.plot(class_5, '-', label ='Damaged accumulator')
    plt.plot(class_1, '*', label ='No Fault')
    plt.plot(class_11, '-', label ='Charge level in accumulator is low')

    plt.xlabel("Value count", fontsize = 16)
    plt.ylabel("Pressure Signal", fontsize = 16)
    plt.legend(loc=1, prop={'size': 12})
    plt.title('Pressure values at different faults for individual 1 at ' + sensor + ' sensor', fontsize = 16)
    plt.show()


def fault_plot(fault):
    fault_class = str(fault)

    df = pd.read_fwf(training+"/data_pdmp1.csv", header=None)
    pdmp_df = df[0].str.split(',', expand=True)

    df = pd.read_fwf(training+ "/data_pin1.csv", header=None)
    pin_df = df[0].str.split(',', expand=True)

    df = pd.read_fwf(training + "/data_po1.csv", header=None)
    po_df = df[0].str.split(',', expand=True)

    class_2 = pdmp_df.loc[pdmp_df[0] == fault_class].iloc[0:1, 1:].values[0].tolist()
    pdmp = [eval(i) for i in class_2 if i != None]

    class_2 = pin_df.loc[pin_df[0] == fault_class].iloc[0:1, 1:].values[0].tolist()
    pin = [eval(i) for i in class_2 if i != None]

    class_2 = po_df.loc[po_df[0] == fault_class].iloc[0:1, 1:].values[0].tolist()
    po = [eval(i) for i in class_2 if i != None]

    plt.figure(figsize=(20, 12))
    plt.plot(pdmp, '-', label='PDMP')
    plt.plot(pin, '-', label='PIN')
    plt.plot(po, '*', label='PO')

    plt.xlabel("Value count", fontsize=16)
    plt.ylabel("Pressure Signal", fontsize=16)
    plt.legend(loc=1, prop={'size': 12})
    plt.title('Pressure values for each sensor for a particular fault in the same cycle', fontsize=16)
    plt.show()


def visualize(sensor, fault):

    plot_the_pressure(sensor)
    fault_plot(fault)

def EDA(path):

    df = pd.read_parquet(path)
    print("\n")
    print(df.info())
    print("Shape of the dataset: ",df.shape)
    print("\nBasic Stats:\n", df.describe())
    print("\n")
    print("Unique Fault_class:\n",df['fault_class'].unique())
    print("\n")
    print("Unique individuals:\n",df['individual'].unique())

if __name__ == "__main__":
    # visualize("PO", 2)
    EDA(stage)

