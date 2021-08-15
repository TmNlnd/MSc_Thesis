# Load the required libraries and frameworks
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from os.path import join
from sklearn.metrics import accuracy_score as accuracy, f1_score, mean_absolute_error as mae
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from pathlib2 import Path
from tensorflow.keras import backend as K, callbacks
import tensorflow as tf
import tensorflow.keras as keras

##########################
######### 2D-CNNpred
##########################

# Define a function that will be used to determine the performance of the model
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision_pos = precision(y_true, y_pred)
    recall_pos = recall(y_true, y_pred)
    precision_neg = precision((K.ones_like(y_true) - y_true), (K.ones_like(y_pred) - K.clip(y_pred, 0, 1)))
    recall_neg = recall((K.ones_like(y_true) - y_true), (K.ones_like(y_pred) - K.clip(y_pred, 0, 1)))
    f_posit = 2 * ((precision_pos * recall_pos) / (precision_pos + recall_pos + K.epsilon()))
    f_neg   = 2 * ((precision_neg * recall_neg) / (precision_neg + recall_neg + K.epsilon()))

    return (f_posit + f_neg) / 2

# Define a function that loads the datasets from the directory
def load_data(file_fir):
    try:
        df_raw = pd.read_csv(file_fir, index_col='Date') # parse_dates=['Date'])
    except IOError:
        print("IO ERROR")
    return df_raw

# Define a function that forms a data warehouse in which the loaded datasets are stored
def costruct_data_warehouse(ROOT_PATH, file_names):
    global number_of_stocks
    global samples_in_each_stock
    global number_feature
    global order_stocks
    global insp_trainval
    global insp_train
    global insp_val
    global insp_test
    data_warehouse = {}

    for stock_file_name in file_names:

        file_dir = os.path.join(ROOT_PATH, stock_file_name)
        ## Loading Data
        try:
            df_raw = load_data(file_dir)
        except ValueError:
            print("Couldn't Read {} file".format(file_dir))

        number_of_stocks += 1

        data = df_raw.iloc[:1984, 0:83]
        df_name = data['Name'][0]           # Identify the name of the stock that is the label of the particular dataset
        order_stocks.append(df_name)        # Append the order of the stock markets for later result representation
        del data['Name']                    # Remove the 'name' column containing the name of the stock that is the label for the dataset

        # Create the target variable consisting of 0 and 1, depending on whether the market goes down or up
        target = (data['Close'][predict_day:] / data['Close'][:-predict_day].values).astype(int)
        # Slice the data and return all dates involved as input (Meaning minus last row)
        data = data[:-predict_day]          
        target.index = data.index
        
        # Cleaning:
        data = data[200:]                  # Becasue of using 200 days Moving Average as one of the features
        data = data.fillna(0)              # Fill missing values with the specified method
        data['target'] = target            # Add the target variable to the main dataset
        target = data['target']            # Adjust the target dataset to match the entries of the main dataset
        del data['target']                 # Remove the target variable from the main dataset

        number_feature = data.shape[1]         # Gives the number of columns in the array, which is 82. This is the nr of features
        samples_in_each_stock = data.shape[0]  # Gives the number of rows in the array, which is 1783, This is the nr of samples
        date = data.index
        
        # Training set
        train_data = data[:1386]                                         # Select the training data. This is roughly 77.7 % of the data. Here 1386
        train_data1 = train_data                                         # Scale the training data
        print(train_data.shape)                                          # Print the shape that the training data has
        train_target1 = target[:1386]                                    # Select the target data for the training set
        train_data = train_data1[:int(0.8 * train_data1.shape[0])]       # Use 75% of the scaled training data as training data. Here 1039 samples
        scaler = StandardScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        train_target = train_target1[:int(0.8 * train_target1.shape[0])] # Use 75% of the target data as target training data. Here 1039 samples
        
        # Validation set
        valid_data = train_data1[int(0.8 * train_data1.shape[0]) - seq_len:] # Select and scale 25% of the training data as validation data  
        valid_data = scaler.transform(valid_data)
        valid_target = train_target1[int(0.8 * train_target1.shape[0]) - seq_len:]  # Select 25 % of the target training data as target validation data
        
        # Test set 
        test_data = data[1386:]                      # Select the data for the test set. Here 397 samples
        test_data = scaler.transform(test_data)
        test_target = target[1386:]                # Select the data for the target test set. 

        # Construct the data_warehouse for the specific market containing train, validation, and test data
        data_warehouse[df_name] = [train_data, train_target, np.array(test_data), np.array(test_target), valid_data,
                                   valid_target]
        

    return data_warehouse

# Define a function that 
def cnn_data_sequence_separately(tottal_data, tottal_target, data, target, seque_len):
    for index in range(data.shape[0] - seque_len + 1):
        tottal_data.append(data[index: index + seque_len])
        tottal_target.append(target[index + seque_len - 1])

    return tottal_data, tottal_target

# Define a function that creates the syntathic images of 60x82
def cnn_data_sequence(data_warehouse, seq_len):
    global tottal_test_target
    global tottal_valid_target
    global tottal_train_target
    
    
    # Create the required lists to store the syntathic images
    tottal_train_data   = []
    tottal_train_target = []
    tottal_valid_data   = []
    tottal_valid_target = []
    tottal_test_data    = []
    tottal_test_target  = []

    for key, value in data_warehouse.items():
        tottal_train_data, tottal_train_target = cnn_data_sequence_separately(tottal_train_data, tottal_train_target,
                                                                              value[0], value[1], seq_len) # Creates 980 "images" of 60x82 and 980 targets (goes from 1108 train set to -60 = 1049)
        tottal_test_data, tottal_test_target = cnn_data_sequence_separately(tottal_test_data, tottal_test_target,
                                                                            value[2], value[3], seq_len)   # Creates 338 "images" of 60x82 and 380 targets (goes from 397 test set to -60 = 338)
        tottal_valid_data, tottal_valid_target = cnn_data_sequence_separately(tottal_valid_data, tottal_valid_target,
                                                                              value[4], value[5], seq_len) # Creates 348 "images" of 60x82 and 348 targets (goes from 338 valid set to -60 = 279)

    # Store these syntathic images as arrays
    tottal_train_data   = np.array(tottal_train_data)
    tottal_train_target = np.array(tottal_train_target)
    tottal_test_data    = np.array(tottal_test_data)
    tottal_test_target  = np.array(tottal_test_target)
    tottal_valid_data   = np.array(tottal_valid_data)
    tottal_valid_target = np.array(tottal_valid_target)

    # Reshape the arrays by adding an extra dimension     
    tottal_train_data = tottal_train_data.reshape(tottal_train_data.shape[0], tottal_train_data.shape[1],
                                                  tottal_train_data.shape[2], 1)
    tottal_test_data = tottal_test_data.reshape(tottal_test_data.shape[0], tottal_test_data.shape[1],
                                                tottal_test_data.shape[2], 1)
    tottal_valid_data = tottal_valid_data.reshape(tottal_valid_data.shape[0], tottal_valid_data.shape[1],
                                                  tottal_valid_data.shape[2], 1)

    return tottal_train_data, tottal_train_target, tottal_test_data, tottal_test_target, tottal_valid_data, tottal_valid_target

# Define a function that 
def sklearn_acc(model, test_data, test_target):
    global overall_results
    global test_pred
    global acc_results
    
    overall_results = model.predict(test_data)
    test_pred = (overall_results > 0.5).astype(int)
    acc_results = [mae(overall_results, test_target), accuracy(test_pred, test_target),
                   f1_score(test_pred, test_target, average='macro')]

    return acc_results

# Define a function that creates the Convolutional Neural Network model and runs it
def train(data_warehouse, i):
    seq_len = 60   # Stands for 60 days
    epochs  = 200  # Stands for 200 epochs
    drop    = 0.1  # Stands for 0.1 dropout 

    global cnn_train_data, cnn_train_target, cnn_test_data, cnn_test_target, cnn_valid_data, cnn_valid_target

    if i == 1:
        print('sequencing ...')
        cnn_train_data, cnn_train_target, cnn_test_data, cnn_test_target, cnn_valid_data, cnn_valid_target = cnn_data_sequence(
            data_warehouse, seq_len)

    my_file = Path(join(Base_dir,
        '2D-models/best-CNN-82-{}-{}-{}-{}-{}.h5'.format(epochs, seq_len, number_filter, drop, i)))
    filepath = join(Base_dir, '2D-models/best-CNN-82-{}-{}-{}-{}-{}.h5'.format(epochs, seq_len, number_filter, drop, i))
    if my_file.is_file():
        print('loading model')

    else:

        print(' fitting model to target')
        
        # Select the model
        model = Sequential()
        
        # layer 1   
        # Consisting of Conv with (1 x number_feature) so (1 x 82)
        # Consisting of input shape of (seq_len x number_feature) so (60 x 82) which is (d x f)
        model.add(Conv2D(number_filter[0], (1, number_feature), activation='relu', input_shape=(seq_len, number_feature, 1)))
        
        # layer 2
        # Consisting of Conv (3 x 1)       
        model.add(Conv2D(number_filter[1], (3, 1), activation='relu'))
        # Consisting of Max Poolin (2 x 1)
        model.add(MaxPool2D(pool_size=(2, 1)))

        # layer 3
        # Consisting of Conv (3 x 1)
        model.add(Conv2D(number_filter[2], (3, 1), activation='relu'))
        # Consisting of Max Pool (2 x 1)
        model.add(MaxPool2D(pool_size=(2, 1)))

        # Flatten
        model.add(Flatten())
        model.add(Dropout(drop))

        # Fully connected
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='Adam', loss='mae', metrics=['acc', f1])

        best_model = callbacks.ModelCheckpoint(filepath, monitor='val_f1', verbose=0, save_best_only=True,
                                               save_weights_only=False, mode='max', period=1, save_freq='epoch')


        model.fit(cnn_train_data, cnn_train_target, epochs=epochs, batch_size=128, verbose=1,
                        validation_data=(cnn_valid_data, cnn_valid_target), callbacks=[best_model])
    model = load_model(filepath, custom_objects={'f1': f1})
    
    model.summary()

    return model, seq_len


def cnn_data_sequence_pre_train(data, target, seque_len):
    new_data = []
    new_target = []
    for index in range(data.shape[0] - seque_len + 1):
        new_data.append(data[index: index + seque_len])
        new_target.append(target[index + seque_len - 1])

    new_data = np.array(new_data)
    new_target = np.array(new_target)

    new_data = new_data.reshape(new_data.shape[0], new_data.shape[1], new_data.shape[2], 1)

    return new_data, new_target


def prediction_f1(data_warehouse, model, seque_len, order_stocks, cnn_results_f1):
    for name in order_stocks:
        value = data_warehouse[name]
        test_data, test_target = cnn_data_sequence_pre_train(value[2], value[3], seque_len)
        cnn_results_f1.append(sklearn_acc(model, test_data, test_target)[2])

    return cnn_results_f1

def prediction_acc(data_warehouse, model, seque_len, order_stocks, cnn_results_acc):
    for name in order_stocks:
        value = data_warehouse[name]
        test_data, test_target = cnn_data_sequence_pre_train(value[2], value[3], seque_len)
        cnn_results_acc.append(sklearn_acc(model, test_data, test_target)[1])

    return cnn_results_acc

def run_cnn(data_warehouse, order_stocks):
    cnn_results_f1 = []
    cnn_results_acc = []
    iterate_no = 11                  # Setting this as 4 means 3 iterations
    for i in range(1, iterate_no):  # See the explanation above
        K.clear_session()
        print(i)
        model, seq_len = train(data_warehouse, i)
        cnn_results_acc = prediction_acc(data_warehouse, model, seq_len, order_stocks, cnn_results_acc)
        cnn_results_f1  = prediction_f1(data_warehouse, model, seq_len, order_stocks, cnn_results_f1)


    cnn_results_acc = np.array(cnn_results_acc)
    cnn_results_acc = cnn_results_acc.reshape(iterate_no - 1, len(order_stocks))
    cnn_results_acc = pd.DataFrame(cnn_results_acc, columns=order_stocks)
    cnn_results_acc = cnn_results_acc.append([cnn_results_acc.mean(), cnn_results_acc.max(), cnn_results_acc.std()], ignore_index=True) # Meaning what is saved in the CSV = 3x cnn_results.means + cnn_results.max + cnn_results.std
    print(cnn_results_acc)            # Prints all results of all markets
    print("Mean Accuracy results: ")
    print(cnn_results_acc.iloc[10])    # Prints mean results of all markets
    print("Best Accuracy results: ")
    print(cnn_results_acc.iloc[11])    # Prints best results of all markets
    
    cnn_results_f1 = np.array(cnn_results_f1)
    cnn_results_f1 = cnn_results_f1.reshape(iterate_no - 1, len(order_stocks))
    cnn_results_f1 = pd.DataFrame(cnn_results_f1, columns=order_stocks)
    cnn_results_f1 = cnn_results_f1.append([cnn_results_f1.mean(), cnn_results_f1.max(), cnn_results_f1.std()], ignore_index=True) # Meaning what is saved in the CSV = 3x cnn_results.means + cnn_results.max + cnn_results.std
    cnn_results_f1.to_csv(join(Base_dir, '2D-models/new CNN 82 results.csv'), index=False)                                    # Saves the results in a new CSV
    print(cnn_results_f1)            # Prints all results of all markets
    print("Mean F1-results: ")
    print(cnn_results_f1.iloc[10])    # Prints mean results of all markets
    print("Best F1-results: ")
    print(cnn_results_f1.iloc[11])    # Prints best results of all markets

Base_dir = ''
TRAIN_ROOT_PATH = join(Base_dir, 'Dataset')
train_file_names = os.listdir(join(Base_dir, 'Dataset'))

# if moving average = 0 then we have no moving average
seq_len = 60                      # Stands for 60 days (d)
moving_average_day = 0
number_of_stocks = 0
number_feature = 0
samples_in_each_stock = 0
number_filter = [8, 8, 8]
predict_day = 1

cnn_train_data, cnn_train_target, cnn_test_data, cnn_test_target, cnn_valid_data, cnn_valid_target= ([] for i in
                                                                                                      range(6))

print('Loading train/val data for all markets...')
order_stocks = []
data_warehouse = costruct_data_warehouse(TRAIN_ROOT_PATH, train_file_names)

print('number of markets per prediction = '), number_of_stocks

# Run the CNN2pred-model
run_cnn(data_warehouse, order_stocks)






##########################
######### S V M 
##########################

# Load the SVM package
from sklearn import svm

# Define a function that loads the datasets from the directory
def load_data(file_fir):
    try:
        df_raw = pd.read_csv(file_fir, index_col='Date') # parse_dates=['Date'])
    except IOError:
        print("IO ERROR")
    return df_raw

# Define a function that forms a data warehouse in which the loaded datasets are stored
def costruct_data_warehouse(ROOT_PATH, file_names):
    global number_of_stocks
    global samples_in_each_stock
    global number_feature
    global order_stocks
    global insp_trainval
    global insp_train
    global insp_val
    global insp_test
    data_warehouse = {}

    for stock_file_name in file_names:

        file_dir = os.path.join(ROOT_PATH, stock_file_name)
        ## Loading Data
        try:
            df_raw = load_data(file_dir)
        except ValueError:
            print("Couldn't Read {} file".format(file_dir))

        number_of_stocks += 1

        data = df_raw.iloc[:1984, 0:83]
        df_name = data['Name'][0]           # Identify the name of the stock that is the label of the particular dataset
        order_stocks.append(df_name)        # Append the order of the stock markets for later result representation
        del data['Name']                    # Remove the 'name' column containing the name of the stock that is the label for the dataset

        # Create the target variable consisting of 0 and 1, depending on whether the market goes down or up
        target = (data['Close'][predict_day:] / data['Close'][:-predict_day].values).astype(int)
        # Slice the data and return all dates involved as input (Meaning minus last row)
        data = data[:-predict_day]          
        target.index = data.index
        
        # Cleaning:
        data = data[200:]                  # Becasue of using 200 days Moving Average as one of the features
        data = data.fillna(0)              # Fill missing values with the specified method
        data['target'] = target            # Add the target variable to the main dataset
        target = data['target']            # Adjust the target dataset to match the entries of the main dataset
        del data['target']                 # Remove the target variable from the main dataset

        number_feature = data.shape[1]         # Gives the number of columns in the array, which is 82. This is the nr of features
        samples_in_each_stock = data.shape[0]  # Gives the number of rows in the array, which is 1783, This is the nr of samples
        date = data.index
        

        # Training set
        train_data = data[:1386]                                         # Select the training data. This is roughly 77.7 % of the data. Here 1386
        train_data1 = train_data                                         # Scale the training data
        print(train_data.shape)                                          # Print the shape that the training data has
        train_target1 = target[:1386]                                    # Select the target data for the training set
        train_data = train_data1[:int(0.8 * train_data1.shape[0])]       # Use 75% of the scaled training data as training data. Here 1039 samples
        scaler = StandardScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        train_data = train_data[(seq_len-1):]        
        train_target = train_target1[:int(0.8 * train_target1.shape[0])] # Use 75% of the target data as target training data. Here 1039 samples
        train_target = train_target[(seq_len-1):]
        
        # Validation set
        valid_data = train_data1[int(0.8 * train_data1.shape[0]) - seq_len:] # Select and scale 25% of the training data as validation data  
        valid_data = scaler.transform(valid_data)
        valid_data = valid_data[(seq_len-1):]
        valid_target = train_target1[int(0.8 * train_target1.shape[0]) - seq_len:]  # Select 25 % of the target training data as target validation data
        valid_target = valid_target[(seq_len-1):]
        
        # Test set 
        test_data = data[1386:]                      # Select the data for the test set. Here 397 samples
        test_data = scaler.transform(test_data)
        test_data = test_data[(seq_len-1):]
        test_target = target[1386:]                # Select the data for the target test set. 
        test_target = test_target[(seq_len-1):]
        
        # Construct the data_warehouse for the specific market containing train, validation, and test data
        data_warehouse[df_name] = [train_data, train_target, np.array(test_data), np.array(test_target), valid_data,
                                   valid_target]
        

    return data_warehouse

# Define a function that 
def svm_data_sequence_separately(tottal_data, tottal_target, data, target, seque_len):
    for index in range(data.shape[0] - seque_len + 1):
        tottal_data.append(data[index: index + seque_len])
        tottal_target.append(target[index + seque_len - 1])

    return tottal_data, tottal_target


def svm_data_sequence(data_warehouse):
    global tottal_test_target
    
    # Create the required lists to store the syntathic images
    tottal_train_data   = []
    tottal_train_target = []
    tottal_valid_data   = []
    tottal_valid_target = []
    tottal_test_data    = []
    tottal_test_target  = []

    for key, value in data_warehouse.items():
        tottal_train_data, tottal_train_target = value[0], value[1] # Creates 980 "images" of 60x82 and 980 targets
        tottal_test_data, tottal_test_target   = value[2], value[3] # Creates 338 "images" of 60x82 and 380 targets
        tottal_valid_data, tottal_valid_target = value[4], value[5] # Creates 348 "images" of 60x82 and 348 targets

    # Store these syntathic images as arrays
    tottal_train_data   = np.array(tottal_train_data)
    tottal_train_target = np.array(tottal_train_target)
    tottal_test_data    = np.array(tottal_test_data)
    tottal_test_target  = np.array(tottal_test_target)
    tottal_valid_data   = np.array(tottal_valid_data)
    tottal_valid_target = np.array(tottal_valid_target)

    return tottal_train_data, tottal_train_target, tottal_test_data, tottal_test_target, tottal_valid_data, tottal_valid_target

# Define a function that 
def sklearn_acc(model, test_data, test_target):
    global overall_results
    global test_pred
    global acc_results
    
    overall_results = model.predict(test_data)
    test_pred = (overall_results > 0.5).astype(int)
    acc_results = [mae(overall_results, test_target), accuracy(test_pred, test_target),
                   f1_score(test_pred, test_target, average='macro')]

    return acc_results

# Define a function that creates the Convolutional Neural Network model and runs it
def train(data_warehouse):

    global svm_train_data, svm_train_target, svm_test_data, svm_test_target, svm_valid_data, svm_valid_target 

    print(' fitting SVM-model to target')
    # Select the model
    svm_train_data, svm_train_target, svm_test_data, svm_test_target, svm_valid_data, svm_valid_target = svm_data_sequence(data_warehouse)
    
    model = svm.SVC(kernel='linear') # Linear Kernel
        
    model.fit(svm_train_data, svm_train_target)
    
    model.predict(svm_valid_data)
    y_true = svm_valid_target
    y_pred = model.predict(svm_valid_data)
    SVM_f1_score_valid = f1_score(y_true, y_pred, average='macro')
   
    return model

def prediction_acc(data_warehouse, model, order_stocks, svm_results_acc):
    global test_data, test_target
    
    for name in order_stocks:
        value = data_warehouse[name]
        test_data, test_target = value[2], value[3]
        
        svm_results_acc.append(sklearn_acc(model, test_data, test_target)[1])

    return svm_results_acc

def prediction_f1(data_warehouse, model, order_stocks, svm_results_f1):
    global test_data, test_target
    
    for name in order_stocks:
        value = data_warehouse[name]
        test_data, test_target = value[2], value[3]
        
        svm_results_f1.append(sklearn_acc(model, test_data, test_target)[2])

    return svm_results_f1

def run_svm(data_warehouse, order_stocks):
    svm_results_f1  = []
    svm_results_acc = []
    iterate_no = 4                  # Setting this as 4 means 3 iterations
    for i in range(1, iterate_no):  # See the explanation above
        K.clear_session()
        print(i)
        model = train(data_warehouse)
        svm_results_f1  = prediction_f1(data_warehouse, model, order_stocks, svm_results_f1)
        svm_results_acc = prediction_acc(data_warehouse, model, order_stocks, svm_results_acc)

        
    svm_results_acc = np.array(svm_results_acc)
    svm_results_acc = svm_results_acc.reshape(iterate_no - 1, len(order_stocks))
    svm_results_acc = pd.DataFrame(svm_results_acc, columns=order_stocks)
    svm_results_acc = svm_results_acc.append([svm_results_acc.mean(), svm_results_acc.max(), svm_results_acc.std()], ignore_index=True) # Meaning what is saved in the CSV = 3x svm_results.means + svm_results.max + svm_results.std
    print("Accuracy results: ")
    print(svm_results_acc.iloc[3])    # Prints mean results of all markets
    
    svm_results_f1 = np.array(svm_results_f1)
    svm_results_f1 = svm_results_f1.reshape(iterate_no - 1, len(order_stocks))
    svm_results_f1 = pd.DataFrame(svm_results_f1, columns=order_stocks)
    svm_results_f1 = svm_results_f1.append([svm_results_f1.mean(), svm_results_f1.max(), svm_results_f1.std()], ignore_index=True) # Meaning what is saved in the CSV = 3x svm_results.means + svm_results.max + svm_results.std
    svm_results_f1.to_csv(join(Base_dir, '2D-models/new SVM 82 results.csv'), index=False)                                    # Saves the results in a new CSV
    print("F1-results: ")
    print(svm_results_f1.iloc[3])    # Prints mean results of all markets

Base_dir = ''
TRAIN_ROOT_PATH = join(Base_dir, 'Dataset')
train_file_names = os.listdir(join(Base_dir, 'Dataset'))

# if moving average = 0 then we have no moving average
seq_len = 60                      # Stands for 60 days (d)
moving_average_day = 0
number_of_stocks = 0
predict_day = 1

svm_train_data, svm_train_target, svm_test_data, svm_test_target, svm_valid_data, svm_valid_target= ([] for i in range(6))

print('Loading train/val data for all markets...')
order_stocks = []
data_warehouse = costruct_data_warehouse(TRAIN_ROOT_PATH, train_file_names)

print('number of markets per prediction = '), number_of_stocks

run_svm(data_warehouse, order_stocks)














##########################
######### A N N  
##########################

# Define a function that will be used to determine the effectiveness of the model
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision_pos = precision(y_true, y_pred)
    recall_pos = recall(y_true, y_pred)
    precision_neg = precision((K.ones_like(y_true) - y_true), (K.ones_like(y_pred) - K.clip(y_pred, 0, 1)))
    recall_neg = recall((K.ones_like(y_true) - y_true), (K.ones_like(y_pred) - K.clip(y_pred, 0, 1)))
    f_posit = 2 * ((precision_pos * recall_pos) / (precision_pos + recall_pos + K.epsilon()))
    f_neg   = 2 * ((precision_neg * recall_neg) / (precision_neg + recall_neg + K.epsilon()))

    return (f_posit + f_neg) / 2

# Define a function that loads the datasets from the directory
def load_data(file_fir):
    try:
        df_raw = pd.read_csv(file_fir, index_col='Date') # parse_dates=['Date'])
    except IOError:
        print("IO ERROR")
    return df_raw

# Define a function that forms a data warehouse in which the loaded datasets are stored
def costruct_data_warehouse(ROOT_PATH, file_names):
    global number_of_stocks
    global samples_in_each_stock
    global number_feature
    global order_stocks
    global insp_trainval
    global insp_train
    global insp_val
    global insp_test
    data_warehouse = {}

    for stock_file_name in file_names:

        file_dir = os.path.join(ROOT_PATH, stock_file_name)
        ## Loading Data
        try:
            df_raw = load_data(file_dir)
        except ValueError:
            print("Couldn't Read {} file".format(file_dir))

        number_of_stocks += 1

        data = df_raw.iloc[:1984, 0:83]
        df_name = data['Name'][0]           # Identify the name of the stock that is the label of the particular dataset
        order_stocks.append(df_name)        # Append the order of the stock markets for later result representation
        del data['Name']                    # Remove the 'name' column containing the name of the stock that is the label for the dataset

        # Create the target variable consisting of 0 and 1, depending on whether the market goes down or up
        target = (data['Close'][predict_day:] / data['Close'][:-predict_day].values).astype(int)
        # Slice the data and return all dates involved as input (Meaning minus last row)
        data = data[:-predict_day]          
        target.index = data.index
        
        # Cleaning:
        data = data[200:]                  # Becasue of using 200 days Moving Average as one of the features
        data = data.fillna(0)              # Fill missing values with the specified method
        data['target'] = target            # Add the target variable to the main dataset
        target = data['target']            # Adjust the target dataset to match the entries of the main dataset
        del data['target']                 # Remove the target variable from the main dataset

        number_feature = data.shape[1]         # Gives the number of columns in the array, which is 82. This is the nr of features
        samples_in_each_stock = data.shape[0]  # Gives the number of rows in the array, which is 1783, This is the nr of samples
        
        # Training set
        train_data = data[:1386]                                         # Select the training data. This is roughly 77.7 % of the data. Here 1386
        train_data1 = train_data                                         # Scale the training data
        print(train_data.shape)                                          # Print the shape that the training data has
        train_target1 = target[:1386]                                    # Select the target data for the training set
        train_data = train_data1[:int(0.8 * train_data1.shape[0])]       # Use 75% of the scaled training data as training data. Here 1039 samples
        scaler = StandardScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        train_data = train_data[(seq_len-1):]        
        train_target = train_target1[:int(0.8 * train_target1.shape[0])] # Use 75% of the target data as target training data. Here 1039 samples
        train_target = train_target[(seq_len-1):]
        
        # Validation set
        valid_data = train_data1[int(0.8 * train_data1.shape[0]) - seq_len:] # Select and scale 25% of the training data as validation data  
        valid_data = scaler.transform(valid_data)
        valid_data = valid_data[(seq_len-1):]
        valid_target = train_target1[int(0.8 * train_target1.shape[0]) - seq_len:]  # Select 25 % of the target training data as target validation data
        valid_target = valid_target[(seq_len-1):]
        
        # Test set 
        test_data = data[1386:]                      # Select the data for the test set. Here 397 samples
        test_data = scaler.transform(test_data)
        test_data = test_data[(seq_len-1):]
        test_target = target[1386:]                # Select the data for the target test set. 
        test_target = test_target[(seq_len-1):]
        
        # Construct the data_warehouse for the specific market containing train, validation, and test data
        data_warehouse[df_name] = [train_data, train_target, np.array(test_data), np.array(test_target), valid_data,
                                   valid_target]
        

    return data_warehouse

def ann_data_sequence(data_warehouse):
    global tottal_test_target
    global tottal_test_data
    
    # Create the required lists to store the syntathic images
    tottal_train_data   = []
    tottal_train_target = []
    tottal_valid_data   = []
    tottal_valid_target = []
    tottal_test_data    = []
    tottal_test_target  = []

    for key, value in data_warehouse.items():
        tottal_train_data, tottal_train_target = value[0], value[1] # Creates 980 "images" of 60x82 and 980 targets
        tottal_test_data, tottal_test_target   = value[2], value[3] # Creates 338 "images" of 60x82 and 380 targets
        tottal_valid_data, tottal_valid_target = value[4], value[5] # Creates 348 "images" of 60x82 and 348 targets

    # Store these syntathic images as arrays
    tottal_train_data   = np.array(tottal_train_data)
    tottal_train_target = np.array(tottal_train_target)
    tottal_test_data    = np.array(tottal_test_data)
    tottal_test_target  = np.array(tottal_test_target)
    tottal_valid_data   = np.array(tottal_valid_data)
    tottal_valid_target = np.array(tottal_valid_target)

    return tottal_train_data, tottal_train_target, tottal_test_data, tottal_test_target, tottal_valid_data, tottal_valid_target

# Define a function that 
def sklearn_acc(model, test_data, test_target):
    global overall_results
    global test_pred
    global acc_results
    
    overall_results = model.predict(test_data)
    test_pred = (overall_results > 0.5).astype(int)
    acc_results = [mae(overall_results, test_target), accuracy(test_pred, test_target),
                   f1_score(test_pred, test_target, average='macro')]

    return acc_results

# Define a function that creates the Convolutional Neural Network model and runs it
def train(data_warehouse, i):
    seq_len = 60   # Stands for 60 days
    epochs  = 200  # Stands for 200 epochs
    drop    = 0.1  # Stands for 0.1 dropout 

    global ann_train_data, ann_train_target, ann_test_data, ann_test_target, ann_valid_data, ann_valid_target

    if i == 1:
        print('sequencing ...')
        ann_train_data, ann_train_target, ann_test_data, ann_test_target, ann_valid_data, ann_valid_target = ann_data_sequence(
            data_warehouse)

    my_file = Path(join(Base_dir,
        '2D-models/best-ANN-82-{}-{}.h5'.format(epochs, i)))
    filepath = join(Base_dir, '2D-models/best-ANN-82-{}-{}.h5'.format(epochs, i))
    if my_file.is_file():
        print('loading model')

    else:

        print(' fitting model to target')
        # Select the model
        model = Sequential()
        
        # input layer  
        model.add(Dense(units=82, activation='relu'))
        
        # hidden layer
        model.add(Dense(30, activation='tanh'))

        # output layer
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='Adam', loss='mae', metrics=['acc', f1])    

        best_model = callbacks.ModelCheckpoint(filepath, monitor='val_f1', verbose=0, save_best_only=True,
                                               save_weights_only=False, mode='max', period=1, save_freq='epoch')
        
        model.fit(ann_train_data, ann_train_target, epochs=epochs, batch_size=128, verbose=1,
                        validation_data=(ann_valid_data, ann_valid_target), callbacks=[best_model])
    
    model = load_model(filepath, custom_objects={'f1': f1})
    
    ann_model = model.summary()

    return model

def prediction_acc(data_warehouse, model, order_stocks, ann_results_acc):
    for name in order_stocks:
        value = data_warehouse[name]
        test_data, test_target = value[2], value[3]

        ann_results_acc.append(sklearn_acc(model, test_data, test_target)[1])

    return ann_results_acc

def prediction_f1(data_warehouse, model, order_stocks, ann_results_f1):
    for name in order_stocks:
        value = data_warehouse[name]
        test_data, test_target = value[2], value[3]

        ann_results_f1.append(sklearn_acc(model, test_data, test_target)[2])

    return ann_results_f1


def run_ann(data_warehouse, order_stocks):
    ann_results_f1  = []
    ann_results_acc = []
    iterate_no = 11                  # Setting this as 4 means 3 iterations
    for i in range(1, iterate_no):  # See the explanation above
        K.clear_session()
        print(i)
        model = train(data_warehouse, i)
        ann_results_acc = prediction_acc(data_warehouse, model, order_stocks, ann_results_acc)
        ann_results_f1  = prediction_f1(data_warehouse, model, order_stocks, ann_results_f1)

    ann_results_acc = np.array(ann_results_acc)
    ann_results_acc = ann_results_acc.reshape(iterate_no - 1, len(order_stocks))
    ann_results_acc = pd.DataFrame(ann_results_acc, columns=order_stocks)
    ann_results_acc = ann_results_acc.append([ann_results_acc.mean(), ann_results_acc.max(), ann_results_acc.std()], ignore_index=True) # Meaning what is saved in the CSV = 3x cnn_results.means + cnn_results.max + cnn_results.std
    print(ann_results_acc)            # Prints all results of all markets
    print("Mean Accuracy results: ")
    print(ann_results_acc.iloc[10])    # Prints mean results of all markets
    print("Best Accuracy results: ")
    print(ann_results_acc.iloc[11])    # Prints best results of all markets
        
    ann_results_f1 = np.array(ann_results_f1)
    ann_results_f1 = ann_results_f1.reshape(iterate_no - 1, len(order_stocks))
    ann_results_f1 = pd.DataFrame(ann_results_f1, columns=order_stocks)
    ann_results_f1 = ann_results_f1.append([ann_results_f1.mean(), ann_results_f1.max(), ann_results_f1.std()], ignore_index=True) # Meaning what is saved in the CSV = 3x cnn_results.means + cnn_results.max + cnn_results.std
    ann_results_f1.to_csv(join(Base_dir, '2D-models/new ANN 82 results.csv'), index=False)                                    # Saves the results in a new CSV
    print(ann_results_f1)            # Prints all results of all markets
    print("Mean F1-results: ")
    print(ann_results_f1.iloc[10])    # Prints mean results of all markets
    print("Best F1-results: ")
    print(ann_results_f1.iloc[11])    # Prints best results of all markets

Base_dir = ''
TRAIN_ROOT_PATH = join(Base_dir, 'Dataset')
train_file_names = os.listdir(join(Base_dir, 'Dataset'))

# if moving average = 0 then we have no moving average
seq_len = 60                      # Stands for 60 days (d)
moving_average_day = 0
number_of_stocks = 0
number_feature = 0
samples_in_each_stock = 0
number_filter = [8, 8, 8]
predict_day = 1

ann_train_data, ann_train_target, ann_test_data, ann_test_target, ann_valid_data, ann_valid_target= ([] for i in
                                                                                                      range(6))

print('Loading train/val data for all markets...')
order_stocks = []
data_warehouse = costruct_data_warehouse(TRAIN_ROOT_PATH, train_file_names)

print('number of markets per prediction = '), number_of_stocks

run_ann(data_warehouse, order_stocks)













###########################
### Baseline Classifier ###
###########################

# Load the required packages
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

# Load in one dataset, run the code, load in a new one, repeat
df_raw = pd.read_csv('DJI_TN.csv', index_col='Date') 
df_raw = pd.read_csv('NASDAQ_TN.csv', index_col='Date') 
df_raw = pd.read_csv('NYSE_TN.csv', index_col='Date') 
df_raw = pd.read_csv('RUSSELL_TN.csv', index_col='Date') 
df_raw = pd.read_csv('SP_TN.csv', index_col='Date') 

# Input required values
seq_len = 60                      # Stands for 60 days (d)
moving_average_day = 0
number_of_stocks = 0
predict_day = 1

# Data prep
data = df_raw.iloc[:1984, 0:83]

df_name = data['Name'][0]           # Identify the name of the stock that is the label of the particular dataset
del data['Name']                    # Remove the 'name' column containing the name of the stock that is the label for the dataset
# Create the target variable consisting of 0 and 1, depending on whether the market goes down or up
target = (data['Close'][predict_day:] / data['Close'][:-predict_day].values).astype(int)

# Slice the data and return all dates involved as input (Meaning minus last row)
data = data[:-predict_day]          
target.index = data.index

# Cleaning:
data = data[200:]                  # Becasue of using 200 days Moving Average as one of the features        
        
data = data.fillna(0)              # Fill missing values with the specified method
data['target'] = target            # Add the target variable to the main dataset
target = data['target']            # Adjust the target dataset to match the entries of the main dataset
del data['target']                 # Remove the target variable from the main dataset
        
number_feature = data.shape[1]         # Gives the number of columns in the array, which is 82. This is the nr of features
samples_in_each_stock = data.shape[0]  # Gives the number of rows in the array, which is 1783, This is the nr of samples        
date = data.index        

# Training set
train_data = data[:1386]                                         # Select the training data. This is roughly 77.7 % of the data. Here 1386
train_data1 = train_data                                         # Scale the training data
print(train_data.shape)                                          # Print the shape that the training data has
train_target1 = target[:1386]                                    # Select the target data for the training set
train_data = train_data1[:int(0.8 * train_data1.shape[0])]       # Use 75% of the scaled training data as training data. Here 1039 samples
scaler = StandardScaler().fit(train_data)
train_data = scaler.transform(train_data)
train_data = train_data[(seq_len-1):]
train_target = train_target1[:int(0.8 * train_target1.shape[0])] # Use 75% of the target data as target training data. Here 1039 samples
train_target = train_target[(seq_len-1):]
        
# Validation set
valid_data = train_data1[int(0.8 * train_data1.shape[0]) - seq_len:] # Select and scale 25% of the training data as validation data  
valid_data = scaler.transform(valid_data)
valid_data = valid_data[(seq_len-1):]
valid_target = train_target1[int(0.8 * train_target1.shape[0]) - seq_len:]  # Select 25 % of the target training data as target validation data
valid_target = valid_target[(seq_len-1):]
        
# Test set 
test_data = data[1386:]                      # Select the data for the test set. Here 397 samples
test_data = scaler.transform(test_data)
test_data = test_data[(seq_len-1):]
test_target = target[1386:]                # Select the data for the target test set. 
test_target = test_target[(seq_len-1):]
        
### Classifier 

# Create the model and fit it to the training data
model = DummyClassifier(strategy="most_frequent")
model.fit(train_data, train_target)

# Prediction
model_prediction = model.predict(test_target)

# Results
accuracy(test_target, model_prediction)*100
f1_score(test_target, model_prediction, average='macro')*100

# Confirm in the report overview
classification_report(test_target, model_prediction, digits=4)










