import pandas as pd
import numpy as np
import math
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def row_ln(row):
    gmean = stats.gmean(np.array(row))
    func = lambda x: math.log(x / gmean, math.e)
    new_row = row.apply(func)
    return new_row

def CLR(x):
    percent_x = (x.T / x.sum(axis=1)).T
    nomalized_x = percent_x.apply(row_ln, axis=1)
    return nomalized_x

def preprocess_data(x_train, x, normalize_method="CLR", scaling=True):
    features1 = ['SiO2(%)', 'Al2O3(%)', 'Fe2O3(%)', 'MgO(%)', 'CaO(%)', 'Na2O(%)', 'K2O(%)', 'MnO(%)', 'TiO2(%)','P2O5(%)']
    features2 = ['Cr(ppm)', 'Ni(ppm)', 'Th(ppm)', 'U(ppm)']
    features3 = ['Mn/Fe', 'Cr/Fe', 'Ni/Fe', 'U/Th', '⅀REY', 'Y/Ho', '(La/Sm)SN', '(La/Yb)SN', '(Sm/Yb)SN', '(Eu/Sm)SN', '(Ce/Ce*)SN', '(Pr/Pr*)SN',
                 '(Eu/Eu*)SN', '(Gd/Gd*)SN',  '(Lu/Lu*)SN', '(La/Lu)SN', '(Nd/Yb)SN','(Pr/Yb)SN', '(Lu/Yb)SN']
    if normalize_method == "CLR":
        pro_x_features1 = CLR(x[features1])
        pro_x_train_features1 = CLR(x_train[features1])
        pro_x_features2 = CLR(x[features2])
        pro_x_train_features2 = CLR(x_train[features2])
        pro_x_features3 = CLR(x[features3])
        pro_x_train_features3 = CLR(x_train[features3])

    # Combine processed major and trace elements
    pro_x = pd.concat([pro_x_features1, pro_x_features2, pro_x_features3], axis=1)
    pro_x_train = pd.concat([pro_x_train_features1, pro_x_train_features2, pro_x_train_features3], axis=1)

    if scaling:
        scaler = preprocessing.StandardScaler().fit(pro_x_train)
        X = scaler.transform(pro_x)
    else:
        X = pro_x

    return X

def prepare_data(file_path, target_column):
    data = pd.read_excel(file_path)

    # Defining Input Elements
    all_elements = ['SiO2(%)', 'Al2O3(%)', 'Fe2O3(%)', 'MgO(%)', 'CaO(%)', 'Na2O(%)', 'K2O(%)', 'MnO(%)', 'TiO2(%)',
                    'P2O5(%)', 'Cr(ppm)', 'Ni(ppm)', 'Th(ppm)', 'U(ppm)','Mn/Fe', 'Cr/Fe', 'Ni/Fe', 'U/Th', '⅀REY', 'Y/Ho', '(La/Sm)SN', '(La/Yb)SN',
                    '(Sm/Yb)SN', '(Eu/Sm)SN', '(Ce/Ce*)SN', '(Pr/Pr*)SN', '(Eu/Eu*)SN', '(Gd/Gd*)SN','(Lu/Lu*)SN', '(La/Lu)SN', '(Nd/Yb)SN', '(Pr/Yb)SN', '(Lu/Yb)SN']
    selected_features = ['(La/Sm)SN', '(La/Yb)SN','Mn/Fe','Ni(ppm)','(Eu/Eu*)SN','Cr/Fe','(Pr/Pr*)SN', '(Ce/Ce*)SN','(Gd/Gd*)SN']

    # Marking the external_testing set、Prediction set、Training set、Testing set
    data.loc[
        data[target_column].isin(['Porpoise Cove', 'Nanfen', 'Griquatown','Gunflint']), 'Set'] = 'External testing set'
    data.loc[
        data[target_column].isin(['Unknown']), 'Set'] = 'Prediction set'
    def label_row(value):
        if value in ['Algoma','Porpoise Cove', 'Nanfen']:
            return 0
        elif value in ['Superior','Griquatown','Gunflint']:
            return 1
        else:
            return 2

    data['Label'] = data[target_column].apply(label_row)
    other_data = data[(data['Iron Formation'] == 'Algoma') | (data['Iron Formation'] == 'Superior')]

    # Perform training and test set partitioning to maintain class balance
    train_subset_data, test_subset_data = train_test_split(
        other_data,
        test_size=0.1,
        stratify=other_data['Label'],
        random_state=19
    )
    train_subset_data['Set'] = 'Training set'
    test_subset_data['Set'] = 'Testing set'

    # Extract feature data for training, testing, external_testing and prediction sets
    X_train_subset_raw = train_subset_data[all_elements]
    X_test_subset_raw = test_subset_data[all_elements]
    X_external_testing_raw = data[data['Set'] == 'External testing set'][all_elements]
    X_predict_raw = data[data['Set'] == 'Prediction set'][all_elements]
    all_X1 = pd.concat([pd.DataFrame(X_train_subset_raw), pd.DataFrame(X_test_subset_raw)], axis=0).reset_index(drop=True)

    # Normalize selected columns
    pro_x_train_subset = preprocess_data(all_X1, X_train_subset_raw, normalize_method="CLR")
    pro_x_test_subset = preprocess_data(all_X1, X_test_subset_raw, normalize_method="CLR")
    pro_x_external_testing = preprocess_data(all_X1, X_external_testing_raw, normalize_method="CLR")
    pro_x_predict = preprocess_data(all_X1, X_predict_raw, normalize_method="CLR")

    # Normalized data added to raw data frame
    train_subset_data[[col + '_normd' for col in all_elements]] = pro_x_train_subset
    test_subset_data[[col + '_normd' for col in all_elements]] = pro_x_test_subset
    data.loc[data['Set'] == 'External testing set', [col + '_normd' for col in all_elements]] = pro_x_external_testing
    data.loc[data['Set'] == 'Prediction set', [col + '_normd' for col in all_elements]] = pro_x_predict

    # Original data added to final data
    train_subset_data[[col + '_raw' for col in all_elements]] = X_train_subset_raw
    test_subset_data[[col + '_raw' for col in all_elements]] = X_test_subset_raw
    data.loc[data['Set'] == 'External testing set', [col + '_raw' for col in all_elements]] = X_external_testing_raw
    data.loc[data['Set'] == 'Prediction set', [col + '_raw' for col in all_elements]] = X_predict_raw

    # Combining data from  training, testing, external_testing and prediction sets
    final_data = pd.concat([
        train_subset_data,
        test_subset_data,
        data[data['Set'] == 'External testing set'],
        data[data['Set'] == 'Prediction set']
    ]).reset_index(drop=True)
    final_data[[col + '_raw' for col in all_elements]] = pd.concat(
        [X_train_subset_raw, X_test_subset_raw, X_external_testing_raw, X_predict_raw]).reset_index(drop=True)
    info_list = list(data.columns[:7])
    final_data = final_data[
        info_list + [col + '_raw' for col in all_elements] + [col + '_normd' for col in all_elements] +
        ['Label'] + ['Set']]

    # Convert to DataFrame and specify column names
    pro_x_train_subset = pd.DataFrame(pro_x_train_subset, columns=all_elements)
    pro_x_test_subset = pd.DataFrame(pro_x_test_subset, columns=all_elements)
    pro_x_external_testing = pd.DataFrame(pro_x_external_testing, columns=all_elements)
    pro_x_predict = pd.DataFrame(pro_x_predict, columns=all_elements)

    # Extraction of the required 7 elements from standardized data
    X_train_subset = pro_x_train_subset[selected_features].reset_index(drop=True)
    Y_train_subset = train_subset_data['Label'].astype(int).reset_index(drop=True)
    X_test_subset = pro_x_test_subset[selected_features].reset_index(drop=True)
    Y_test_subset = test_subset_data['Label'].astype(int).reset_index(drop=True)
    X_external_testing = pro_x_external_testing[selected_features].reset_index(drop=True)
    Y_external_testing = data.loc[data['Set'] == 'External testing set', 'Label'].reset_index(drop=True)
    X_predict = pro_x_predict[selected_features].reset_index(drop=True)
    all_X = pd.concat([X_train_subset, X_test_subset], axis=0).reset_index(drop=True)
    all_Y = pd.concat([Y_train_subset, Y_test_subset], axis=0).reset_index(drop=True)

    return X_train_subset, Y_train_subset, X_test_subset, Y_test_subset, X_external_testing, Y_external_testing, X_predict, all_X, all_Y, final_data, selected_features,all_elements

# input data
df = pd.read_excel("D:\iron_formation_recognition_main\data\Dataset-S1. Global iron formations dataset.xlsx")
file_path = r"D:\iron_formation_recognition_main\data\Dataset-S2. Training, external testing and application iron formations datasets.xlsx"
target_column = 'Iron Formation'

X_train_subset, Y_train_subset, X_test_subset, Y_test_subset, X_external_testing,Y_external_testing, X_predict, all_X, all_Y, final_data, selected_features,all_elements = prepare_data(file_path,
                                                                                                               target_column)


