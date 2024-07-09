import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
from os.path import join, exists
from os import makedirs


def calibrate_proba_monthly_recal(df, start_month, end_month, save_folder=None):
    """
    For each observation of the dataset, if M is the corresponding month, outputs the calibrated predicted probability of the model from the uncalibrated one, 
    using an isotonic regression trained on the M-12 to M-1 period.
    We hence fit a number of isotonic regressions equal to the number of months in the dataset.
    This allows to calibrate the probability output of the model in a "live" setting, where each month, the probabilities are calibrated with the new data.
    """
    observation_month = df.index.str[:6].astype(int)
    months = sorted(observation_month.unique())
    test_months = [month for month in  months if month >= start_month and month <= end_month]
    y_probs_cal_list =  []

    for test_month in tqdm(test_months):
        # For every month M, we take the training period as M-12 to M-1
        idx = months.index(test_month)
        train_months = months[idx-12:idx]
        select_train = observation_month.isin(train_months)
        x_train = df.loc[select_train, 'y_probs']
        y_train = df.loc[select_train, 'Result']
        # And the test period as month M
        x_test = df.loc[observation_month == test_month, 'y_probs']

        ir = IsotonicRegression(y_min=0, y_max=1).fit(x_train, y_train)
        if save_folder:
            if not exists(save_folder):
                makedirs(save_folder)
            save_path = join(save_folder, f'calibrator_{test_month}.joblib')
            joblib.dump(ir, save_path)
        y_probs_cal = ir.predict(x_test)
        # print("Test month is: ", test_month)
        # print("Train months are: ", train_months)
        # print("\n")

        y_probs_cal_list.append(y_probs_cal)
    
    return pd.Series(np.concatenate(y_probs_cal_list, axis=0), index=df[observation_month.isin(test_months)].index)