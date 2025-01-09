"""
Author: Yingdong Mao
Date: December 2020
"""
import time
from numpy import argmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
import numpy as np
import pandas as pd
import wrds
import statsmodels.api as sm
from sklearn import metrics
from pathos.multiprocessing import ProcessingPool as Pool
import warnings
warnings.filterwarnings('ignore')

# What I learned from this codes:
# I used multiprocessing and failed since function is not picklable
# Then I tried pathos, which works in my case and significantly
# improve the speed.
# see https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function/21345423#21345423

class Linear():
    """
    the input ought to be a daily data series
    """
    def __init__(self):
        self.svix = pd.read_pickle('Data/svix.pkl')
        self.freq = 'month'
        self.predictor = 'rv'
        self.window = 22
        self.rolling_window = 120
        self.percentile = 0.05
        self.beta = 1
        self.history = "rolling"
        self.run = 'series'
        self.log_method = 'sklearn'
        self.position = 0.5

    def frequency(self, input):
        df = input.copy()

        if self.freq == 'week':
            df['date'] = df.index
            df['prc'] = (df['return']+1).cumprod()
            df = df.join(df['date'].dt.isocalendar())
            df = df.groupby(['year', 'week']).nth(-1)
            df['return'] = df['prc'].pct_change()

        elif self.freq == 'month':
            df['date'] = df.index
            df['prc'] = (df['return']+1).cumprod()
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            df = df.groupby(['year', 'month']).nth(-1)
            df['return'] = df['prc'].pct_change()

        return df['return']

    def dstd(self, data):
        df = data
        df['return'] = df['return'][df['return']<0]

        if self.freq == 'month':
            df['date'] = df.index
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year

            temp1 = df.groupby(['year', 'month'])['return'].std()
            temp2 = df.groupby(['year', 'month'])['return'].count()

            temp = pd.Series(np.where(temp2 < 3, temp1.shift(1),
                                      temp1), index = temp1.index)
            temp.name = 'dstd'

            # temp = df['return'].rolling(
            #     self.window, closed='left', min_periods=3).std()
            # df['dstd'] = temp
            # df = df.groupby(['year', 'month']).nth(-1)
            # df = df['dstd']

            df = temp

        elif self.freq == 'week':
            df['date'] = df.index
            df = df.join(df['date'].dt.isocalendar())

            temp = df['return'].rolling(
                self.window, min_periods=3).std()
            df['dstd'] = temp
            df = df.groupby(['year', 'week']).nth(-1)
            df = df['dstd']
        df.fillna(method="ffill")
        outdata = df

        return outdata

    def assign_predictor(self, input):
        df = input.copy()

        if self.predictor == 'min':
            if self.freq == 'month':
                df['date'] = df.index
                df['month'] = df['date'].dt.month
                df['year'] = df['date'].dt.year
                df = df.groupby(['year', 'month']).min()
                df = df['return']
                df.name = 'min'


            elif self.freq == 'week':
                df['date'] = df.index
                df = df.join(df['date'].dt.isocalendar())
                df = df.groupby(['year', 'week']).min()
                df = df['return']
                df.name = 'min'

            outdf = df

        elif self.predictor == 'drv':
            if self.freq == 'month':
                df['date'] = df.index
                df['month'] = df['date'].dt.month
                df['year'] = df['date'].dt.year
                temp = df['return'].rolling(
                    self.window, closed = 'left').std()
                df['rv'] = temp
                df = df.groupby(['year', 'month']).nth(-1)
                df = df['rv']
                df = df.diff()

            elif self.freq == 'week':
                df['date'] = df.index
                df = df.join(df['date'].dt.isocalendar())
                temp = df['return'].rolling(
                    self.window).std()
                df['rv'] = temp
                df = df.groupby(['year', 'week']).nth(-1)
                df = df['rv']
            outdf = df


        elif self.predictor == 'rv':
            if self.freq == 'month':
                df['date'] = df.index
                df['month'] = df['date'].dt.month
                df['year'] = df['date'].dt.year
                temp = df['return'].rolling(
                    self.window, closed = 'left').std()
                df['rv'] = temp
                df = df.groupby(['year', 'month']).nth(-1)
                df = df['rv']

            elif self.freq == 'week':
                df['date'] = df.index
                df = df.join(df['date'].dt.isocalendar())
                temp = df['return'].rolling(
                    self.window).std()
                df['rv'] = temp
                df = df.groupby(['year', 'week']).nth(-1)
                df = df['rv']
            outdf = df

        elif self.predictor == 'lag_rv':
            if self.freq == 'month':
                df['date'] = df.index
                df['month'] = df['date'].dt.month
                df['year'] = df['date'].dt.year
                temp = df['return'].rolling(
                    self.window, closed = 'left').std()
                df['rv'] = temp
                df = df.groupby(['year', 'month']).nth(-1)
                df = df['rv']

            elif self.freq == 'week':
                df['date'] = df.index
                df = df.join(df['date'].dt.isocalendar())
                temp = df['return'].rolling(
                    self.window).std()
                df['rv'] = temp
                df = df.groupby(['year', 'week']).nth(-1)
                df = df['rv']
            outdf = df

        elif self.predictor in ['svix1','svix2','svix3','svix6',
                              'svix9']:
            svix = self.svix[['svix1','svix2','svix3','svix6',
                              'svix9']]
            if self.freq == 'week':
                svix['date'] = svix.index
                svix = svix.join(svix['date'].dt.isocalendar())
                price = svix.groupby(['year', 'week']).nth(-1)
                # week mean

                svix = svix.groupby(['year', 'week'])[['svix1',
                                                       'svix2',
                                                       'svix3',
                                                       'svix6',
                                                       'svix9']].mean()
                # temp = svix[['svix1','svix2','svix3','svix6',
                #               'svix9']].rolling(
                #     self.window, closed = 'left').mean()
                # svix[['svix1','svix2','svix3','svix6',
                #               'svix9']] = temp
                # svix = svix.groupby(['year', 'week']).nth(-1)

            elif self.freq == 'month':
                svix['date'] = svix.index
                svix['month'] = svix['date'].dt.month
                svix['year'] = svix['date'].dt.year
                price = svix.groupby(['year', 'month']).nth(-1)
                svix = svix.groupby(['year', 'month'])[['svix1',
                                                        'svix2',
                                                        'svix3',
                                                        'svix6',
                                                        'svix9']].mean()
            outdf = svix[self.predictor]

        elif self.predictor == 'dstd':
            outdf = self.dstd(df)
        return outdf

    def ols(self, y, x):
        n = len(x)
        k = 1
        x = np.concatenate([np.ones((n, 1)), np.array([x.to_numpy()]).transpose()],
                           axis=1)

        y = np.array(y)
        beta_hat = np.matmul(np.matmul(np.linalg.inv(
            np.matmul(np.array(x).transpose(), np.array(x))),
                                       x.transpose()), y)
        bh = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))
        ## Calculate vector of residuals
        residual = y - np.matmul(x,
                                 beta_hat)  # calculate the residual
        sigma_hat = sum(residual ** 2) / (
                    n - k - 1)  # estimate of error term variance
        variance_beta_hat = sigma_hat * np.linalg.inv(
            np.matmul(x.transpose(),
                      x))  # Calculate variance of OLS estimate
        se = np.array([variance_beta_hat[0,0], variance_beta_hat[1,
                                                                 1]]) ** 0.5

        t_stat = beta_hat / se

        return beta_hat, t_stat

    def get_rf(self):
        self.conn = wrds.Connection(wrds_username='maoyingdong')
        rf = self.conn.raw_sql("""select date, 
        rf from ff.factors_daily 
        """, date_cols=['date'])
        rf.index = rf['date']
        rf.columns = ['date', '1m']

        if self.freq == 'week':

            rf = rf.join(rf['date'].dt.isocalendar())
            rf = rf.groupby(['year', 'week']).sum()
            rf = np.exp(rf['1m'])-1

        elif self.freq == 'month':
            rf = self.conn.raw_sql("""select date, 
            rf from ff.factors_monthly 
            """, date_cols=['date'])

            rf['month'] = rf['date'].dt.month
            rf['year'] = rf['date'].dt.year
            rf.index = rf['date']
            rf = rf.drop(columns=['date'])
            # rf['eom'] = rf.groupby(['year', 'month'])['date'].transform(max)
            rf = rf.groupby(['year','month']).sum()
            rf.columns = ['1m']

        self.conn.close()
        return rf

    def dropna(self, x, y):
        temp = pd.concat([x, y], axis = 1)
        index = temp.dropna().index
        x = x[index]
        y = y[index]
        return x, y

    def ols_calculate(self):
        mandata = self.frequency()
        mandata = mandata.shift(-1)
        predictor = self.assign_predictor()
        mandata, predictor = self.dropna(mandata, predictor)
        bar = np.quantile(mandata ,self.percentile,
                          interpolation='linear')


        def rolling_ols(j):
            predict = {}
            position = {}
            true_next_event = {}

            if self.history == 'recursive':
                start = 1
            elif self.history == 'rolling':
                start = j - self.rolling_window + 1
                if start < 1:
                    start = 1
                else:
                    start = start

            y = mandata[start: j]
            x = predictor[start: j]

            b, tstat = self.ols(y, x)
            y_hat = b[1] * x + b[0]

            true_event = np.where(y < bar, 1, 0)

            thresh = self.optimal_thresh(y_hat, true_event)

            print('searching complete')
            today_x = np.array([1, predictor.iloc[j]])
            y_t = (b * today_x.transpose()).sum()
            predict_y = np.where(y_t < thresh, 1, 0)
            predict[mandata.index[j+1]] = predict_y
            position[mandata.index[j + 1]] = np.where(y_t < thresh,
                                                   0, 1)
            true_next_event[mandata.index[j + 1]] = np.where(
                mandata.iloc[
                                                              j + 1] < bar, 1, 0)

            predict_df = self.dic_to_df(predict)
            position_df = self.dic_to_df(position)
            true_next_event_df = self.dic_to_df(true_next_event)

            df = pd.concat([predict_df, position_df, true_next_event_df], axis = 1)
            df.columns = ['predict', 'position', 'true']
            return df

        # df = rolling_ols(200)

        start = time.time()
        pool_obj = Pool(22)

        result = pool_obj.map(rolling_ols, range(156,
                                      len(mandata) - 2))

        end = time.time()
        print(end - start)

        pool_obj.close()
        pool_obj.join()
        pool_obj.clear()

        df = pd.concat([i for i in result])
        df['return'] = mandata.shift(1)

        rf = self.get_rf()
        df['rf'] = rf
        binary = self.binary_series(df)
        return df, binary

    def optimal_thresh(self, fitted_y, true_y):
        F1 = []
        y_scores_list = fitted_y.tolist()
        for k in y_scores_list:
            y_predict = pd.Series(np.where(fitted_y > k, 1, 0),
                                  index=fitted_y.index)
            f1 = metrics.fbeta_score(true_y, y_predict,
                                     beta=self.beta)  # assign more weight to precision
            F1.append(f1)
        a = [index for index, value in enumerate(F1) if
             value == np.nanmax(np.array(F1, dtype=np.float64))]
        thresh = y_scores_list[a[0]]
        F1 = F1[a[0]]
        return thresh, F1

    def logit_calculate(self, data):
        data = data
        clum_num = len(data.columns)

        def deal_ser(i):
            print('dealing with '+str(i)+' simulation')
            ser = data.iloc[:,i]
            ser = ser.to_frame(name = 'return')
            mandata = self.frequency(ser)
            mandata = mandata.shift(-1)
            predictor = self.assign_predictor(ser)
            mandata, predictor = self.dropna(mandata, predictor)

            # global mandata
            # global logit
            collect_df = []

            if self.predictor in ['svix1', 'svix2', 'svix3', 'svix6',
                               'svix9']:
                starting_point = 36
            else:
                starting_point = self.rolling_window

            for j in range(starting_point, len(mandata) - 2):
                # print(j)
                predict = {}
                position = {}
                true_next_event = {}
                threshold = {}

                if self.history == 'recursive':
                    start = 1
                elif self.history == 'rolling':
                    start = j - self.rolling_window + 1
                    if start < 1:
                        start = 1
                    else:
                        start = start

                # set binary event
                y = mandata[start: j]
                x = predictor[start: j]

                bar = np.quantile(y, self.percentile,
                                  interpolation='linear')
                y = pd.Series(np.where(y < bar, 1, 0), index=y.index)

                if self.log_method == 'stats':
                    # print('stats')
                    one = pd.Series(np.ones(len(x)), index=x.index,
                                    name='constant')
                    x = pd.concat([one, x], axis=1)

                    logit_model = sm.Logit(y, x)
                    result = logit_model.fit(method='bfgs',
                                             maxiter=90,disp=False)  # it is possible
                    # that the Hessian is not positive definite when we evaluate it far away from the optimum, for example at bad starting values. Switching to an optimizer that does not use the Hessian often succeeds in those cases.

                    y_fitted = result.fittedvalues
                    y_true = y


                    # log_thresh, F1 = self.optimal_thresh(y_fitted, y_true)
                    # log_thresh = np.float64(log_thresh)
                    #
                    # thresh = np.exp(log_thresh) / (1 + np.exp(log_thresh))
                    # today_x = np.array([1, predictor.iloc[j]])
                    # y_t = (result.params * today_x.transpose()).sum()
                    # prob = np.exp(y_t) / (1 + np.exp(y_t))
                    #
                    #

                    # using function
                    y_fitted = np.exp(y_fitted) / (1 + np.exp(y_fitted))

                    # calculate roc curves
                    precision, recall, thresholds = precision_recall_curve(
                        y_true, y_fitted)
                    # convert to f score
                    fscore = ((1 + self.beta**2) * precision *
                              recall) / ((self.beta**2 * precision) + recall)
                    fscore = np.nan_to_num(fscore)
                    # locate the index of the largest f score
                    ix = argmax(fscore)
                    # print('Best Threshold=%f, F-Score=%.3f' % (
                    #     thresholds[ix], fscore[ix]))
                    thresh = thresholds[ix]
                    log_thresh = np.log(thresh / (1 - thresh))


                    today_x = np.array([1, predictor.iloc[j]])
                    y_t = (result.params * today_x.transpose()).sum()
                    prob = np.exp(y_t) / (1 + np.exp(y_t))

                elif self.log_method == 'sklearn':
                    # print('sklearn')
                    trainX, testX, trainy, testy = train_test_split(x, y,
                                                                    test_size=0.5,
                                                                    random_state=2,
                                                                    stratify=y)

                    ########### 11111111111111111111111111111111111111111
                    # model = LogisticRegression(solver='lbfgs')
                    # model.fit(trainX.to_numpy().reshape(-1,1), trainy)
                    # # predict probabilities
                    # yhat = model.predict_proba(testX.to_numpy().reshape(-1,1))
                    # yhat = yhat[:, 1]
                    # # calculate roc curves
                    # precision, recall, thresholds = precision_recall_curve(
                    #     testy, yhat)
                    ########### 2222222222222222222222222222222222222222
                    model = LogisticRegression(solver='lbfgs')
                    model.fit(x.to_numpy().reshape(-1,1), y)
                    # predict probabilities
                    yhat = model.predict_proba(x.to_numpy().reshape(-1,1))
                    yhat = yhat[:, 1]
                    # calculate roc curves
                    precision, recall, thresholds = precision_recall_curve(
                        y, yhat)


                    # convert to f score
                    fscore = ((1 + self.beta**2) * precision *
                              recall) / ((self.beta**2 * precision) + recall)
                    fscore = np.nan_to_num(fscore)
                    # locate the index of the largest f score
                    ix = argmax(fscore)
                    # print('Best Threshold=%f, F-Score=%.3f' % (
                    # thresholds[ix], fscore[ix]))
                    thresh = thresholds[ix]
                    log_thresh = np.log(thresh / (1-thresh))
                    yhat = model.predict_proba(np.float64(predictor.iloc[j]).reshape(-1,1))
                    prob = yhat[:,1][0]

                # print('thresh:',thresh.round(3))
                # print('log_thresh:',log_thresh.round(3))
                # print('y_t:',prob)

                predict_y = np.where(prob > thresh, 1, 0).max()

                predict[mandata.index[j + 1]] = predict_y
                threshold[mandata.index[j + 1]] = thresh
                position[mandata.index[j + 1]] = np.where(prob >
                thresh, 0, 1).max()
                true_next_event[mandata.index[j + 1]] = np.where(
                    mandata.iloc[j + 1] < bar, 1, 0)

                predict_df = self.dic_to_df(predict)
                position_df = self.dic_to_df(position)
                true_next_event_df = self.dic_to_df(true_next_event)
                threshold_df = self.dic_to_df(threshold)


                df = pd.concat([predict_df, position_df,
                                true_next_event_df,threshold_df], axis = 1)
                df.columns = ['predict', 'position', 'true',
                              'threshold']
                collect_df.append(df)
            DF = pd.concat(collect_df)
            DF['return'] = mandata.shift(1)
            return DF
        # df = logit(self.rolling_window)

        if self.run == 'parallel':
            start = time.time()
            pool_obj = Pool(4)

            result = pool_obj.map(deal_ser, range(clum_num))

            end = time.time()
            print(end - start)
            pool_obj.close()
            pool_obj.join()
            pool_obj.clear()

        elif self.run == 'series':
            result = []
            for i in range(clum_num):
                result.append(deal_ser(i))


        Dict = {}
        binary_table = []
        for i in range(len(data.columns)):
            name = data.columns[i]

            Dict[name] = result[i]
            binary = self.binary_series(result[i])
            temp = self.assign_position(result[i], self.position)
            sr = self.sharpe_ratio(temp, self.freq)
            binary = pd.concat([binary, sr])
            binary_table.append(binary)
        binary_table = pd.concat(binary_table, axis = 1)
        binary_table.columns = data.columns
        return Dict, binary_table

    def binary_series(self, data):
        true_event = data['true']
        predict = data['predict']

        tn, fp, fn, tp = confusion_matrix(true_event,
                                          predict).ravel()
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * recall * precision / (precision + recall)
        ser = pd.Series([tn, fp, fn, tp, recall, precision, f1],
        index=['tn', 'fp', 'fn', 'tp', 'recall', 'precision', 'f1'])
        return ser

    def dic_to_df(self, dic):
        df = pd.DataFrame.from_dict(dic, orient='index',
                                            columns=[self.predictor])
        df = df.astype(float)
        # predict_index = pd.MultiIndex.from_tuples(dic.keys(),names=('year', self.freq))
        # df.index = predict_index
        return df

    @ staticmethod
    def sharpe_ratio(data, freq, spx = False):
        if freq == 'month':
            x = 12
        elif freq == 'week':
            x = 52

        SR = data['st_return'].mean() / data[
            'st_return'].std() * x \
             ** 0.5
        Original_SR = data['return'].mean() / data['return'].std() \
                      * x ** 0.5

        ser = pd.Series([SR, Original_SR],index=['SR', 'Original_SR'])
        return ser


    @ staticmethod
    def assign_position(data, pos, spx=False):
        if (spx == True) & (pos == 0):
            data['position'] = np.where(data['position'] == 1, 1, pos)
            data['st_return'] = (data['return'] - data['rf']) * data[
                'position']
        else:
            data['position'] = np.where(data['position'] == 1, 1, pos)
            data['st_return'] = data['return'] * data['position']
        return data


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    data = pd.read_csv('Data/example.pickle')
    start = Linear()
    start.percentile = 0.05
    start.window = 1 * 22
    start.rolling_window = 10 * 12
    start.beta = 1
    start.log_method = 'stats'
    start.predictor = 'rv'
    start.history = 'rolling'
    # start.gamma = gamma
    start.run = 'series'
    Dict, binary = start.logit_calculate(data)