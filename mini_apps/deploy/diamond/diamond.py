# -*- coding: utf-8 -*-

'''
@author: Xiaoyu Xie
@email: xiaoyuxie2020@u.northwestern.edu
@date: Sep, 2021
'''

import io

from derivative import dxdt
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
from numpy import linalg as LA
import pandas as pd
from scipy.linalg import svd
import streamlit as st
from sklearn.linear_model import Lasso



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

np.set_printoptions(precision=2)

matplotlib.use('agg')
encoder = preprocessing.LabelEncoder()

def removeoutliers(df, features, z):
    """
    Reomve outliers
    """
    for var in features:
        df_new = df[np.abs(stats.zscore(df[var])) < z]
    return df_new

def convert_catg(df):
    """
    convert categorical label to numerical label
    """
    # Find the columns of object type along with their column index
    object_cols = list(df.select_dtypes(exclude=[np.number]).columns)
    object_cols_ind = []
    for col in object_cols:
        object_cols_ind.append(df.columns.get_loc(col))
    
    # Encode the categorical columns with numbers 
    for i in object_cols_ind:
        df.iloc[:,i] = encoder.fit_transform(df.iloc[:,i])
    return df

def main():
    apptitle = 'diamond-app'
    st.set_page_config(
        page_title=apptitle,
        page_icon=':eyeglasses:',
        # layout='wide'
    )
    st.title('Diamond Price Prediction Service')

    st.image('src/dia.jpg')
    st.image('src/features.jpg')

    # level 1 font
    st.markdown("""
        <style>
        .L1 {
            font-size:40px !important;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    # level 2 font
    st.markdown("""
        <style>
        .L2 {
            font-size:20px !important;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    #########################Objectives#########################

    # st.markdown('<p class="L1">Objectives</p>', unsafe_allow_html=True)

    # dataset can be download from https://www.kaggle.com/shivam2503/diamonds/home
    df = pd.read_csv("src/diamonds.csv")
    
    def preprocessing(df):
        df.drop("Unnamed: 0", axis=1, inplace=True)
        df[['x','y','z']] = df[['x','y','z']].replace(0,np.NaN)
        df.dropna(inplace=True)

        col_names = df.columns.tolist()
        col_names.remove("price")
        numerical_cols = list(df.select_dtypes(include=[np.number]).columns)
        numerical_cols.remove("price")

        df = removeoutliers(df, numerical_cols, 3)
        
        df = convert_catg(df)

        X_df = df.drop(["price"],axis=1)
        y_df = df.price

        return X_df, y_df

    # def preprocessing_test(df):
    #     df = convert_catg(df)
    #     y_df = df.price
    #     return X_df, y_df

    X_df, y_df = preprocessing(df)

    # scaling
    scaler = StandardScaler()
    scaler.fit(X_df)
    X_df = scaler.transform(X_df)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, random_state = 2, test_size=0.3)

    reg_rf = RandomForestRegressor(n_estimators = 10)
    reg_rf.fit(X_train,y_train.values.ravel())
    y_pred = reg_rf.predict(X_test)

    print("Mean absolute error: {:.2f}".format(mean_absolute_error(y_test, y_pred)))
    print("Mean squared error: {:.2f}".format(mean_squared_error(y_test, y_pred)))
    print("R-Squared: {:.2f}".format(r2_score(y_test,y_pred)))

    depthlist = []
    tablelist = []
    caratlist = []
    xlist = []
    ylist = []
    zlist = []

    for i in np.arange(43.0, 79.0, 0.1):
        depthlist.append(round(i,1)) 

    for i in range(43,96):
        tablelist.append(i)

    for i in np.arange(0.20, 3.05,0.01):
        caratlist.append(round(i,2))

    for i in np.arange(3.73, 9.55,0.01):
        xlist.append(round(i,2))

    for i in np.arange(3.68, 31.9,0.01):
        ylist.append(round(i,2))

    for i in np.arange(1.53, 5.65,0.01):
        zlist.append(round(i,2))

    cut = st.sidebar.selectbox('Cut', ('Fair', 'Good', 'Ideal', 'Premium'))
    clarity = st.sidebar.selectbox('Clarity',
         ('I1', 'IF', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2'))
    color = st.sidebar.selectbox('Color', ('D','E','F','G','H','I','J'))
    carat = st.sidebar.selectbox('Carat', (caratlist))
    table = st.sidebar.selectbox('Table (The width of the diamond\'s table)', (tablelist))
    x = st.sidebar.selectbox('x', (xlist))
    y = st.sidebar.selectbox('y', (ylist))
    z = st.sidebar.selectbox('z', (zlist))
    depth = st.sidebar.selectbox('Depth', (depthlist))

    if st.sidebar.button("Predict Price"):
        X_test_new = np.array([carat, cut, color, clarity, depth, table, x, y, z]).reshape(-1, 1)
        df_test = pd.DataFrame([
            {
                'carat': carat,
                'cut': cut,
                'color': color,
                'clarity': clarity,
                'depth': depth,
                'table': table,
                'x': x,
                'y': y,
                'z': z,
                # 'price': 1
            },
            # {
            #     'carat': 4.0,
            #     'cut': 'ideal',
            #     'color': 'D',
            #     'clarity': 'IF',
            #     'depth': 65.5,
            #     'table': 59,
            #     'x': 10.74,
            #     'y': 10.54,
            #     'z': 6.98,
            #     # 'price': 1
            # },
            # {
            #     'carat': 0.5,
            #     'cut': 'ideal',
            #     'color': 'D',
            #     'clarity': 'IF',
            #     'depth': 65.5,
            #     'table': 59,
            #     'x': 10.74,
            #     'y': 10.54,
            #     'z': 6.98,
            #     # 'price': 1
            # },
        ])
        X_df_temp = convert_catg(df_test)
        X_test_new = scaler.transform(X_df_temp)
        y_test_new = reg_rf.predict(X_test_new)

        str_1 = 'Predicted price: {}'.format(y_test_new[0])
        st.markdown('<p class="L1">{}</p>'.format(str_1), unsafe_allow_html=True)

        if y_test_new[0] > 

        str_2 = 'Predicted label: {}'.format(y_test_new[0])
        st.markdown('<p class="L1">{}</p>'.format(str_2), unsafe_allow_html=True)




if __name__ == '__main__':
    main()
