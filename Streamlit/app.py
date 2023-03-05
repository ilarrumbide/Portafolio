import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.metrics import precision_recall_curve, average_precision_score

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from sklearn.metrics import classification_report
#from secret import access_key, secret_access_key
import joblib
import streamlit as st

df = pd.read_csv('loan_sanction_train.csv')
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df =  df.fillna(df.mean())
df  = df.ffill().bfill()
df =  df.drop(['Loan_ID'],axis= 1)

model = joblib.load('xgb.joblib')

class SkewnessHandler(BaseEstimator, TransformerMixin):
    def __init__(self,feat_with_skewness=['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']):
        self.feat_with_skewness = feat_with_skewness
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.feat_with_skewness).issubset(df.columns)):
            # Handle skewness with cubic root transformation
            df[self.feat_with_skewness] = np.cbrt(df[self.feat_with_skewness])
            return df
        else:
            print("One or more features are not in the dataframe")
            return df


def get_dummies(df):
    df = pd.get_dummies(df)
    return df

class MinMaxWithFeatNames(BaseEstimator,TransformerMixin):
    def __init__(self,min_max_scaler_ft = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']):
        self.min_max_scaler_ft = min_max_scaler_ft
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.min_max_scaler_ft).issubset(df.columns)):
            min_max_enc = MinMaxScaler()
            df[self.min_max_scaler_ft] = min_max_enc.fit_transform(df[self.min_max_scaler_ft])
            return df
        else:
            print("One or more features are not in the dataframe")
            return df

df1 = df.copy()

target = df1['Loan_Status']
df1 = df1.drop(['Loan_Status'],axis =1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df1,
                                              target,
                                              test_size=0.2,
                                              stratify = target,
                                              random_state = 42,
                                              )

from sklearn.preprocessing import StandardScaler, FunctionTransformer

y_train = y_train.map({'Y': 1, 'N': 0})
y_test = y_test.map({'Y': 1, 'N': 0})

# Define the pipeline
preprocessor_transformer = FunctionTransformer(get_dummies)
#to_dataframe_transformer = FunctionTransformer(to_dataframe, kw_args={'columns':columns})
pipeline = Pipeline([
    ('get_dummies', preprocessor_transformer),
    ('SkewnessHandler',SkewnessHandler()),
    ('MinMaxWithFeatNames',MinMaxWithFeatNames())


    
])
X_processed = pipeline.fit_transform(X_train)





columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
           'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
           'Loan_Amount_Term', 'Credit_History', 'Property_Area']



st.title('Can i get a Loan Status? :house:')           

Gender = st.select_slider("Choose sex", ['Male','Female'])
Married = st.select_slider("Are you Married?", ['No','Yes'])
Dependents = st.select_slider("Choose Dependents", ['0','1','2','3+']
)
Education = st.select_slider("Choose your education level", ['Graduate', 'Not Graduate'])
Self_Employed = st.select_slider("Are you self employed", ['No', 'Yes'])
ApplicantIncome = float(st.number_input("Input your income", 0,91000))
CoapplicantIncome = st.number_input("Input your  coapplicant income", 0,51000)
LoanAmount  = st.number_input("Input your  loan amount", 0,10000)
Loan_Amount_Term = st.slider("Whats is the loan amount term",0,600)
Credit_History  = st.slider("Credit History",0,1)
Property_Area = st.select_slider("Choose the property area", ['Urban', 'Semiurban', 'Rural'])           

profile_to_predict_df = pd.DataFrame([[Gender, Married, Dependents, Education, Self_Employed, 
                                        ApplicantIncome, CoapplicantIncome, LoanAmount, 
                                        Loan_Amount_Term, Credit_History, Property_Area]], 
                                      columns=columns)
# Create a dataframe from the profile to predict

#train_copy_with_profile_to_pred = pd.concat([X_train,profile_to_predict_df],ignore_index=True)
#train_copy_with_profile_to_pred_prep = pipeline.fit_transform(train_copy_with_profile_to_pred)


test_copy_with_profile_to_pred = pd.concat([X_test,profile_to_predict_df],ignore_index=True)
pipeline.fit_transform(X_train)
X_test1 = pipeline.transform(X_test)
hola = pipeline.transform(test_copy_with_profile_to_pred)


#model = XGBClassifier(learning_rate = 0.001,n_estimators=241,sumsample = 0.5,max_depth = 7,colsample_bytree = 0.9)
model.fit(X_processed, y_train)

last_row = hola.tail(1)


def predict():
    prediction = model.predict(last_row)
    if prediction[0] == 1:
        st.success('You get a loan :thumbsup:')
    else:
        st.error('You don\'t get a loan :thumbsdown:')

# assign the button to a variable called 'trigger'
trigger = st.button('Predict')

# when the button is clicked, call the predict function
if trigger:
    predict()
import plotly.express as px
fi = model.feature_importances_

colorsy =['#ee4035','#f37736', '#fdf498', '#7bc043', '#0392cf', '#008744', '#0057e7', '#d62d20', '#ffa700']
tricolor=['#7bc043','#0392cf', '#f37736']
doucolor=['#008744','#0057e7']
twocolor=['#7bc043','#f37736']


def feature_importance():
    imp_df = pd.DataFrame(fi, columns = ['Imp'], index =X_processed.columns ).reset_index()\
    .sort_values('Imp', ascending=False)
    imp_df['Imp'] = imp_df['Imp'].round(decimals = 4)
    imp_df.columns = ['Features', 'Imp']

    fig =px.bar(imp_df, y='Features',x='Imp', orientation='h', color='Features', 
                color_discrete_sequence=colorsy,
                template='simple_white', text_auto='True')
    fig.update_layout(
        title='<b>Feature Importance</b>',
        font_family="Times New Roman",
        title_font_family="Times New Roman",title_font_color="#000000",
        title_font_size=20,
        xaxis_title="<b>Relative Importance</b>",
        yaxis_title="<b>Features</b>",
        legend_title='<b>Features</b>',
        legend_title_font_color="#000000",
                    plot_bgcolor ='#ffffff'
    )

    fig.show()

importance_feature = st.button('Show feature importance')
if importance_feature:
    feature_importance()