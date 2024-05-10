import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

st.title('Customer Churn Model')
st.markdown("<h3></h3>", unsafe_allow_html=True)

st.markdown("<h3></h3>", unsafe_allow_html=True)
st.markdown("""
This is a customer churn prediction model, made with the Kaggle Dataset - Telco Customer Churn. 

The goal is to explore the dataset from the perspective of a Telco business manager and understand which parameters can be adjusted to reduce customer churn.
""")
st.markdown("<h3></h3>", unsafe_allow_html=True)

#read data
@st.cache_data
def read_data(data_path):
    df1 = pd.read_csv(data_path)
    df1.replace(' ', '0', inplace=True)
    return df1


df_original = read_data('Data.csv')

# display Data Table

df_display = df_original.copy(deep=True)
st.header('Data')
st.markdown("<h4></h4>", unsafe_allow_html=True)
table1 = st.empty()
table1.dataframe(df_display)
st.markdown("<h3></h3>", unsafe_allow_html=True)

#Read Clustering Centroids data

#Clustering already done in K-Means-Clustering.ipynb jupyter notebook in this same directory

@st.cache_data
def read_cent(data_path):
    dfc = pd.read_csv(data_path)
    return dfc

df_cent = read_cent("Data_centroids.csv")

# displaying centroid table
df_cent_disp = df_cent.copy(deep=True)
st.header('Centroids')
st.markdown("""
The data has been clustered using K-Means and the cluster centroids are displayed in the table below. A centroid is the central point of a cluster. It represents the mean or average position of all the points in the cluster. """)
st.markdown("<h4></h4>", unsafe_allow_html=True)
tabel2 = st.empty()
tabel2.dataframe(df_cent_disp)
st.markdown("<h3></h3>", unsafe_allow_html=True)

# trained random forest model- rfc_best as a imported pickle file
rfc_best = pickle.load(open('finalized_model.sav', 'rb'))


# filtering Data

st.sidebar.subheader("Filter:")
df_category = [x for x in df_display.columns if len(df_display[x].unique()) < 6]
df_category.append('cluster_number')
df_num = list(set(df_display.columns) - set(df_category))
group = st.sidebar.multiselect("Select parameter(s) to Filter by: ", [x for x in df_category])

df_fil = df_display.copy(deep=True)
df_temp = df_fil.copy(deep=True)

try:

    l_df = []

    for x in range(len(group)):
        d = {'key1': str(group[x])}
        s = "{key1}:"
        s = s.format(**d)
        attri = st.sidebar.multiselect(label=s, options=df_display[(group[x])].unique(), key=str(x))
        # st.write('selected',attri,type(attri))

        for y in range(0, len(attri)):
            if (df_fil.equals(df_temp)):
                df_fil = df_fil[df_fil[(group[x])] == attri[y]]
            else:
                df_fil = pd.concat([df_fil, df_temp[df_temp[(group[x])] == attri[y]]])
                df_fil.sort_index(inplace=True)
        df_temp = df_fil
    
except(IndexError):
    st.error("Select parameters Please!")

df_display = df_temp

# customer Search

st.sidebar.subheader('Customer Search:')
sbar = st.sidebar.text_input("Enter Customer ID:", key='sb')

df_sear = pd.read_csv('Searched_Records.csv')

sdf_title = st.empty()
st.markdown("<h4></h4>", unsafe_allow_html=True)
searched_df = st.empty()
st.markdown("<h3></h3>", unsafe_allow_html=True)
cdf_title = st.empty()
st.markdown("<h4></h4>", unsafe_allow_html=True)
custom_df = st.empty()

if st.sidebar.button('Search'):
    df_sear = df_original[df_original['customerID'] == sbar]
    df_sear.to_csv("Searched_Records.csv", mode='w+', index=False)
    sdf_title.subheader('Searched Record')
    searched_df.dataframe(df_sear)

    my_dict = {'customerID': df_sear['customerID'], 'gender': df_sear['gender'],
               'SeniorCitizen': df_sear['SeniorCitizen'],
               'Partner': df_sear['Partner'], 'Dependents': df_sear['Dependents']}


# centroid Search
st.sidebar.subheader('Centroid Search:')
sbar1 = st.sidebar.text_input("Enter Cluster Number:", key='cn')

if st.sidebar.button('Search', key='s2'):
    sdf_title.subheader("Searched Record~")
    df_sear = df_cent[df_cent['cluster_number'] == int(sbar1)]
    df_sear.to_csv("Searched_Records.csv", mode='w+', index=False)
    searched_df.dataframe(df_sear)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    my_dict = {'cluster_number': df_sear['cluster_number'], 'gender': df_sear['gender'],
               'SeniorCitizen': df_sear['SeniorCitizen'],
               'Partner': df_sear['Partner'], 'Dependents': df_sear['Dependents']}


# customized Use Case

st.sidebar.subheader('Customized Use-case:')
select_param = {}
count = 0
df_category.remove('Churn')
df_category1 = df_category
categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']

for x in categorical_features:
    df_category1.remove(x)

# temporary changes for presentation
df_category1 = ['InternetService', 'OnlineSecurity']

for x in df_category1:
    select_param_description = {'key1': str(x)}
    churn_status = "Select {key1}:"
    s = churn_status.format(**select_param_description)
    unique_values = list(df_original[x].unique())
    selected_index = unique_values.index(str(df_sear[x].iloc[0]))
    temp = st.sidebar.selectbox(label=s, index=selected_index, options=unique_values,
                                key=str(x))  # index=(unique_values.index(df_sear[x])),
    select_param[x] = temp

df_num1 = df_num
df_num1.remove('customerID')

# temporary changes for presentation

df_num1 = ['tenure', 'TotalCharges']

for y in df_num1:
    d = {'key1': str(y)}
    s = "Set {key1}:"
    s = s.format(**d)
    temp1 = st.sidebar.slider(label=s, min_value=(df_original[y].unique()).astype(float).min(),
                              max_value=(df_original[y].unique()).astype(float).max(),
                              value=float(df_sear[y]),
                              step=(((df_original[y].unique()).astype(float).max()) - int(
                                  (df_original[y].unique()).astype(float).min())) / 10,
                              key=str(y))
    select_param[y] = temp1


try:
    for x in my_dict.keys():
        df_sear[x] = my_dict[x]
except NameError:
    print('')
for x in select_param.keys():
    df_sear[x] = select_param[x]


cdf_title.subheader("Example User")

try:
    custom_df.dataframe(df_sear)  # .transpose())
except:
    st.error("Search Value not entered yet!")

if ('customerID' in df_sear.columns):
    l_list = [df_sear.iloc[0:1], df_original.iloc[0:1000]]

else:
    searched = df_sear
    searched['customerID']='Dummy'
    searched = searched[df_original.columns]
    l_list=[searched, df_original.iloc[0:1000]]

df_concatenated_data = pd.concat(l_list)

def convert_data(df_churn):
    df_churn = df_churn.reset_index(drop=True)
    empty_cols = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                  'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                  'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                  'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']

    for i in empty_cols:
        df_churn[i] = df_churn[i].replace(" ", np.nan)

    df_churn.drop(['customerID', 'cluster_number'], axis=1, inplace=True)
    df_churn = df_churn.dropna()
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

    for i in binary_cols:
        df_churn[i] = df_churn[i].replace({"Yes": 1, "No": 0})

    # Encoding column 'gender'
    df_churn['gender'] = df_churn['gender'].replace({"Male": 1, "Female": 0})
    df_churn['Churn'] = df_churn['Churn'].replace({"Yes": 1, "No": 0})
    df_churn['PaymentMethod'] = df_churn['PaymentMethod'].replace({"Yes": 1, "No": 0})

    category_cols = ['PaymentMethod', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract']

    for cc in category_cols:
        dummies = pd.get_dummies(df_churn[cc], drop_first=False)
        dummies = dummies.add_prefix("{}#".format(cc))
        df_churn.drop(cc, axis=1, inplace=True)
        df_churn = df_churn.join(dummies)

    return df_churn


df_convert_data = convert_data(df_concatenated_data)

l = list(df_convert_data.columns)
l = [l[-1]] + l[:-1]
df_convert_data = df_convert_data[l]
to_pred = df_convert_data.iloc[0:1]

X1 = to_pred.loc[:, df_convert_data.columns != 'Churn']
y = to_pred.loc[:, df_convert_data.columns == 'Churn']
y_pred_prob = rfc_best.predict_proba(X1)
y_pred = rfc_best.predict(X1)
st.markdown("<h3></h3>", unsafe_allow_html=True)
st.markdown("<h3></h3>", unsafe_allow_html=True)

s = "Not Churned"
if (y_pred_prob[0, 0] < y_pred_prob[0, 1]):
    s = 'Churned'

not_churned_degrees = y_pred_prob[0, 0] * 360
churned_degrees = y_pred_prob[0, 1] * 360


def make_pie(sizes, text, colors, labels):
    col = [[i / 255. for i in c] for c in colors]
    fig, ax = plt.subplots()
    ax.axis('equal')
    width = 0.45
    kwargs = dict(colors=col, startangle=180)
    outside, _ = ax.pie(sizes, radius=1, pctdistance=1 - width / 2, labels=labels, **kwargs)
    plt.setp(outside, width=width, edgecolor='white')
    kwargs = dict(size=15, fontweight='bold', va='center')
    ax.text(0, 0, text, ha='center', **kwargs)
    ax.set_facecolor('#e6eaf1')

red_color = (226, 33, 7)
green_color = (20, 100, 20)

make_pie([not_churned_degrees, churned_degrees], s, [green_color, red_color], ['Probability(Not Churn): \n{0:.2f}%'.format(y_pred_prob[0, 0] * 100),
                                    'Probability (Churn): \n{0:.2f}%'.format(y_pred_prob[0, 1] * 100)])

st.pyplot()