import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report


st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.title('Bienvenue dans votre application.')

'## Dashboard'

'#### Observe the full dataset.'

data = pd.read_csv('data/concrete_data.csv')

col1, col2, col3, col4 = st.columns([7,1,2,2])
col1.dataframe(data, use_container_width = True)
col3.metric(label='Total count', value=data['cement'].count(), help ="Nombre total d'échantillons")
col4.metric(label='Cement mean', value=np.round(data['cement'].mean(), 2), help = "Moyenne de la colonne cement")
col3.metric(label='Age min', value=data['age'].min(), help = "Minimum de la colonne Age")
col4.metric(label='Age max', value=data['age'].max(), help = "Maximum de la colonne Age")
col3.metric(label='Water min', value=data['water'].min(), help = "Minimum de la colonne water")
col4.metric(label='Water max', value=data['water'].max(), help = "Maximum de la colonne water")
col3.write("Run profiling (statistiques avancées, matrices de corrélation, ...), peut prendre jusqu'à 30 secondes")
col4.write('Cliquer ici :')
pp_run = col4.button('Advanced profiling')

if pp_run :
    pr = data.profile_report()
    st_profile_report(pr)



'---'

'#### Explore the dataset.'

fig = px.scatter_matrix(data, dimensions=["cement", "blast_furnace_slag", "fly_ash", "water", "coarse_aggregate", "concrete_compressive_strength"], color="age", color_continuous_scale='Portland', template = 'simple_white', height = 1000)
st.plotly_chart(fig, use_container_width = True)

'---'

'#### Explore correlations.'

list_columns = list(data)

col1, col2, col3, col4 = st.columns(4)
first_column = col1.selectbox('Choisir colonne 1', list_columns, 0)
second_column = col2.selectbox('Choisir colonne 2', list_columns, 1)
color = col3.selectbox('Choisir la couleur', list_columns, 3)

col1, col2 = st.columns([3,2])
with col1 :
    fig = px.scatter(data, x=first_column, y=second_column, color=color, color_continuous_scale='Portland', height = 600,
                     template = 'simple_white', title = 'Scatter plot : ' + first_column + ' vs ' + second_column )
    st.plotly_chart(fig, use_container_width = True)
with col2 :
    ""
    ""
    ""
    fig = px.histogram(data, x=first_column, nbins=40, color_discrete_sequence=['indianred'], height = 250, template = 'simple_white')
    st.plotly_chart(fig, use_container_width = True)
    fig = px.histogram(data, x=second_column, nbins=40, color_discrete_sequence=['green'], height = 250, template = 'simple_white')
    st.plotly_chart(fig, use_container_width = True)

'---'

'#### Modelise the data.'


num= data.select_dtypes(include=['int64','float64']).keys()
from sklearn.impute import SimpleImputer
impute=SimpleImputer(strategy='mean')
impute_fit= impute.fit(data[num])
data[num]= impute_fit.transform(data[num])

# independent variables
x = data.drop(['concrete_compressive_strength'],axis=1)
# dependent variables
y = data['concrete_compressive_strength']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3,random_state=42)

from sklearn.preprocessing import StandardScaler
stand= StandardScaler()
Fit = stand.fit(xtrain)
xtrain_scl = Fit.transform(xtrain)
xtest_scl = Fit.transform(xtest)

col1, col2 = st.columns([4,6])
with col1 :
    ''
    ''
    'Exemple de Machine Learning sur la colonne "concrete_compressive_strength", choisissez ci-dessous le model à appliquer (librairie sklearn)'
    method = st.selectbox('Choisir quel model de Machine Learning appliquer', ['Linear regression', 'Lasso regression', 'Ridge regression', 'RandomForest Regression'], label_visibility = 'collapsed')

    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression, Lasso, Ridge
    from sklearn.ensemble import RandomForestRegressor



    if method == 'Linear regression' :
        lr=LinearRegression()
        fit=lr.fit(xtrain_scl,ytrain)
        score = np.round(lr.score(xtest_scl,ytest), 4)
        y_predict = lr.predict(xtest_scl)

    elif method == 'Lasso regression' :
        ls = Lasso(alpha = 0.3)
        fit=ls.fit(xtrain_scl,ytrain)
        score = np.round(ls.score(xtest_scl,ytest), 4)
        y_predict = ls.predict(xtest_scl)

    elif method == 'Ridge regression' :
        rd= Ridge(alpha=0.4)
        fit=rd.fit(xtrain_scl,ytrain)
        score = np.round(rd.score(xtest_scl,ytest), 4)
        y_predict = rd.predict(xtest_scl)

    elif method == 'RandomForest Regression' :
        rnd= RandomForestRegressor(ccp_alpha=0.0)
        fit_rnd= rnd.fit(xtrain_scl,ytrain)
        score = np.round(rnd.score(xtest_scl,ytest), 4)
        y_predict = rnd.predict(xtest_scl)


    'predicted score : ', score 
    mse = np.round(mean_squared_error(ytest,y_predict), 4)
    'mean_sqrd_error : ', mse
    rms = np.round(np.sqrt(mean_squared_error(ytest,y_predict)), 4)
    'root_mean_squared error : ', rms
    
with col2 :
    fig = px.scatter(data, x=y_predict, y=ytest, template = 'simple_white', title='Concrete strenght prediction', height = 600)
    fig.add_trace(go.Scatter(x=[ytest.min(), ytest.max()], y = [ytest.min(), ytest.max()], mode = 'lines', line_color = 'red', name = method))
    fig.update_xaxes(title = 'predicted').update_yaxes(title = 'original')
    st.plotly_chart(fig, use_container_width = True)