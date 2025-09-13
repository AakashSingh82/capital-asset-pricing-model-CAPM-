# importing libraries
import streamlit as st
import datetime
import pandas_datareader.data as web
import yfinance as yf
import pandas as pd
import capm_functions
import numpy as np
import plotly.express as px

# setting page config
st.set_page_config(

        page_title="CAPM",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )

st.title('Calculate Beta and Return for individual stock')

# getting input from user
col1, col2 = st.columns([1,1])
with col1:
    stock = st.selectbox("Choose a stock" , ('TSLA', 'AAPL','NFLX','MGM','MSFT','AMZN','NVDA','GOOGL'))
with col2:
    year = st.number_input("Number of Years",1,10)

# downloading data for SP500
end = datetime.date.today()
start = datetime.date(datetime.date.today().year - year, datetime.date.today().month, datetime.date.today().day)
SP500 = web.DataReader(['sp500'], 'fred', start, end)

# downloading data for the stock
stocks_df = yf.download(stock, period = f'{year}y')
stocks_df = stocks_df[['Close']]
stocks_df.columns = [f'{stock}']
stocks_df.reset_index(inplace = True)
SP500.reset_index(inplace = True)
SP500.columns = ['Date','sp500']
stocks_df['Date'] = stocks_df['Date'].astype('datetime64[ns]')
stocks_df['Date'] = stocks_df['Date'].apply(lambda x:str(x)[:10])
stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
stocks_df = pd.merge(stocks_df, SP500, on = 'Date', how = 'inner')

# calculating daily return 
stocks_daily_return = capm_functions.daily_return(stocks_df)
rm = stocks_daily_return['sp500'].mean()*252

# calculate beta and alpha
beta, alpha = capm_functions.calculate_beta(stocks_daily_return, stock)

# risk free rate of return
rf = 0

# market potfolio return
rm = stocks_daily_return['sp500'].mean()*252

# calculate return
return_value = round(rf+(beta*(rm-rf)),2)

# showing results
st.markdown(f'### Beta : {beta}')
st.markdown(f'### Return  : {return_value}')
fig = px.scatter(stocks_daily_return, x = 'sp500', y = stock, title = stock)
fig.add_scatter(x = stocks_daily_return['sp500'], y = beta*stocks_daily_return['sp500'] + alpha,  line=dict(color="crimson"))
st.plotly_chart(fig, use_container_width=True)