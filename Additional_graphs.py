import streamlit as st
import pandas as pd
from plotly import graph_objs as go
import plotly.express as px
from datetime import date
import yfinance as yf
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from SinglePeriod import SinglePeriod
from classical_po import ClassicalPO
from scipy.optimize import minimize
from SinglePeriod import SinglePeriod
import optuna
from itertools import product
import dimod
import datetime
from dimod import quicksum, Integer, Binary
from dimod import ConstrainedQuadraticModel
from dwave.system import LeapHybridCQMSampler
#from tabulate import tabulate
from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo
from dwave.samplers import SimulatedAnnealingSampler
import ml_collections
from statsmodels.tsa.filters.hp_filter import hpfilter
optuna.logging.set_verbosity(optuna.logging.WARNING)

seed = 42
cfg = ml_collections.ConfigDict()
cfg.hpfilter_lamb = 6.25
cfg.q = 1.0  # risk-aversion factor should ideally be 1.0
# classical
cfg.fmin = 0.001  # 0.001
cfg.fmax = 0.5  # 0.5

st.set_page_config(page_title="Requirements", page_icon=":chart_with_upwards_trend:", layout="wide")
st.title("Requirements")
if st.button('a'):
    hdfc50 = pd.read_csv('HDFCNIFETF.NS.csv')
    hdfc50['HDFCNIFETF.NS'] = hdfc50['Close']
    hdfc50 = hdfc50[['Date', 'HDFCNIFETF.NS']]
    total_investment_amount = 1604500.43
    day1_price = hdfc50['HDFCNIFETF.NS'].iloc[0]
    optimal_stocks_to_buy_hdfcnifty50 = total_investment_amount // day1_price
    portfolio_values_nifty50 = hdfc50['HDFCNIFETF.NS']*optimal_stocks_to_buy_hdfcnifty50
    hdfc50['Portfolio Value'] = portfolio_values_nifty50
    hdfc50['Return'] = hdfc50['Portfolio Value'] * 100 / hdfc50['Portfolio Value'][0]
    hdfc50['Date'] = pd.to_datetime(hdfc50['Date'], format='%d-%m-%Y')
    stock_closing_prices = pd.read_csv('stock_closing_prices.csv')
    weights = {
    'BHARTIARTL.NS': 0.0523,
    'HDFCBANK.NS':  0.0936,
    'HINDUNILVR.NS': 0.1491,
    'ICICIBANK.NS': 0.0552,
    'INFY.NS':  0.0841,
    'ITC.NS': 0.0253,
    'LT.NS':   0.1588,
    'RELIANCE.NS':  0.1449,
    'SBIN.NS': 0.0342,
    'TCS.NS': 0.2025}
    investment_per_stock = {stock: total_investment_amount * weight for stock, weight in weights.items()}
    optimal_stocks = {stock: investment // stock_closing_prices.loc[0, stock] for stock, investment in investment_per_stock.items()}
    portfolio_values = stock_closing_prices.apply(lambda row: sum(row[stock] * optimal_stocks[stock] for stock in optimal_stocks), axis=1)
    stock_closing_prices['Portfolio Value'] = portfolio_values
    stock_closing_prices['Return'] = stock_closing_prices['Portfolio Value'] * 100 / stock_closing_prices['Portfolio Value'][0]
    
    fig_compare = go.Figure()

    fig_compare.add_trace(go.Scatter(x= stock_closing_prices['Date'], 
                    y=  stock_closing_prices['Return'],
                    mode='lines+markers', 
                    name='Return AMAR', 
                    line=dict(color='red')))
    
    fig_compare.add_trace(go.Scatter(x=hdfc50['Date'], 
                    y=hdfc50['Return'], 
                    mode='lines+markers', 
                    name='Return HDFCNIFTY50ETF', 
                    line=dict(color='blue')))
    
    fig_compare.update_layout(title='Return Over Time',
                xaxis_title='Date', 
                yaxis_title='Return',
                autosize=False, 
                width=1000, 
                height=600,)
    st.write('Line chart against the benchmark (Rebased to 100 for initial date)')
    st.plotly_chart(fig_compare)

if st.button('b'):
    st.write('Portfolio weight vs the Benchmark weight (Clearly highlight the overweight and underweight sectors)')
    port = {'Information Technology': '28.66', 'Financials':'18.29', 'Consumer Staples':'17.44', 'Industrials':'15.88', 'Energy':'14.49', 'Communication Services':'5.23'}
    bench = {'Information Technology': '13.64', 'Financials':'36.37', 'Consumer Staples':'9.38', 'Industrials':'5.40', 'Energy':'11.89', 'Communication Services':'2.63'}
    common_keys = set(port.keys()) & set(bench.keys())
    for key in common_keys:
        port_value = float(port[key])
        bench_value = float(bench[key])
        difference = port_value - bench_value
        if difference > 0:
            st.text(f"{key}: Overweight by {abs(difference)}")
        elif difference < 0:
            st.text(f"{key}: Underweight by {abs(difference)}")
        else:
            st.text(f"{key}: Neutral")


if st.button('c'):
    st.write('Sector wise weight and Sebi classification-wise weights')
    bench_weights = {'HDFC Bank Ltd': 11.48,
         'Reliance Industries Ltd': 9.96,
         'ICICI Bank Ltd': 8.11,
         'Infosys Ltd':5.09,
         'Larsen & Toubro Ltd':4.27,
         'Tata Consultancy Services Ltd':3.89,
         'ITC Ltd':3.88,
         'Bharti Airtel Ltd':3.45,
         'Axis Bank Ltd':3.32,
         'State Bank of India':3.18,
         'Kotak Mahindra Bank Ltd':2.40 ,
         'Mahindra & Mahindra Ltd' :2.07 ,
         'Hindustan Unilever Ltd' :2.00 ,
         'Bajaj Finance Ltd' :1.94 ,
         'Tata Motors Ltd' :1.78 ,
         'NTPC Ltd' :1.73 ,
         'Maruti Suzuki India Ltd' :1.70 ,
         'Sun Pharmaceuticals Industries Ltd' : 1.63 ,
         'Titan Co Ltd' :1.50 ,
         'HCL Technologies Ltd' : 1.45,
         'Power Grid Corp Of India Ltd' :1.38 ,
         'Tata Steel Ltd' : 1.36 ,
         'Asian Paints Ltd' : 1.30,
         'UltraTech Cement Ltd' : 1.16,
         'Oil & Natural Gas Corp Ltd' :1.11 ,
         'Coal India Ltd' :1.04 ,
         'Bajaj Auto Ltd' :1.01 ,
         'IndusInd Bank Ltd' :1.01 ,
         'Adani Ports & Special Economic Zone Ltd' : 0.98 ,
         'Hindalco Industries Ltd' : 0.94 ,
         'Nestle India Ltd' : 0.90 ,
         'Grasim Industries Ltd' : 0.89 ,
         'Bajaj Finserv Ltd' : 0.88 ,
         'JSW Steel Ltd' : 0.84 ,
         'Tech Mahindra Ltd' : 0.81 ,
         'Adani Enterprises Ltd' : 0.80 ,
         'Dr Reddys Laboratories Ltd' : 0.76 ,
         'Cipla Ltd' : 0.74 ,
         'Shriram Finance Ltd' : 0.71 ,
         'Tata Consumer Products Ltd' : 0.70 ,
         'Wipro Ltd' : 0.65 ,
         'SBI Life Insurance Company Limited' : 0.65 ,
         'Eicher Motors Ltd' : 0.63 ,
         'HDFC Life Insurance Company Limited' : 0.62 ,
         'Apollo Hospitals Enterprise Ltd' : 0.60 ,
         'Hero MotoCorp Ltd' : 0.59 ,
         'Bharat Petroleum Corp Ltd' : 0.58 ,
         'Britannia Industries Ltd' : 0.57 ,
         'Divis Laboratories Ltd' : 0.51 ,
         'LTIMindtree Ltd' : 0.43 }

    port_weights = {'Tata Consultancy Services Ltd':20.25,
        'Infosys Ltd':8.41,
        'HDFC Bank Ltd':9.36,
        'ICICI Bank Ltd':5.52,
        'State Bank of India':3.42,
        'Hindustan Unilever Ltd':14.91,
        'ITC Ltd':2.53,
        'Larsen & Toubro Ltd':15.88,
        'Reliance Industries Ltd': 14.49,
        'Bharti Airtel Ltd': 5.23}
    
    large_cap = ['HDFC Bank Ltd',
         'Reliance Industries Ltd',
         'ICICI Bank Ltd',
         'Infosys Ltd',
         'Larsen & Toubro Ltd',
         'Tata Consultancy Services Ltd',
         'ITC Ltd',
         'Bharti Airtel Ltd',
         'Axis Bank Ltd',
         'State Bank of India',
         'Kotak Mahindra Bank Ltd',
         'Mahindra & Mahindra Ltd'  ,
         'Hindustan Unilever Ltd'  ,
         'Bajaj Finance Ltd'  ,
         'Tata Motors Ltd'  ,
         'NTPC Ltd' ,
         'Maruti Suzuki India Ltd' ,
         'Sun Pharmaceuticals Industries Ltd'  ,
         'Titan Co Ltd' ,
         'HCL Technologies Ltd',
         'Power Grid Corp Of India Ltd' ,
         'Tata Steel Ltd' ,
         'Asian Paints Ltd',
         'UltraTech Cement Ltd' ,
         'Oil & Natural Gas Corp Ltd'  ,
         'Coal India Ltd'  ,
         'Bajaj Auto Ltd' ,
         'IndusInd Bank Ltd' ,
         'Adani Ports & Special Economic Zone Ltd' ,
         'Hindalco Industries Ltd'  ,
         'Nestle India Ltd' ,
         'Grasim Industries Ltd'  ,
         'Bajaj Finserv Ltd' ,
         'JSW Steel Ltd'  ,
         'Tech Mahindra Ltd'  ,
         'Adani Enterprises Ltd'  ,
         'Dr Reddys Laboratories Ltd' ,
         'Cipla Ltd',
         'Shriram Finance Ltd' ,
         'Tata Consumer Products Ltd' ,
         'Wipro Ltd' ,
         'SBI Life Insurance Company Limited'  ,
         'Eicher Motors Ltd',
         'HDFC Life Insurance Company Limited',
         'Apollo Hospitals Enterprise Ltd' ,
         'Bharat Petroleum Corp Ltd' ,
         'Britannia Industries Ltd' ,
         'Divis Laboratories Ltd' ,
         'LTIMindtree Ltd']

    mid_cap = ['Hero MotoCorp Ltd']

    large_cap_weight = sum(float(bench_weights.get(company, 0)) for company in large_cap)
    mid_cap_weight = sum(float(bench_weights.get(company, 0)) for company in mid_cap)

    labels = ['Large Cap', 'Mid Cap']
    sizes = [large_cap_weight, mid_cap_weight]

    colors = ['gold', 'DeepPink' ]
    fig_sector = go.Figure(data=[go.Pie(labels=labels,values=sizes, hole=.3)])
    fig_sector.update_traces(hoverinfo='label+percent',textfont_size=15, marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    st.markdown("**Benchmark Weight Distribution**")
    st.plotly_chart(fig_sector)

    large_cap_weight_port = sum(float(port_weights.get(company, 0)) for company in large_cap)
    mid_cap_weight_port = sum(float(port_weights.get(company, 0)) for company in mid_cap)

    labels_port = ['Large Cap', 'Mid Cap']
    sizes_port = [large_cap_weight_port, mid_cap_weight_port]

    colors = ['gold', 'DeepPink' ]
    fig_sector_port = go.Figure(data=[go.Pie(labels=labels_port,values=sizes_port, hole=.3)])
    fig_sector_port.update_traces(hoverinfo='label+percent',textfont_size=15, marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    st.markdown("**Portfolio Weight Distribution**")
    st.plotly_chart(fig_sector_port)

if st.button('e'):
    st.write('Top 10 best and Bottom 10 worst performers(Based on Return)')
    adj_close_df = pd.read_csv(f'stock_closing_prices.csv', usecols=range(11))
    data = adj_close_df.drop(columns=['Date'])
    tickers = sorted(["BHARTIARTL.NS", "HDFCBANK.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INFY.NS", "ITC.NS", "LT.NS", "RELIANCE.NS", "SBIN.NS", "TCS.NS"])
    for s in data.columns:
        cycle, trend = hpfilter(data[s], lamb=cfg.hpfilter_lamb)
        data[s] = trend

    log_returns = np.log(data) - np.log(data.shift(1))
    null_indices = np.where((log_returns.isna().sum() > 1).to_numpy())[0]
    drop_stocks = data.columns[null_indices]
    log_returns = log_returns.drop(columns=drop_stocks)
    log_returns = log_returns.dropna()
    tickers = log_returns.columns

    cfg.num_stocks = len(tickers)

    mu = log_returns.mean()* 252
    sigma = log_returns.cov().to_numpy() * 252

    st.markdown('**Bottom 10 Worst**')
    st.write(mu.sort_values())

    st.markdown('**TOp 10 Best**')
    st.write(mu.sort_values(ascending=False))

if st.button('f'):
    st.text('Top 10 relative weight, Bottom 10 relative weight')
    port_weights = {'Tata Consultancy Services Ltd':'20.25',
        'Infosys Ltd':'8.41',
        'HDFC Bank Ltd':'9.36',
        'ICICI Bank Ltd':'5.52',
        'State Bank of India':'3.42',
        'Hindustan Unilever Ltd':'14.91',
        'ITC Ltd':'2.53',
        'Larsen & Toubro Ltd':'15.88',
        'Reliance Industries Ltd': '14.49',
        'Bharti Airtel Ltd': '5.23'}

    benchmark_weights = {'Tata Consultancy Services Ltd':'4.17',
        'Infosys Ltd':'6.07',
        'HDFC Bank Ltd':'11.20',
        'ICICI Bank Ltd':'7.76',
        'State Bank of India':'2.64',
        'Hindustan Unilever Ltd':'2.68',
        'ITC Ltd':'4.49',
        'Larsen & Toubro Ltd':'3.84',
        'Reliance Industries Ltd': '9.89',
        'Bharti Airtel Ltd': '2.63'}
    
    keys = set(port_weights.keys()) & set(benchmark_weights.keys())
    results = []
    for key in keys:
        port_value = float(port_weights[key])
        bench_value = float(benchmark_weights[key])
        difference = port_value - bench_value
        results.append((key, abs(difference)))
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    sorted_results_bottom = sorted(results, key=lambda x: x[1], reverse=False)
    st.markdown('**Top 10 Relative Weight**')
    st.text(sorted_results)

    st.markdown('**Bottom 10 Relative Weight**')
    st.text(sorted_results_bottom)

if st.button('g'):
    st.write('Top 10 holdings, Bottom 10 holdings (With performance of 1m, 3m, 6m, 1 yr)')
    stock_data = pd.read_csv('stock_closing_prices.csv')
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    
    def func(dataframe, days):
        log_returns = np.log(dataframe) - np.log(dataframe.shift(1))
        null_indices = np.where((log_returns.isna().sum() > 1).to_numpy())[0]
        drop_stocks = dataframe.columns[null_indices]
        log_returns = log_returns.drop(columns=drop_stocks)
        log_returns = log_returns.dropna()
        tickers = log_returns.columns

        mu = log_returns.mean() * days
        sigma = log_returns.cov().to_numpy() * days

        return mu.sort_values()
    
    start_date = pd.to_datetime('2023-02-01')
    end_date = pd.to_datetime('2023-03-01')
    filtered_df = stock_data[(stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)].drop(columns=['Date'])

    start_date_3 = pd.to_datetime('2023-02-01')
    end_date_3 = pd.to_datetime('2023-05-01')
    filtered_df_3 = stock_data[(stock_data['Date'] >= start_date_3) & (stock_data['Date'] <= end_date_3)].drop(columns=['Date'])

    start_date_6 = pd.to_datetime('2023-02-01')
    end_date_6 = pd.to_datetime('2023-08-01')
    filtered_df_6 = stock_data[(stock_data['Date'] >= start_date_6) & (stock_data['Date'] <= end_date_6)].drop(columns=['Date'])

    start_date_1y = pd.to_datetime('2023-02-01')
    end_date_1y = pd.to_datetime('2024-02-01')
    filtered_df_1y = stock_data[(stock_data['Date'] >= start_date_1y) & (stock_data['Date'] <= end_date_1y)].drop(columns=['Date'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('**One Month Return**')
        st.write(func(filtered_df, 21))
    with col2:
        st.markdown('**Three Month Return**')
        st.write(func(filtered_df_3, 57))
    with col3:
        st.markdown('**Six Month Return**')
        st.write(func(filtered_df_6, 123))
    with col4:
        st.markdown('**One Year Return**')
        st.write(func(filtered_df_1y, 246))

if st.button('h'):
    st.write('Top 10 contribution to returns, Bottom 10 contribution to returns')
    port_weights = {'Tata Consultancy Services Ltd': 20.25,
        'Infosys Ltd': 8.41,
        'HDFC Bank Ltd': 9.36,
        'ICICI Bank Ltd': 5.52,
        'State Bank of India': 3.42,
        'Hindustan Unilever Ltd': 14.91,
        'ITC Ltd': 2.53,
        'Larsen & Toubro Ltd': 15.88,
        'Reliance Industries Ltd': 14.49,
        'Bharti Airtel Ltd': 5.23}

    benchmark_weights = {'Tata Consultancy Services Ltd': 4.17,
        'Infosys Ltd': 6.07,
        'HDFC Bank Ltd': 11.20,
        'ICICI Bank Ltd': 7.76,
        'State Bank of India': 2.64,
        'Hindustan Unilever Ltd': 2.68,
        'ITC Ltd': 4.49,
        'Larsen & Toubro Ltd':3.84,
        'Reliance Industries Ltd': 9.89,
        'Bharti Airtel Ltd': 2.63}

    total_return = {'Tata Consultancy Services Ltd': 19.70,
        'Infosys Ltd': 9.90,
        'HDFC Bank Ltd': -11.60,
        'ICICI Bank Ltd': 25.35,
        'State Bank of India': 46.71,
        'Hindustan Unilever Ltd': -5.16,
        'ITC Ltd':19.20,
        'Larsen & Toubro Ltd':63.59,
        'Reliance Industries Ltd':40.04,
        'Bharti Airtel Ltd': 44.99}

    contribution_return_port = {}
    for key in port_weights:
        contribution_return_port[key] = port_weights[key] * total_return[key]
    st.markdown('**Contribution to return - PORT**')
    st.text(sorted(contribution_return_port.items(), key=lambda x: x[1]))

    contribution_return_bench = {}
    for key in benchmark_weights:
        contribution_return_bench[key] = benchmark_weights[key] * total_return[key]
    st.markdown('**Contribution to return - BENCH**')
    st.text(sorted(contribution_return_bench.items(), key=lambda x: x[1]))

    





        






