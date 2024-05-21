
import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from selenium import webdriver
from selenium.webdriver.chrome.service import Service # Classe Service
from selenium.webdriver.common.by import By

def webscraping_ipea(url):

    try:

        # Instacia uma classe Service que será usada para iniciar uma instância do Chrome 
        service = Service()

        # Defini a preferência para o browser do Chrome
        options = webdriver.ChromeOptions()
        options.add_argument('headless')

        # Iniciando uma instância do Chrome WebDriver
        driver = webdriver.Chrome(service=service, options=options) 

        # Abrindo a url na minha instância do Chrome

        driver.get(url)
        table = driver.find_elements(By.ID,'grd_DXMainTable')[0].text
        rows = table.split("\n")
        rows = rows[2:]

        dados = []

        for row in rows:
            data, preco = row.split(" ")
            dados.append((data, float(preco.replace(',', '.'))))

        df = pd.DataFrame(dados, columns=['Data', 'Preço - petróleo bruto - Brent (FOB)'])
        df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')
        df = df.sort_values(by='Data',ascending=True)

        return df

    except:
        print('Falha ao acessar a página. Confira a url!')
        return pd.DataFrame(columns = ['Data'])

def update_dataipea(url):

    new_df = webscraping_ipea(url)

    try:
        existing_df = pd.read_csv('./data/df_ipea.csv')
    except FileNotFoundError:
        new_df.to_csv('./data/df_ipea.csv', index = False)
        existing_df = new_df

    existing_df['Data'] = pd.to_datetime(existing_df['Data'],format='%Y-%m-%d')
    new_df['Data'] = pd.to_datetime(new_df['Data'],format='%Y-%m-%d')
    
    # Encontra a data mais recente no DataFrame existente
    last_date = existing_df['Data'].max()
    
    # Filtra as novas linhas que são mais recentes do que a última data
    new_rows = new_df[new_df['Data'] > last_date]

    # Atualiza o arquivo se houver novas linhas
    if not new_rows.empty:
        new_df = pd.concat([existing_df, new_rows], ignore_index=True)
        new_df.to_csv('./data/df_ipea.csv', index = False)
        print('Atualizou!')
    else:
        print('Não atualizou!')

def create_lag_feature(input):
    # Criação de lag features
    for lag in range(1, 3):  # Criar atrasos de 1 dia até 2 dias
        input[f'Preço_lag_{lag}'] = input['Preço - petróleo bruto - Brent (FOB)'].shift(lag)

    input = input.dropna()
    x = input[['Preço_lag_1','Preço_lag_2']].values
    return x

def prev_week_gradient(modelo,x):
    # Fazer previsões para a próxima semana usando os últimos dados conhecidos
    input = create_lag_feature(x)

    last_known_data = input[-1].reshape(1, -1)
    last_data = x.iloc[-1:,0].iloc[0]
    today = datetime.now()
    
    business_days = pd.date_range(start=last_data, end=today, freq='B')
    h = len(business_days) + 5

    next_week_predictions = []
    for _ in range(h):  # para cada dia da próxima semana
        next_day_pred = modelo.predict(last_known_data)[0]
        next_week_predictions.append(next_day_pred)
        last_known_data = np.roll(last_known_data, -1)
        last_known_data[0, -1] = next_day_pred
    
    return next_week_predictions

def plot_gradient(current_week_dates, current_week_prices,next_week_dates, next_week_predictions):
    
    current_week_dates = [date.strftime('%d/%m/%y') for date in current_week_dates]
    next_week_dates = [date.strftime('%d/%m/%y') for date in next_week_dates]
  
    fig = make_subplots()
    
    # Adicionar trace dos preços atuais
    fig.add_trace(go.Scatter(
        x=current_week_dates,
        y=current_week_prices,
        mode='lines+markers',
        name='Preços Reais',
        line=dict(color='blue'),
        marker=dict(color='blue')
    ))

    # Adicionar trace das previsões
    fig.add_trace(go.Scatter(
        x=next_week_dates,
        y=next_week_predictions,
        mode='lines+markers',
        name='Previsões para a Próxima Semana',
        line=dict(color='red', dash='dash'),
        marker=dict(color='red')
    ))

    # Atualizar layout do gráfico
    fig.update_layout(
        # title='Histórico Recente Preço e Previsões de Preço para a Próxima Semana',
        yaxis_title='Preço (US$)',
        xaxis=dict(
            tickangle=-45
        ),
        legend=dict(
            x=0,
            y=1,
            traceorder='normal'
        )
    )

    # Atualizar eixos x e y para exibir o gráfico corretamente
    fig.update_xaxes(type='category')
    fig.update_yaxes(type='linear')

    data = {'Data': next_week_dates, 'Preço (US$)': next_week_predictions}
    resultado = pd.DataFrame(data)

    return [resultado, fig]   

def prev_week_prophet(modelo):
    last_date = datetime.strptime('2024-05-03', '%Y-%m-%d')

    # Obter a data atual
    today = datetime.now()

    # Criar um range de datas úteis entre a data atual e a data específica
    business_days = pd.date_range(start=last_date, end=today, freq='B')

    # Calcular a quantidade de dias úteis e adicionar 5
    h = len(business_days) + 5

    future = modelo.make_future_dataframe(periods=h,freq='B')

    forecast = modelo.predict(future)
    fig_prophet = modelo.plot(forecast)

    previsoes = forecast.rename(columns = {'ds':'Dia', 'yhat':'Preço (US$)'})
    previsoes['Dia'] = previsoes['Dia'].dt.strftime('%d/%m/%y')
    previsoes = previsoes[['Dia','Preço (US$)']].tail(h)
    
    return [previsoes, fig_prophet]

def wmape(y_true, y_pred):
  return np.abs(y_true-y_pred).sum() / np.abs(y_true).sum()



 