import pandas as pd
import numpy as np
from datetime import datetime

from Utils import update_dataipea

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import plotly.subplots as sp

import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsforecast import StatsForecast

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsforecast.models import AutoARIMA


import streamlit as st

# Chama função que atualiza base de dados
url = 'http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view'
update_dataipea(url)
# Lendo base
df_ipea = pd.read_csv('./data/df_ipea.csv')

# Título da Página
st.title("Preço por barril do petróleo bruto Brent (FOB)")

# Abas do app
tab0, tab1, tab2, tab3 = st.tabs(['Geral','Previsões - Gradient Boosting', 'Previsões - AutoARIMA', 'Conclusão'])

# Separando as Tabs
with tab0:
    '''
    ## Petróleo Brent
    '''
    df = pd.DataFrame(df_ipea)
    df = df.rename(columns={'Preço - petróleo bruto - Brent (FOB)':'Preço (U$)'})
    st.dataframe(df,use_container_width=True)

        # Plotar o gráfico de linha usando Plotly Express
    fig = px.line(df, x='Data', y='Preço (U$)', title='Preço do Petróleo Brent ao longo do tempo')

    # Exibir o gráfico no Streamlit
    st.plotly_chart(fig)
    '''

    ## Histórico
    ### Ascensão do preço após 2004:

    Crescimento econômico global:
    Durante esse período, houve um forte crescimento econômico global, especialmente nos países em 
    desenvolvimento, como China, Índia e Brasil. O aumento da atividade econômica resultou em uma 
    maior demanda por energia, incluindo petróleo, o que contribuiu para o aumento dos preços.


    Além disso, também houve o aumento da demanda por petróleo, implusionado pelo rápido crescimento 
    econômico em países emergentes, juntamente com o aumento da industrialização e urbanização, levou 
    a um aumento significativo na demanda por petróleo. Isso colocou pressão adicional sobre os 
    suprimentos globais e contribuiu para a alta dos preços.

    ### Queda em 2008:

    
    Um dos eventos mais significativos que impactou o preço do petróleo em 2008 foi a crise financeira global 
    que eclodiu na segunda metade daquele ano. A crise financeira teve origem nos Estados Unidos com a crise 
    do mercado imobiliário e a falência do banco de investimento Lehman Brothers em setembro de 2008.

    A queda na demanda global por petróleo, combinada com preocupações sobre o excesso de oferta, levou a uma 
    queda significativa nos preços do petróleo. No segundo semestre de 2008, o preço do petróleo bruto atingiu 
    máximas históricas em torno de 'U$147' por barril, antes de cair drasticamente para menos de 'U$40' por barril no 
    final do ano, refletindo a rápida deterioração das condições econômicas globais.

    ### Queda 2014

    Após essa crise o preço se recuperou com certa estabilidade ente 2011 e 2014. Em 2014

    ### Queda 2020

    ...

    ### Alta em Março 2022
    
    Base de Dados: 
    
    http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view

    Links relevantes:
    [1] https://brsa.org.br/wp-content/uploads/wpcf7-submissions/1016/1%C2%BA-arquivo.pdf
    [2] https://g1.globo.com/economia/noticia/2015/01/entenda-queda-do-preco-do-petroleo-e-seus-efeitos.html
    https://www.dw.com/pt-br/oito-motivos-para-a-queda-do-pre%C3%A7o-do-petr%C3%B3leo/a-19051686
    https://www.cnnbrasil.com.br/economia/precos-de-petroleo-caem-de-maxima-de-2014-e-preocupacao-com-oferta-limita-perdas/
    https://www.poder360.com.br/economia/pandemia-faz-preco-do-barril-de-petroleo-fechar-ano-20-mais-barato/#:~:text=O%20barril%20do%20petr%C3%B3leo%20encerrou,US%24%2066%2C00).
    https://economia.uol.com.br/noticias/afp/2020/12/31/o-preco-do-petroleo-fecha-2020-com-uma-queda-de-mais-de-20-devido-a-covid-19.htm
    https://einvestidor.estadao.com.br/investimentos/preco-petroleo-2020/
    https://www.cnnbrasil.com.br/economia/precos-do-petroleo-caem-us10-barril-em-maior-queda-diaria-desde-abril-de-2020/
    https://oglobo.globo.com/economia/preco-do-petroleo-fecha-2020-com-uma-queda-de-mais-de-20-devido-pandemia-de-covid-19-1-24818960
    https://g1.globo.com/economia/noticia/2020/03/09/o-que-explica-o-tombo-do-preco-do-petroleo-e-quais-os-seus-efeitos.ghtml


    
    '''


