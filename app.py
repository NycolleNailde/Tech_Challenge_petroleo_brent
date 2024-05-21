import pandas as pd

from Utils import webscraping_ipea
from Utils import create_lag_feature, prev_week_gradient,plot_gradient
from Utils import prev_week_prophet

import plotly.express as px
import streamlit as st
import pickle

# EXTRACT: 

# webscraping dos dados mais recentes, para input dos modelos
url = 'http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view'

df_ipea = webscraping_ipea(url)
df_ipea['Data'] = pd.to_datetime(df_ipea['Data'],format='%Y-%m-%d')
X_60= df_ipea.iloc[-60:,:]
X_5= df_ipea.iloc[-5:,:]

# Leitura dos artefatos com modelos treinados
modelo_gradient = pickle.load(open('modelo_gradientboosting.pkl','rb'))
modelo_prophet = pickle.load(open('modelo_prophet.pkl','rb'))

# TRANSFORM: fazendo previsões

# GradienteBoosting
# input_grad = create_lag_feature(X_5)
output_gr = prev_week_gradient(modelo_gradient,X_5)

# FIGURA
# As datas correspondentes à próxima semana
h = len(output_gr)
next_week_dates = pd.date_range(X_5['Data'].iloc[-1], periods=h+1,freq='B')[1:]
next_week_dates = pd.to_datetime(next_week_dates,format='%Y-%m-%d')

# Selecionar os dados da semana atual (últimos 7 dias do dataset)
current_week_dates = X_60.iloc[-7:,0]
current_week_prices = X_60.iloc[-7:,1]

resultado_gr, fig_gradient = plot_gradient(current_week_dates, current_week_prices, next_week_dates, output_gr)


# Prophet

resultado_pr, fig_prophet= prev_week_prophet(modelo_prophet)


# LOAD: streamlit
# Título da Página
st.title("Preço por barril do petróleo bruto Brent (FOB)")

# Abas do app
tab0, tab1, tab2= st.tabs(['Geral','Previsões - Prophet', 'Previsões - Gradient Boosting'])

# Separando as Tabs
with tab0:
    '''
    ## Petróleo Brent
    '''
    df = pd.DataFrame(df_ipea)
    df = df.rename(columns={'Preço - petróleo bruto - Brent (FOB)':'Preço (US$)'})
    df_table = df.copy()
    df_table['Data'] = df_table['Data'].dt.strftime('%d/%m/%y')
    st.dataframe(df_table,use_container_width=True)
    '''
    Fonte: [IPEA](http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view).
    '''
    

    # Plotar o gráfico de linha usando Plotly Express
    fig = px.line(df, x='Data', y='Preço (US$)', title='Preço do Petróleo Brent ao longo do tempo')

    # Exibir o gráfico no Streamlit
    st.plotly_chart(fig)
    '''

    ## Análise do Histórico
    ### Ascensão do preço após 2004

    Durante esse período, houve um forte crescimento econômico global, especialmente nos países em 
    desenvolvimento, como China, Índia e Brasil. Com o aumento da demanda por petróleo, implusionado 
    pelo rápido crescimento econômico em países emergentes, juntamente com o aumento da industrialização 
    e urbanização, levou a um aumento significativo na demanda por petróleo. Isso colocou pressão adicional 
    sobre os suprimentos globais e contribuiu para a alta dos preços.

    ### Queda em 2008

    
    Um dos eventos mais significativos que impactou o preço do petróleo em 2008 foi a crise financeira global 
    que eclodiu na segunda metade daquele ano. A crise financeira teve origem nos Estados Unidos com a crise 
    do mercado imobiliário e a falência do banco de investimento Lehman Brothers em setembro de 2008.

    Essa crise espalhou-se pelo mundo, culminando em uma recessão global em 2009. Esse colapso financeiro levou 
    à nacionalização de bancos, derrubou governos, gerou altas taxas de desemprego e desencadeou protestos. Esses
     eventos fizeram com que o preço do petróleo Brent caísse drasticamente em 2008.

    ### Queda 2014

    Passada a crise de 2008, o preço se recuperou com certa estabilidade entre 2011 e 2014. Contudo, em 2014, o preço do petróleo 
    Brent caiu devido a uma combinação de superprodução e demanda mais fraca do que o esperado na Europa e na 
    Ásia. 
    
    O aumento da produção de petróleo de xisto nos EUA contribuiu significativamente para o excesso de 
    oferta. Em novembro de 2014, a OPEP recusou-se a reduzir sua produção, exacerbando a queda dos preços, com a 
    intenção de manter sua participação de mercado e inviabilizar a produção de rivais, especialmente produtores 
    norte-americanos de petróleo de xisto.

    ### Queda 2020

    A pandemia de COVID-19 causou uma queda acentuada na demanda por petróleo devido ao isolamento social e à redução 
    das atividades econômicas globais. Em março de 2020, a guerra de preços entre a Rússia e a Arábia Saudita 
    também contribuiu para a queda vertiginosa dos preços. 
    
    Em abril, o preço atingiu seu valor mínimo na história, devido à falta de armazenamento e compradores. Embora os 
    preços tenham se recuperado posteriormente, eles não retornaram aos níveis pré-pandêmicos até 2022.

    ## Metodologia

    A série temporal dos preços do petróleo Brent é notoriamente volátil e suscetível a uma variedade de fatores externos, 
    como crises financeiras, mudanças na produção global de petróleo e eventos geopolíticos significativos. Dada essa 
    complexidade, a previsão precisa dos preços do petróleo requer modelos que possam capturar não apenas tendências 
    e sazonalidades, mas também responder a variações abruptas e não lineares nos dados. Por isso, esse deploy apresenta
    duas abordagens. Com uso dois modelos para a previsão desta série temporal: **Prophet** e **GradientBoostingRegressor**.

    
 
    Fontes:

    https://economia.uol.com.br/noticias/bbc/2021/10/10/crise-financeira-colapso-que-ameacou-o-capitalismo.htm


    https://g1.globo.com/economia/noticia/2015/01/entenda-queda-do-preco-do-petroleo-e-seus-efeitos.html

    
    https://www.poder360.com.br/economia/pandemia-faz-preco-do-barril-de-petroleo-fechar-ano-20-mais-barato/#:~:text=O%20barril%20do%20petr%C3%B3leo%20encerrou,US%24%2066%2C00).


    https://economia.uol.com.br/noticias/afp/2020/12/31/o-preco-do-petroleo-fecha-2020-com-uma-queda-de-mais-de-20-devido-a-covid-19.htm
    
    '''


with tab1:
    '''
    ## Prophet

    O Prophet é uma ferramenta de previsão de séries temporais desenvolvida pelo Facebook. 
    Ele é projetado para lidar com dados que possuem padrões sazonais e outliers, podendo 
    assim capturar padrões sazonais anuais, mensais, semanais ou diários. Além disso, essa
    ferramenta também é capaz de ajustar pontos de troca de tendência.


    O modelo foi treinado com dados, disponibilizados pelo [IPEA](http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view), 
    entre período entre 08/02/2024 até 03/05/2024, com sazonalidade semanal. 
    ```
    modelo = Prophet()
    modelo.add_seasonality(name='weekly', period=2, fourier_order=5)
    modelo.fit(treino.iloc[-60:,:])
    ```
    '''
    st.markdown('#### Histórico e Previsão de Preços do Petróleo Bruto')
    # fig_prophet.suptitle('Histórico e Previsão de Preços do Petróleo Bruto')
    fig_prophet.axes[0].set_xlabel('Data')
    fig_prophet.axes[0].set_ylabel('Preço (US$)')
    st.pyplot(fig_prophet, use_container_width=True)

    st.markdown('### Previsões')
    st.table(resultado_pr)
    '''
    ### Métricas de validação

    As métricas foram calculadas pelo conjunto de teste com dados do período 07/05/2024 a 13/05/2024.

    WMAPE: 1.38%  
    Mean Squared Error: 1.83  
    Mean Absolute Error: 1.15

    Vale ressaltar que essas métricas se alteram na medida em que os dados previstos estão mais distantes dos dados de treino.
    '''

with tab2:
    '''
    ## Gradient Boosting

    
    O GradientBoostingRegressor é um modelo de Machine Learning, baseado em árvores de decisão, que é usado para resolver problemas de regressão. 
    Ele pertence à família de modelos de boosting, que trabalham construindo uma sequência de modelos de forma iterativa, onde cada modelo 
    subsequente tenta corrigir os erros do modelo anterior. É adequado para capturar relações complexas e não lineares nos dados.

    O modelo foi treinado com toda a série histórica, disponibilizada pelo [IPEA](http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view) 
    até o dia 17/05/2024. Ou seja, período compreendido entre 20/05/1987 até 13/05/2024. 

    ```
    modelo = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, loss='squared_error')
    model.fit(X_train, y_train)
    ```
    
    '''
    st.markdown('#### Histórico Recente Preço e Previsões de Preço para a Próxima Semana')
    st.plotly_chart(fig_gradient, use_container_width=True)

    st.markdown('### Previsões')
    st.table(resultado_gr)

    '''
    ### Métricas de validação:

    As métricas foram calculadas para todo conjunto de treino, com dados do período de 20/05/1987 até 13/05/2024.

    WMAPE: 1.68%
    Mean Squared Error: 2.80
    Mean Absolute Error: 1.18

    Apesar de o Prophet apresentar métricas de desempenho superiores, o modelo GradientBoostingRegressor se destaca por sua robustez. 
    Isso se deve ao fato de que o GradientBoostingRegressor foi treinado com todos os dados históricos disponíveis da série temporal. 
    Ao utilizar o conhecimento adquirido durante o treinamento, o modelo faz previsões com base nas lag features dos dados mais recentes, 
    evitando assim o problema que pode ocorrer com o Prophet à medida que as previsões se distanciam dos dados de treinamento.
    '''