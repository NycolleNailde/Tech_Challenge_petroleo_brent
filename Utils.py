
import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.service import Service # Classe Service
from selenium.webdriver.common.by import By
# from selenium.common.exceptions import WebDriverException

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

        new_df = pd.DataFrame(dados, columns=['Data', 'Preço - petróleo bruto - Brent (FOB)'])
        new_df['Data'] = pd.to_datetime(new_df['Data'], format='%d/%m/%Y')
        new_df = new_df.sort_values(by='Data',ascending=True)

        return new_df

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




 