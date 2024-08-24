import os
import yfinance as yf

from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from datetime import datetime


import streamlit as st

# Configurar a variável de ambiente para a chave da API
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Função para buscar o preço das ações
def fetch_stock_price(ticker):
    stock = yf.download(ticker, start="2023-08-08", end="2024-08-08")
    return stock

# Criando a ferramenta do Yahoo Finance
yahoo_finance_tool = Tool(
    name="Yahoo Finance Tool",
    description="Busca preços de ações do ticker {ticker} de um período específico",
    func=lambda ticker: fetch_stock_price(ticker)
)

# Importando OpenAI LLM - GPT
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Configuração do agente de análise de preços de ações
stockPriceAnalyst = Agent(
    role="Senior stock price Analyst",
    goal="Find the {ticker} stock price and analysis trends",
    backstory="You're a highly experienced in analyzing the price of a specific stock and make predictions about its future price.",
    llm=llm,
    verbose=True,
    max_iter=5,
    memory=True,
    tools=[yahoo_finance_tool],
    allow_delegation=False
)

# Definição da tarefa de análise de preços
getStockPrice = Task(
    description="Analyze the stock {ticker} price history and create a trend analysis of up, down or sideways",
    expected_output="Specify the current trend stock price - up, down or sideways",
    agent=stockPriceAnalyst
)

# Configuração da ferramenta de busca
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)

# Configuração do agente de análise de notícias
newsAnalyst = Agent(
    role="Senior stock news Analyst",
    goal="Create a short summary of the market news related to the stock {ticker} company. Specify the current trend - up, down and sideways with the news context. For each request stock specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed",
    backstory="You're a highly experienced in analyzing the market trends and news and have tracked assets for more than 10 years. You're also a master-level analyst in the traditional market and have a deep understanding of human psychology. You understand news, their titles and information, but you look at those with a healthy dose of skepticism. You consider also the source of the news articles.",
    llm=llm,
    verbose=True,
    max_iter=5,
    memory=True,
    tools=[search_tool],
    allow_delegation=False
)

# Definição da tarefa de obtenção de notícias
getNews = Task(
    description="""Take the stock and always include BTC to it (if not requested). Use the search tool to search each one individually.
    The current date is {current_date}.
    Compose the results into a helpful report
    """,
    expected_output="""A summary of the overall market and one sentence summary for each requested asset. 
    Include a fear/greed score for each asset based on the news. Use format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>
    """,
    agent=newsAnalyst
)

# Configuração do agente de escrita de análises de ações
stockAnalystWriter = Agent(
    role="Senior stock Analyst Writer",
    goal="Analyze the trends price and news and write an insightful, compelling, and informative 3-paragraph long newsletter based on the stock",
    backstory="You're widely accepted as the best stock analyst in the market. You understand complex concepts and create compelling stories and narratives that resonate with wider audiences. You understand macro factors and combine multiple theories - e.g., cycle theory and fundamental analyses. You're able to hold multiple opinions when analyzing anything",
    llm=llm,
    verbose=True,
    max_iter=5,
    memory=True,
    tools=[yahoo_finance_tool],
    allow_delegation=True
)

# Definição da tarefa de escrita de análises
writeAnalyses = Task(
    description="""Use the stock price trend and the stock news report to create an analysis and write the newsletter about the {ticker} company that is brief and highlights the most important points. Focus on the stock price trend, news, and fear/greed score. What are the near future considerations? Include the previous analysis of stock trend and news summary.
    """,
    expected_output="""An eloquent 3-paragraph newsletter formatted as markdown in an easy readable manner. It should contain:
    - 3 bullet points executive summary
    - Introduction - set the overall picture and spike up the interest
    - Main part provides the core of the analysis including the news summary and fear/greed scores
    - Summary - key facts and concrete future trend prediction - up, down or sideways. 
    """,
    agent=stockAnalystWriter,
    context=[getStockPrice, getNews]
)

# Configuração do Crew
crew = Crew(
    agents=[stockPriceAnalyst, newsAnalyst, stockAnalystWriter],
    tasks=[getStockPrice, getNews, writeAnalyses],
    verbose=True,
    process=Process.sequential,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15
)

with st.sidebar:
    st.header('Enter the stock ticker')

    with st.form(key='research form'):
        topic = st.text_input("Select the ticker")
        submit_button = st.form_submit_button(label = "Run research")

if submit_button:
    if not topic:
        st.error("Please fill the ticker field")
    else:
        results=crew.kickoff({
            "ticker": topic,
            "current_date": datetime.now().strftime("%Y-%m-%d")
        })

        st.subheader("Result of your research;")
        st.write(results['final_output'])
