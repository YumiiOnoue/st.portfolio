import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import joblib


st.set_page_config(page_title="Erica Yumi Onoue - Portfólio Analista de Dados",
                    page_icon="📊",
                    layout="wide")

# --- Carrega o CSS ---
css_path = os.path.join(os.path.dirname(__file__), "style", "main.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -- Link para carregar os ícones do Font Awesome -- 
    st.markdown("""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
        """, unsafe_allow_html=True)


# ============= Layout no Streamlit ===============

# --- Início ---
def pagina_inicio():
    st.title("Bem vindo(a) ao meu portfólio!")
    col1, col2 = st.columns(2, gap="small")
    with col1:
        url = "https://i.postimg.cc/Rhj9Cg5Y/perfil-erica.jpg"
        st.image(url, width=200)
        st.markdown("📄[Visualizar Currículo](https://drive.google.com/file/d/11WRTjxHsgX6m_YtW0NkLyb8B-U3XFrrE/view?usp=sharing)", unsafe_allow_html=True)

    with col2:
        st.markdown('''
                    <style>
                    .no-spacing-h3 {
                        line-height: 1; 
                        margin: 0;    
                        padding: 0;
                    }
                    </style>
                    <h3 class="no-spacing-h3">
                    Eu sou Erica Yumi Onoue <br>
                    Analista de Dados 
                    </h3>
                    ''',
                    unsafe_allow_html=True)
        st.markdown('''Estou em transição de carreira para a área de Dados, trazendo comigo mais de quatro 
                    anos de experiência como auxiliar administrativo e uma sólida formação em Economia, 
                    com graduação e mestrado na área. 
                    ''')
        
    st.markdown('-------------') 
    st.markdown('### Sobre mim')
    st.markdown('''
                Desde a faculdade, desenvolvi interesse por dados, 
                tendo elaborado projetos de análise sobre o setor cultural e a importação de produtos farmacêuticos.
                Durante minha trajetória profissional, tive a oportunidade de aplicar a análise de dados 
                no dia a dia da empresa: criei dashboards por tipo de entrega, produzi relatórios de vendas 
                e estoque, além de propor sugestões de produtos e promoções com base em dados. Essas 
                experiências reforçaram meu desejo de migrar definitivamente para o campo da análise de dados.
                ''')
    st.markdown('''                
                Atualmente, venho me dedicando ao desenvolvimento das competências essenciais para atuar 
                como Analista de Dados, com foco em **Análise e Manipulação de Dados**, **Python**, **SQL**, **Power BI** 
                e ferramentas de controle de versão (Git/GitHub). Estou em constante aprendizado e busco 
                aplicar esse conhecimento em projetos práticos que resolvam problemas reais de negócio.
                ''')
    

# --- Habilidades ---    
    st.markdown('---------')
    st.markdown("### Habilidades Técnicas")
    st.markdown("""
    <div style='display: flex; flex-wrap: wrap; gap: 8px;'>

    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>Python</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>Pandas</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>NumPy</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>SQL</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>Power BI</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>Seaborn</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>Matplotlib</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>Scikit-learn</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>Git & GitHub</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>Jupyter Notebook</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>VS Code</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>Estatística</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>Excel</span>
    </div>
    """, unsafe_allow_html=True)


# --- Breve descrição dos Projetos ---
    st.markdown('-------------') 
    with st.container():
            st.markdown('### Projetos')
            st.markdown('Aqui você encontrará uma breve descrição de cada projeto, ok?')
            st.markdown('⬅️ Para verificar os projetos completos é só clicar no menu lateral à esquerda ou no botão logo abaixo.')
            st.markdown('---------')
            
            tab1, tab2, tab3 = st.tabs(["📊 Dashboard de Vendas", "🔍 Análise Exploratória de Dados", "🔮 Modelo Preditivo"])
            with tab2:
                st.markdown('#### Análise Exploratória de um Marketplace')
                st.markdown('''
                    * **Objetivo:** objetivo identificar pontos chaves da empresa para que o CEO possa 
                            entender melhor o negócio e conseguir tomar decisões estratégicas.
                    * **Metodologia:** Python, manipulação de dados, visualização de dados em Streamlit e 
                        versionamento com Github.
                    * **Resultados:** um painel gerencial com as principais métricas da empresa.
                            ''')
                if st.button('Visualizar projeto completo', key='projeto_2'):
                    projeto_2() 


            with tab3:    
                st.markdown('#### Modelo Preditivo dos imóveis de Boston')
                st.markdown(''' 
                    * **Objetivo:** fazer uma previsão dos valores dos imóveis de Boston.
                    * **Metodologia:** foi utilizado a Regressão Linear e considerando MEDV 
                        (valor médio de casas ocupadas) a variável dependente.                   
                    * **Resultados:** um painel interativo, onde o usuário pode alterar as características do imóvel
                            para verificar o preço estimado. 
                            ''')
                if st.button('Visualizar projeto completo', key='projeto_3'):
                    projeto_3()

            with tab1:
                st.markdown('#### Dashboard de Vendas Cafeteria')
                st.markdown('''
                    * **Objetivo:** Criar um painel gerencial em Power BI para insights das vendas.
                    * **Metodologia:** os dados foram analisados no jupyter notebook para obter cálculos estatísticos e
                            utilizado o Power BI para gerar gráficos interativos com os principais KPIs de vendas.
                    * **Resultado:** a empresa apresenta receita crescente nas suas três lojas. Sendo, a categoria Café e Chá os
                            mais comercializados.
                            ''')
                if st.button('Visualizar projeto completo', key='projeto_1'):
                    projeto_1() 

# ---- Contato ------
    st.markdown('-------------')
    st.markdown('### Contato')
    st.markdown('Obrigada por visitar meu portfólio! 💙')
    st.markdown('''
                Se você quiser conversar sobre oportunidades, tirar dúvidas sobre meus projetos ou apenas 
                bater um papo sobre dados e tecnologia, estou à disposição.
                ''')
    st.markdown("""
        <div style="display: flex; gap: 30px; font-size: 18px; align-items: center;">
            <a href="https://linkedin.com/in/ericayumionoue" target="_blank" style="text-decoration: none; color: inherit;">
                <i class="fab fa-linkedin"></i> LinkedIn
            </a>
            <a href="https://github.com/YumiiOnoue" target="_blank" style="text-decoration: none; color: inherit;">
                <i class="fab fa-github"></i> GitHub
            </a>
            <a href="mailto:eyumiio@gmail.com" style="text-decoration: none; color: inherit;">
                <i class="fas fa-envelope"></i> E-mail
            </a>
        </div>
        """, unsafe_allow_html=True)
    
# ============= Função dos Projetos ===============
# ================================================= 
# =================  Projeto 1 ==================== 
def projeto_1():
    st.title('Dashboard de Vendas')
    st.markdown('## Cafeteria ☕')
    st.markdown("""
    <div style='display: flex; flex-wrap: wrap; gap: 8px;'>
    
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>Python</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>Pandas</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>NumPy</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>Power BI</span>
    
    </div>
    """, unsafe_allow_html=True)

    st.markdown('### Objetivo')
    st.markdown('''
            Este projeto tem como objetivo realizar uma análise exploratória dos dados de vendas de uma 
                cafeteria, com foco em fornecer ao CEO uma visão abrangente sobre:

            * As receitas totais e as quantidades vendidas;
            * O desempenho individual de cada loja;
            * Os produtos mais rentáveis.    
            
            A análise visa apoiar decisões estratégicas baseadas em dados, identificando oportunidades de 
            crescimento e otimizando o mix de produtos.    
                ''')
    st.markdown('### Metodologia')
    st.markdown('''
            * Análise descritiva em Power BI
            * Cálculos estatísticos em Python
                ''')
    st.markdown("""
    <a href="https://github.com/YumiiOnoue/Coffee_shop_sales_EDA/tree/main" target="_blank">
        <button style="
            background-color: #3b9ecf;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            font-size: 12px;
            cursor: pointer;
            transition: background-color 0.3s;
        ">
            <i class="fab fa-github"></i> Ver código no GitHub
        </button>
    </a>
    """, unsafe_allow_html=True)    
    st.markdown('### Visão Geral dos Dados')
    st.markdown('''
            * O período da análise é de janeiro a junho de 2023.                
            * O banco de dados possui 149.116 transações de três lojas (Hell's Kitchen, Astoria e Lower Manhattan).
            * A média de produtos por transação é de 1,44 e no máximo 8 produtos.
            * O desvio padrão da receita e do preço apresentaram dispersão alta.
            * O maior preço único é do café civete (US$45.00), que está muito acima da média, indicando outlier.
            ''')

    st.markdown('### Análise Gráfica')   
    st.markdown('A análise gráfica foi realizada no Power BI. Para interagir com o dashboard de vendas é só clicar [aqui](https://app.powerbi.com/view?r=eyJrIjoiYmMwMmQyNzgtMzJjMC00ZTViLThjNzAtYWRlODFhOGE0Y2E1IiwidCI6IjJlYjE0NDQ3LTQ0YWQtNDllZi04YjhmLTA5OWEzNTlhYjZkYSJ9).')
    st.markdown('Logo abaixo você terá o dashboard de forma estático. (Em atualização)')

    st.markdown('### Insights')
    st.markdown('''
            * Faturamento de US$698.81 mil, totalizando 214 mil vendas. 
            * Ticket médio de US$3.26 nos seis primeiros meses de 2023. 
            * Tanto a receita quanto as vendas aumentaram ao longo do período analisado.
            * As três lojas apresentaram receitas superiores a US$230 mil.
            * O horário de abertura, às 6h, é o de maior movimento, e o das 19h, o de menor fluxo.
            * As categorias de produto que mais contribuem para a receita das lojas são: café, chá e padaria. 
            * O produto líder nas três lojas é o Sustainably Grown Organic e os blends de chai tradicional.
            * A loja de Astoria opera duas horas a menos.
            * Todas as categorias apresentaram crescimento nas vendas, com destaque para café e chá.
            * A categoria padaria, terceira colocada no ranking de vendas, representando 74,7% a menos que o 
                café e 66,9% a menos que o chá.
            ''')

    st.markdown('### Conclusões') 
    st.markdown('''
            * As vendas das três lojas estão em crescimento, com destaque para a unidade de Hell's Kitchen. 
            * A unidade de Astoria tem desempenho inferior de vendas e menor tempo de funcionamento.
            * Pode ser interessante investir em promoções ou programas de fidelidade focados em produtos sustentáveis 
                e orgânico.  
            * A categoria padaria mostra potencial de crescimento e pode se beneficiar de um maior investimento estratégico.
            * Produtos como Chili Mayan, Caramel Syrup, Brazilian Organic, Chocolate Chip Biscotti e Almond Croissant 
                apresentam baixo desempenho e precisam de maior atenção estratégica, seja para reposicionamento, ajuste de preço ou revisão da oferta.
            ''')

# ================================================= 
# =================  Projeto 2 ==================== 
def projeto_2():
    st.title('Análise Exploratória de Dados')
    st.markdown('## Marketplace 🛍️')
    st.markdown("""
    <div style='display: flex; flex-wrap: wrap; gap: 8px;'>
    
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>Python</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>Pandas</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>NumPy</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>Streamlit</span>
    
    </div>
    """, unsafe_allow_html=True)
    st.markdown('### Objetivo')
    st.markdown('Esse projeto teve como objetivo identificar pontos chaves da empresa ' \
            'para que o CEO possa entender melhor o negócio e conseguir tomar as melhores decisões ' \
            'estratégicas.')
    
    st.markdown('### Visão Geral dos Dados')
    st.markdown('''
            * A análise foi realizada para 15 países.
            * Analisado mais de 4 milhões de avaliações.
            * 165 tipos de culinária.
            * Foram utilizadas três tipos de visões da empresa: por país, por cidade e por restaurante
                ''')        
    st.markdown("""
    <a href="https://github.com/YumiiOnoue/projeto_fome_zero" target="_blank">
        <button style="
            background-color: #3b9ecf;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            font-size: 12px;
            cursor: pointer;
            transition: background-color 0.3s;
        ">
            <i class="fab fa-github"></i> Ver código no GitHub
        </button>
    </a>
    """, unsafe_allow_html=True)   
    st.markdown('### Resultados')
    st.markdown('''
            * A Índia representa grande parcela dos restaurantes cadastrados na plataforma.
            * Muitos dos restaurantes não apresentam a opção de pedido online e entregas.
            * O dashboard criado dessa análise está nesse [link](https://eyo-projeto-fome-zero.streamlit.app/).
                ''')

    st.markdown('### Conclusões') 
    st.markdown('Após analisar os dados, pode-se observar que são poucos os países que utilizam a plataforma e a ' \
            'maioria dos restaurantes cadastrados estão na Índia.')

# ================================================= 
# =================  Projeto 3 ==================== 
def projeto_3():
    st.title('Modelo Preditivo')
    st.markdown('## Imóveis de Boston 🏢')
    st.markdown("""
    <div style='display: flex; flex-wrap: wrap; gap: 8px;'>

    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>Python</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>Pandas</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>NumPy</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>Matplotlib</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>Seaborn</span>
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>Sklearn</span>            
    <span style='background-color:#7e5bef; color:white; padding:6px 14px; margin:4px; border-radius:20px; font-size:14px; display:inline-block;'>Streamlit</span>

    </div>
    """, unsafe_allow_html=True)    
    st.markdown('### Objetivo')
    st.markdown('O objetivo principal é criar um modelo que estime o valor de um imóvel em Boston considerando diversos atributos.')
    st.markdown('### Metodologia')    
    st.markdown('''
                * Os dados foram coletadas pelo U.S Census Service e estão disponíveis nesse [link](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html).
                * Para essa análise foram utilizados 14 atributos.
                * Para prever os valores dos imóveis, foi utilizado a Regressão Linear e considerando MEDV (valor médio de casas 
                ocupadas) a variável dependente.
                * Os gráficos foram feitos com seaborn e matplotlit.
                * Para regressão e métricas de validação foi utilizado o sklearn.
                ''')
    
    st.markdown('### Análise Exploratória dos dados')
    
    #---- Carregar dados -------------
    @st.cache_data
    def load_data():
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
        columns = [
            'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
        ]
        df = pd.read_csv(url, delim_whitespace=True, names=columns)
        return df

    df = load_data()

    with st.expander('🔍 Visualizar dados'):
        st.dataframe(df.head())

    st.markdown('##### Dispersão entre Número de Quartos (RM) e Preço (MEDV)')
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=df, x='RM', y='MEDV', ax=ax1)
    st.pyplot(fig1)

    st.markdown('##### Mapa de Correlação entre Variáveis')
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    st.markdown('--------')
    tab1, tab2 = st.tabs(["📊 Métricas de Validação", "📈 Previsão de Preço"])
    with tab1:
        nomes_extenso = {
            'CRIM': 'CRIM - Taxa de criminalidade',
            'ZN': 'ZN - Proporção de terrenos residenciais',
            'INDUS': 'INDUS - Proporção de área industrial',
            'CHAS': 'CHAS - Fronteira com rio Charles',
            'NOX': 'NOX - Óxidos de nitrogênio',
            'RM': 'RM - Número médio de quartos',
            'AGE': 'AGE - Proporção de unidades antigas',
            'DIS': 'DIS - Distância para centros de emprego',
            'RAD': 'RAD - Acessibilidade a rodovias',
            'TAX': 'TAX - Imposto sobre propriedade',
            'PTRATIO': 'PTRATIO - Proporção aluno/professor',
            'B': 'B - Proporção de afro-americanos',
            'LSTAT': 'LSTAT - Baixo status socioeconômico'
        }
        st.markdown('### Métricas de Validação')
        # Lista de variáveis disponíveis para o usuário escolher (exceto a variável alvo)
        variaveis = [col for col in df.columns if col != 'MEDV']
        variaveis_escolhidas = st.multiselect("Escolha as variáveis independentes (X):", variaveis, default=variaveis)

        # Separar variáveis
        if variaveis_escolhidas:
            X = df[variaveis_escolhidas]
            y = df['MEDV']

            # Treinar modelo de regressão linear
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            modelo = LinearRegression()
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

            # Coeficientes
            st.markdown("#### Coeficientes da Regressão")
            coef_df = pd.DataFrame({
                'Variável': [nomes_extenso.get(col, col) for col in X.columns],
                'Coeficiente': modelo.coef_
            })
            st.dataframe(coef_df)
            st.write(f"**Intercepto**: {modelo.intercept_:.2f}")

            # Mostrar métricas de validação
            st.markdown("#### Resultado da Validação")
            st.write(f"**Erro Absoluto Médio** (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
            st.write(f"**Erro Quadrático Médio** (MSE): {mean_squared_error(y_test, y_pred):.2f}")
            st.write(f"**Raiz do Erro Quadrático Médio** (RMSE): {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
            st.write(f"**Coeficiente de Determinação** (R²): {r2_score(y_test, y_pred):.2f}")
        else:
            st.warning("Selecione ao menos uma variável para prosseguir.")

        st.markdown('#### Valores Reais vs Valores Previsto')
        df_result = pd.DataFrame({'Real': y_test, 'Previsto': y_pred})
        # Gráfico de dispersão real x previsto
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x='Real', y='Previsto', data=df_result, ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel('Preço Real (milhares US$)')
        ax.set_ylabel('Preço Previsto (milhares US$)')
        st.pyplot(fig)

    with tab2:
        #--- Previsão -----  
        st.markdown('### Previsão de Preços')
        @st.cache_data
        def load_data():
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
            columns = [
                'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
            ]
            df = pd.read_csv(url, delim_whitespace=True, names=columns)
            return df
        df = load_data()

        @st.cache_resource
        def train_model(df):
            X = df.drop('MEDV', axis=1)
            y = df['MEDV']
            modelo = LinearRegression()
            modelo.fit(X, y)
            return modelo

        modelo = train_model(df)

        # Entrada de dados
        st.markdown('Ajuste os valores das variáveis para prever o preço do imóvel:')
        st.markdown('')
        # Entradas do usuário
        crim = st.slider('Taxa de criminalidade (CRIM)', 0.0, 100.0, 0.1)
        zn = st.slider('Proporção de terrenos residenciais (ZN)', 0.0, 100.0, 0.0)
        indus = st.slider('Proporção de área industrial (INDUS)', 0.0, 30.0, 1.0)
        chas = st.selectbox('Faz fronteira com rio Charles? (CHAS) (0 = Não; 1 = Sim)', [0, 1])
        nox = st.slider('Concentração de óxidos de nitrogênio (NOX)', 0.3, 1.0, 0.5)
        rm = st.slider('Número médio de quartos (RM)', 3.0, 9.0, 6.0)
        age = st.slider('Proporção de unidades antigas (AGE)', 0.0, 100.0, 50.0)
        dis = st.slider('Distância para centros de emprego (DIS)', 1.0, 12.0, 4.0)
        rad = st.slider('Acessibilidade a rodovias (RAD)', 1, 24, 1)
        tax = st.slider('Taxa de imposto sobre propriedade (TAX)', 100, 800, 300)
        ptratio = st.slider('Proporção aluno/professor (PTRATIO)', 10.0, 30.0, 18.0)
        b = st.slider('Proporção de afro-americanos (B)', 0.0, 400.0, 300.0)
        lstat = st.slider('% de status socioeconômico baixo (LSTAT)', 0.0, 40.0, 12.0)

        # Prever
        input_data = pd.DataFrame(
            [[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]],
            columns=[
                'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
            ]
        )

        if st.button('Prever Preço'):
            prediction = modelo.predict(input_data)[0]
            st.success(f'💰 Preço estimado: US${prediction * 1000:,.2f}')
 
    st.markdown('### Conclusões') 
    st.markdown(' 🚧 Em desenvolvimento 🔧')

# --- Barra Lateral ---
st.sidebar.title('Menu')
selection = st.sidebar.radio('Ir para:',
                            ['**Início**',
                            'Dashboard de Vendas',
                            'Análise Exploratória de Dados',
                            'Modelo Preditivo'])

if selection == '**Início**':
    pagina_inicio()
elif selection == 'Dashboard de Vendas':
    projeto_1()
elif selection == 'Análise Exploratória de Dados':
    projeto_2()
elif selection == 'Modelo Preditivo':
    projeto_3()


# --- Rodapé ---
st.markdown('-------------')
st.markdown('Desenvolvido com [Streamlit](https://streamlit.io) | © 2025 Erica Yumi Onoue.')