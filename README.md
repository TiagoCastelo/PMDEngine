# 🏘️ PMDEngine - Propriedade Machine Learning Engine

## 🎯 Objetivo do Projeto

O **PMDEngine** é um motor de processamento de dados e análise preditiva desenhado para recolher, processar e analisar listagens imobiliárias. Utiliza o framework Scrapy (com Playwright) para fazer web scraping incremental de sites de imobiliário e um motor de Machine Learning (ML) para classificar e prever o valor de imóveis.

O foco principal é identificar oportunidades de mercado, comparando o preço pedido com o preço estimado pelo modelo de ML.

## 🏗️ Arquitetura

O projeto está dividido em duas componentes principais:

1.  **MLEngine (Scrapy Crawler):** Responsável por recolher os dados de forma eficiente.
2.  **MLEngine (ML Pipeline/API):** Responsável pelo processamento, treino de modelos e serviço de predição.

### 1. Web Scraping (Scrapy)

* **Framework:** Scrapy (Python)
* **Aceleração/Renderização:** `scrapy-playwright` para lidar com sites JavaScript-heavy.
* **Spider Exemplo:** `remax_spider.py`
* **Persistência:** `PostgresPipeline` para salvar os dados numa base de dados PostgreSQL.
* **Otimização:** Implementa lógica TTL (Time-To-Live) para scraping incremental, evitando recolher páginas de detalhes de imóveis que não mudaram de preço ou que foram visitadas recentemente.

### 2. Machine Learning

* **Modelos:** Utiliza modelos pré-treinados (e.g., `apartamento_model.pkl`, `moradia_model.pkl`, `terreno_model.pkl`) para prever o preço de diferentes tipos de imóveis.
* **API:** O ficheiro `api/main.py` sugere uma interface para servir estas predições, provavelmente usando o FastAPI (padrão para `main.py` em APIs Python).
* **Processamento:** O módulo `common/processing.py` contém a lógica de pré-processamento de dados para garantir que os inputs para o modelo estão formatados corretamente.
* **Treino:** O diretório `ML_Training/` indica a existência de scripts para treinar e atualizar os modelos de ML.

## ⚙️ Configuração do Projeto

### Pré-requisitos

* Python 3.x
* Docker e Docker Compose (Recomendado para ambiente de produção/desenvolvimento)
* Playwright Browsers (instalados via `playwright install`)

### 1. Variáveis de Ambiente

Este projeto depende de variáveis de ambiente, particularmente para a base de dados e URLs iniciais de *crawling*.

Crie um ficheiro `.env` no diretório `MLEngine/docker/` (ou similar) com as seguintes variáveis:

| Variável | Descrição | Exemplo de Valor |
| :--- | :--- | :--- |
| `PGDATABASE` | Nome da base de dados PostgreSQL. | `imoveis_db` |
| `PGUSER` | Utilizador da base de dados. | `user` |
| `PGPASSWORD` | Palavra-passe da base de dados. | `mypassword` |
| `PGHOST` | Host da base de dados. | `localhost` ou nome do serviço Docker |
| `PGPORT` | Porta da base de dados. | `5432` |
| `START_URLS_LIST` | Lista JSON de URLs para iniciar o scraping. | `["https://www.remax.pt/imoveis/venda/apartamento?p=1"]` |

### 2. Instalação e Execução (Sem Docker)

1.  **Instalar dependências:**
    ```bash
    pip install -r MLEngine/requirements.txt
    playwright install
    ```

2.  **Configurar e Iniciar a Base de Dados:**
    Assegure-se de que o PostgreSQL está a correr e que a base de dados está criada (com a tabela `imoveis` esperada pelo `PostgresPipeline`).

3.  **Executar o Crawler:**
    ```bash
    cd MLEngine/src/
    # Certifique-se de que as variáveis de ambiente (DB e START_URLS_LIST) estão definidas antes de executar!
    scrapy crawl remax_imovel 
    ```

### 3. Execução (Com Docker)

O diretório `MLEngine/docker/` contém ficheiros para orquestração Docker:

1.  **Configurar `.env`** (conforme a secção acima).
2.  **Construir e Correr os Serviços:**
    ```bash
    cd MLEngine/docker/
    docker-compose up --build
    ```
    (Isto deve inicializar a base de dados, a aplicação Scrapy/ML, dependendo do seu `docker-compose.yml`).

## 📈 Machine Learning

O pipeline de ML pode ser executado para atualizar os modelos preditivos:

* **Treino do Modelo:** Consulte `ML_Training/treino_modelo.py`.
* **Encontrar Oportunidades:** Consulte `ML_Training/encontrar_oportunidades.py` para o script que usa os modelos para analisar os dados recolhidos.

---

Espero que este `README` seja um bom ponto de partida para a documentação do seu projeto! Quer que eu adicione mais alguma secção ou detalhe?
