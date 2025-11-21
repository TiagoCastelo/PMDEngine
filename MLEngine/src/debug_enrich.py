import os
import pandas as pd
from sqlalchemy import create_engine, text
import ollama

# --- CONFIGURAÇÃO PARA RODAR NO WINDOWS (FORA DO DOCKER) ---
# Garante que estas credenciais batem certo com o teu docker-compose
DB_USER = 'user'     # Confirma no teu .env ou docker-compose
DB_PASSWORD = 'password' 
DB_HOST = 'localhost' 
DB_PORT = '5432'
DB_NAME = 'imoveis'

# URL do Ollama (localhost porque estás no Windows)
OLLAMA_HOST = 'http://localhost:11434'

def debug_run():
    print("1. [INIT] A iniciar diagnóstico...")
    
    # 1. Testar Ollama
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        print(f"2. [OLLAMA] A conectar a {OLLAMA_HOST}...")
        client.show('qwen2.5:7b')
        print("   [OLLAMA] ✅ Conectado e modelo encontrado!")
    except Exception as e:
        print(f"   [OLLAMA] ❌ Erro crítico: {e}")
        return

    # 2. Testar Base de Dados
    try:
        conn_str = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        print(f"3. [DB] A conectar ao Postgres: {DB_HOST}:{DB_PORT}/{DB_NAME}...")
        engine = create_engine(conn_str)
        with engine.connect() as conn:
            res = conn.execute(text("SELECT 1")).fetchone()
            print(f"   [DB] ✅ Conexão bem sucedida! Teste: {res[0]}")
            
            # Verificar quantos imoveis existem
            count_raw = conn.execute(text("SELECT COUNT(*) FROM imoveis")).scalar()
            print(f"   [DB] 📊 Total de imóveis na tabela 'imoveis': {count_raw}")
            
            # Verificar tabela AI
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS imoveis_ai_data (
                    imovel_id VARCHAR(255) PRIMARY KEY,
                    estado_conservacao INTEGER,
                    venda_urgente BOOLEAN,
                    potencial_investimento VARCHAR(50),
                    analisado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            count_ai = conn.execute(text("SELECT COUNT(*) FROM imoveis_ai_data")).scalar()
            print(f"   [DB] 📊 Total já analisados na 'imoveis_ai_data': {count_ai}")
            conn.commit()
    except Exception as e:
        print(f"   [DB] ❌ Erro na Base de Dados. Verifica user/pass e se o container postgres está 'healthy'.\n   Erro: {e}")
        return

    # 3. Simular a Query do Enricher
    print("4. [QUERY] A procurar o 'Delta' (o que falta analisar)...")
    query = """
        SELECT t1.url_id 
        FROM imoveis t1
        LEFT JOIN imoveis_ai_data t2 ON t1.url_id = t2.imovel_id
        WHERE t2.imovel_id IS NULL 
          AND t1.descricao_bruta IS NOT NULL
        LIMIT 5
    """
    df = pd.read_sql(query, engine)
    
    print(f"   [QUERY] 🔍 Registos encontrados para processar: {len(df)}")
    
    if df.empty:
        print("   [RESULTADO] ⚠️ O DataFrame está vazio. Motivos prováveis:")
        print("      a) O scraper ainda não correu (tabela imoveis vazia).")
        print("      b) Todos os imóveis já foram analisados.")
        print("      c) Os IDs não coincidem (t1.url_id vs t2.imovel_id).")
    else:
        print("   [RESULTADO] ✅ O script devia estar a correr! IDs encontrados:")
        print(df.head())
        
        # Teste real de inferência
        print("5. [TESTE] A enviar um teste real para a GPU...")
        res = client.chat(model='qwen2.5:7b', messages=[{'role':'user', 'content':'Diz olá!'}])
        print(f"   [GPU] Resposta: {res['message']['content']}")

if __name__ == "__main__":
    debug_run()