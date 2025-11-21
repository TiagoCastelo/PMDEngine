import os
import time
import json
import re
import argparse
import ollama
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, Field

# ==========================================
# 1. CONFIGURAÇÃO
# ==========================================
MODEL = "llama3.2"

# Ambiente
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
DB_USER = os.getenv('PGUSER', 'user')
DB_PASSWORD = os.getenv('PGPASSWORD', 'password')
DB_HOST = os.getenv('PGHOST', 'localhost')
DB_PORT = os.getenv('PGPORT', '5432')
DB_NAME = os.getenv('PGDATABASE', 'imoveis')

DB_STR = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ==========================================
# 2. SCHEMA
# ==========================================
class AnaliseAI(BaseModel):
    estado: int = Field(description="Rating 1-5")
    urgencia: bool = Field(description="True/False")
    tipo: str = Field(description="Short summary string")

# ==========================================
# 3. FUNÇÕES DE MANUTENÇÃO (--fix / --clean)
# ==========================================
def run_fix(engine):
    print("🔧 A aplicar FIX: Alterar coluna 'potencial_investimento' para TEXT...")
    sql = "ALTER TABLE imoveis_ai_data ALTER COLUMN potencial_investimento TYPE TEXT;"
    try:
        with engine.connect() as conn:
            conn.execute(text(sql))
            conn.commit()
        print("✅ Sucesso! Limite de caracteres removido.")
    except Exception as e:
        print(f"ℹ️  Resultado: {e}")

def run_clean(engine):
    print("⚠️  PERIGO: A apagar TODOS os dados da tabela 'imoveis_ai_data'...")
    # Confirmação simples (opcional, removi para automação)
    sql = "TRUNCATE TABLE imoveis_ai_data;"
    try:
        with engine.connect() as conn:
            conn.execute(text(sql))
            conn.commit()
        print("✅ Tabela limpa. Tudo pronto para re-processar.")
    except Exception as e:
        print(f"❌ Erro ao limpar: {e}")

def setup_database(engine):
    """Cria tabela se não existir (já com a coluna TEXT correta)."""
    sql = """
    CREATE TABLE IF NOT EXISTS imoveis_ai_data (
        imovel_id VARCHAR(255) PRIMARY KEY,
        estado_conservacao INTEGER,
        venda_urgente BOOLEAN,
        potencial_investimento TEXT, 
        analisado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT fk_imovel FOREIGN KEY(imovel_id) REFERENCES imoveis(url_id) ON DELETE CASCADE
    );
    """
    try:
        with engine.connect() as conn:
            conn.execute(text(sql))
            conn.commit()
    except SQLAlchemyError:
        pass

# ==========================================
# 4. MOTOR DE IA (O Loop Principal)
# ==========================================
def extract_json_fallback(text_content):
    try:
        match = re.search(r'\{.*\}', text_content, re.DOTALL)
        if match: return json.loads(match.group())
    except: pass
    return None

def run_enrichment(engine):
    print(f"🚀 A iniciar MOTOR IA com {MODEL}...", flush=True)
    
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        try: client.show(MODEL)
        except: 
            print(f"⬇️ A baixar {MODEL}...", flush=True)
            client.pull(MODEL)
    except Exception as e:
        print(f"❌ Erro Ollama: {e}")
        return

    while True:
        # Batch de 50
        query = """
            SELECT t1.url_id, t1.descricao_bruta 
            FROM imoveis t1
            LEFT JOIN imoveis_ai_data t2 ON t1.url_id = t2.imovel_id
            WHERE t2.imovel_id IS NULL 
              AND t1.descricao_bruta IS NOT NULL
              AND LENGTH(t1.descricao_bruta) > 20
            LIMIT 50
        """
        
        try:
            df = pd.read_sql(query, engine)
        except Exception:
            time.sleep(5)
            continue

        if df.empty:
            print("✅ Tudo analisado! A terminar.", flush=True)
            break

        print(f"\n📦 Lote de {len(df)} imóveis. A processar...", flush=True)
        dados_ai = []
        t_batch_start = time.time()
        
        for i, row in df.iterrows():
            imovel_id = row['url_id']
            print(f"   👉 {imovel_id} ", end="", flush=True)
            
            t_start = time.time()
            texto = row['descricao_bruta'][:1200].replace('"', "'").replace('\n', ' ')
            
            # Prompt (Inglês para melhor performance do Llama 3)
            system_msg = """
            You are a Real Estate Analyst. Analyze the Portuguese text and extract JSON.
            RULES:
            - 'estado' (Int 1-5): 1=Ruin/Demolish, 2=Renovation Needed, 3=Habitable/Used, 4=Good, 5=New/Luxury. Default to 3.
            - 'urgencia' (Bool): true ONLY if debt/bank/urgent/divorce mentioned.
            - 'tipo' (String): Summarize opportunity in Portuguese (max 15 words).
            """
            
            # Defaults
            estado = 3
            urgencia = False
            tipo = "N/A"

            try:
                res = client.chat(
                    model=MODEL,
                    messages=[
                        {'role': 'system', 'content': system_msg},
                        {'role': 'user', 'content': f"Description: {texto}"}
                    ],
                    format=AnaliseAI.model_json_schema(),
                    options={'temperature': 0.1}
                )
                
                raw = res['message']['content']
                try: content = json.loads(raw)
                except: content = extract_json_fallback(raw)

                if content:
                    # Validação
                    estado = content.get('estado', 3)
                    if estado not in [1, 2, 3, 4, 5]: estado = 3
                    urgencia = content.get('urgencia', False)
                    tipo = str(content.get('tipo', 'N/A'))
                    
                    # Visuals
                    tempo = time.time() - t_start
                    icon = "🏚️" if estado <= 2 else ("💎" if estado == 5 else "🏠")
                    urg_icon = "🔥" if urgencia else ""
                    tipo_print = (tipo[:40] + '..') if len(tipo) > 40 else tipo
                    
                    print(f"✅ {tempo:.2f}s | {estado} {icon} | {tipo_print} {urg_icon}", flush=True)
                else:
                    print(f"⚠️ JSON Inválido", flush=True)

            except Exception as e:
                print(f"❌ Erro: {e}", flush=True)

            dados_ai.append({
                'imovel_id': imovel_id,
                'estado_conservacao': estado,
                'venda_urgente': urgencia,
                'potencial_investimento': tipo # Agora grava texto longo sem medo
            })

        if dados_ai:
            try:
                pd.DataFrame(dados_ai).to_sql('imoveis_ai_data', engine, if_exists='append', index=False)
                print(f"💾 Lote guardado ({time.time()-t_batch_start:.1f}s).", flush=True)
            except Exception as e:
                print(f"❌ ERRO SQL: {e}", flush=True)

# ==========================================
# 5. ENTRY POINT (CLI)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PMD Engine AI Worker")
    parser.add_argument('--fix', action='store_true', help="Corrige o tipo da coluna na BD para TEXT")
    parser.add_argument('--clean', action='store_true', help="Apaga TODOS os dados analisados para recomeçar")
    
    args = parser.parse_args()
    
    try:
        engine = create_engine(DB_STR)
        setup_database(engine) # Garante sempre que a tabela existe
        
        if args.fix:
            run_fix(engine)
        elif args.clean:
            run_clean(engine)
        else:
            # Se não houver flags, corre o programa normal
            run_enrichment(engine)
            
    except Exception as e:
        print(f"❌ Erro de conexão inicial: {e}")