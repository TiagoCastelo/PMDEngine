# api/main.py
import sys
import os
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel,ConfigDict
from typing import Optional # Para o float

# --- CONFIGURAÇÃO DE CAMINHOS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

# Caminho para onde o modelo foi guardado (ML_Training/models)
MODEL_PATH = os.path.join(root_dir, 'ML_Training', 'models')

# Importar lógica partilhada (Assumindo que está em common/processing.py)
from common.processing import feature_engineering

app = FastAPI(title="MLEngine API", version="1.0", description="Motor de Previsão de Preços Imobiliários (MLEngine)")

# Carregar Modelo e Colunas
try:
    model = joblib.load(os.path.join(MODEL_PATH, 'ml_engine_model.pkl'))
    model_cols = joblib.load(os.path.join(MODEL_PATH, 'model_columns.pkl'))
    print("✅ API Pronta: Modelo carregado.")
except Exception as e:
    print(f"❌ Erro fatal: Não encontrei o modelo em {MODEL_PATH}. Corra o treino primeiro! Detalhe: {e}")
    model = None
    model_cols = None

# Define o formato dos dados que a API espera receber
class ImovelInput(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "area_bruta_m2": 135.0,
            "num_quartos": 3,
            "num_wc": 2,
            "ano_construcao": 1995,
            "freguesia": "Avenidas Novas",
            "tipologia": "Apartamento T3",
            "elevador": "Sim",
            "estacionamento": "1 Lugar",
            "certificado_energetico": "B",
            "preco_compra": 300000.0,
            "custo_obra": 80000.0
        }
    })
    
    area_bruta_m2: float
    num_quartos: int = 0
    num_wc: int = 0
    ano_construcao: int = 2020
    freguesia: str
    tipologia: str = "Apartamento"
    elevador: str = "Não"
    estacionamento: str = "Não"
    certificado_energetico: str = "Desconhecido"
    
    preco_compra: float = 0.0
    custo_obra: float = 0.0

@app.post("/predict")
def predict_price(imovel: ImovelInput):
    if not model: 
        return {"error": "Modelo não carregado. Treine o modelo primeiro."}

    # 1. Criar DataFrame com o input do utilizador
    dados = {
        'area_bruta_m2': [imovel.area_bruta_m2],
        'num_quartos': [imovel.num_quartos],
        'num_wc': [imovel.num_wc],
        'ano_construcao': [imovel.ano_construcao],
        'freguesia': [imovel.freguesia],
        'tipologia': [imovel.tipologia],
        'elevador': [imovel.elevador],
        'estacionamento': [imovel.estacionamento],
        'certificado_energetico': [imovel.certificado_energetico], 
        'link': [''] # Dummy para o processing.py
    }
    
    # 2. Usar a função partilhada para processar (Feature Engineering)
    df_input = pd.DataFrame(dados)
    df_processed = feature_engineering(df_input)
    
    # 3. Alinhar colunas (Reindexar para garantir 500+ colunas)
    df_final = df_processed.reindex(columns=model_cols, fill_value=0)
    
    # 4. Prever o Preço de Venda Final (ARV)
    preco_m2_previsto = model.predict(df_final)[0]
    preco_venda_total_previsto = preco_m2_previsto * imovel.area_bruta_m2
    
    # 5. CÁLCULO DE GANHOS (Flipping Logic)
    investimento_total = imovel.preco_compra + imovel.custo_obra
    lucro_potencial = preco_venda_total_previsto - investimento_total
    
    # ROI: Retorno do investimento (Lucro / Investimento)
    roi = (lucro_potencial / investimento_total) if investimento_total > 0 else 0

    return {
        "estimativa_valor_venda": round(preco_venda_total_previsto, 2),
        "preco_m2_previsto": round(preco_m2_previsto, 2),
        "lucro_potencial_bruto": round(lucro_potencial, 2),
        "roi_percentagem": f"{round(roi * 100, 2)}%",
        "moeda": "EUR"
    }

if __name__ == "__main__":
    import uvicorn
    # A correr na porta 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)