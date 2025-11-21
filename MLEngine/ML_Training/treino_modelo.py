import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Setup de caminhos para importar o common
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
from common.processing import get_data_from_db, feature_engineering

# Diretoria para guardar os modelos
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def treinar_especialista(df_total, tipo_nome, filtros_tipo, features_cols, target_col):
    """
    Treina um modelo especializado usando a nova lógica de Área Relevante.
    """
    print(f"\n🤖 A treinar Especialista: {tipo_nome.upper()}...")

    # 1. Filtro de Tipo de Imóvel
    pattern = '|'.join(filtros_tipo)
    df = df_total[df_total['listing_type'].str.contains(pattern, case=False, na=False)].copy()

    # 2. Filtro de Consistência (Usa a nova coluna unificada)
    # Remove imóveis sem área ou com preços estranhos
    df = df[
        (df['area_relevante_m2'] > 10) & 
        (df['preco_atual'] > 5000)
    ].copy()

    
    # 3. Preparar Target e Remover Outliers (CRUCIAL PARA TERRENOS)
    if target_col == 'preco_m2_relevante':
        # Habitacional: Aceitamos tudo entre 500€ e 15.000€ o metro
        if tipo_nome == 'habitacional':
            df = df[(df[target_col] > 500) & (df[target_col] < 15000)]
        
        # Terrenos: A dispersão é gigante. Vamos focar apenas em terrenos de construção "normais"
        # Removemos quintas gigantes (baratas ao m2) e micro-lotes de ouro.
        elif tipo_nome == 'terreno':
            df = df[(df[target_col] > 10) & (df[target_col] < 2000)]
            
        # Garagens: Geralmente entre 200€ e 3000€ o metro
        elif tipo_nome == 'garagem':
            df = df[(df[target_col] > 200) & (df[target_col] < 5000)]

    # 4. Seleção de Features
    # Garante que só usamos colunas que existem
    cols_disponiveis = [c for c in df.columns if c in features_cols or c.startswith('freg_') or c.startswith('tipo_')]
    
    X = df[cols_disponiveis].fillna(0)
    y = df[target_col]

    # 5. Treino
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # 6. Avaliação
    score = model.score(X_test, y_test)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    
    print(f"✅ {tipo_nome}: R²={score:.2%} | Erro Médio (MAE)= {mae:.2f}")

    # 7. Feature Importance (Para veres se a IA está a ajudar)
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("   🔝 Top 3 Fatores mais importantes:")
        for i in range(min(3, len(indices))):
            print(f"      {i+1}. {X.columns[indices[i]]} ({importances[indices[i]]:.1%})")
    except: pass

    # Guardar
    joblib.dump(model, os.path.join(MODEL_DIR, f'modelo_{tipo_nome}.pkl'))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, f'columns_{tipo_nome}.pkl'))

# ============================
# EXECUÇÃO
# ============================
if __name__ == "__main__":
    # 1. Carregar dados frescos (com IA)
    df_raw = get_data_from_db()
    
    # 2. Processamento (Cria area_relevante_m2, score_estado, etc.)
    df_full = feature_engineering(df_raw)

    # 3. Definição das Features Inteligentes
    # NOTA: Agora usamos 'area_relevante_m2' para tudo, porque ela adapta-se.
    # Adicionámos 'score_estado' e 'flag_urgente' vindos da IA.
    
    feats_comuns = [
        'area_relevante_m2', 
        'idade', 
        'num_quartos', 
        'num_wc', 
        'score_estado',    # <--- Ouro da IA (1-5)
        'flag_urgente',    # <--- Ouro da IA (True/False)
        'flag_ruina',      # Derivado do score_estado
        'flag_novo',       # Derivado do score_estado
        'tem_elevador', 
        'tem_estacionamento'
    ]

    feats_terreno = [
        'area_relevante_m2', # No caso de terrenos, isto é a área do lote (definido no processing)
        'flag_urbano', 
        'flag_rustico', 
        'flag_viabilidade'
    ]

    feats_garagem = [
        'area_relevante_m2'
    ]

    # --- TREINO DOS ESPECIALISTAS ---
    
    # A. Habitacional
    # Target: Preço por m2 (é mais estável para prever)
    treinar_especialista(
        df_full,
        'habitacional',
        ['apartamento', 'moradia', 'duplex', 'predio', 'quinta'],
        feats_comuns,
        'preco_m2_relevante' # Coluna criada no novo processing.py
    )

    # B. Terrenos
    # Target: Preço por m2 de lote
    treinar_especialista(
        df_full,
        'terreno',
        ['terreno', 'lote'],
        feats_terreno,
        'preco_m2_relevante'
    )

    # C. Garagens
    treinar_especialista(
        df_full,
        'garagem',
        ['garagem', 'arrecadacao'],
        feats_garagem,
        'preco_m2_relevante'
    )

    print(f"\n🏁 Treino concluído. Modelos guardados em: {MODEL_DIR}")