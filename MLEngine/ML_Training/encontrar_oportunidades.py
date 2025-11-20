import sys
import os
import pandas as pd
import joblib
import numpy as np

# --- CONFIGURAÇÃO DE CAMINHOS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

MODEL_DIR = os.path.join(current_dir, 'models')

from common.processing import get_data_from_db, feature_engineering

# =========================================================================
# 1. MAPA DE MODELOS (A Ponte entre Dados e IA)
# =========================================================================
# Define qual modelo deve ser usado para qual tipo de imóvel
MODELO_MAPPING = {
    'apartamento': 'habitacional',
    'moradia': 'habitacional',
    'duplex': 'habitacional',
    'predio': 'habitacional',
    'terreno': 'terreno',
    'quinta': 'terreno',
    'garagem': 'garagem',
    'arrecadacao': 'garagem'
}

# =========================================================================
# 2. CARREGAR E PROCESSAR DADOS
# =========================================================================
print("🚀 A carregar dados da base de dados...")
df_raw = get_data_from_db()

if df_raw.empty:
    print("❌ Sem dados para analisar.")
    sys.exit(1)

# Feature Engineering (para ter as colunas listing_type, flags, etc.)
df_features = feature_engineering(df_raw.copy())

# Colunas apenas para o relatório final (não entram no modelo)
cols_info = ['link', 'freguesia', 'preco_atual', 'area_bruta_m2', 'area_terreno_m2', 'listing_type']

# =========================================================================
# 3. AVALIAÇÃO INTELIGENTE
# =========================================================================

dfs_avaliados = []
modelos_carregados = {} # Cache para não carregar o mesmo modelo 1000 vezes

print(f"\n🧠 A iniciar avaliação de mercado...")

# Agrupar os imóveis pelo tipo de MODELO que precisam (ex: todos os habitacionais juntos)
df_features['modelo_necessario'] = df_features['listing_type'].map(MODELO_MAPPING).fillna('outros')

for modelo_nome, df_grupo in df_features.groupby('modelo_necessario'):
    
    if modelo_nome == 'outros': continue

    # Tentar carregar o modelo se ainda não estiver em memória
    if modelo_nome not in modelos_carregados:
        path_model = os.path.join(MODEL_DIR, f"modelo_{modelo_nome}.pkl")
        path_cols = os.path.join(MODEL_DIR, f"columns_{modelo_nome}.pkl")
        
        if os.path.exists(path_model):
            try:
                modelos_carregados[modelo_nome] = {
                    'model': joblib.load(path_model),
                    'cols': joblib.load(path_cols)
                }
            except Exception as e:
                print(f"❌ Erro ao carregar modelo {modelo_nome}: {e}")
                continue
        else:
            print(f"⚠️ Modelo '{modelo_nome}' não existe (provavelmente não houve dados suficientes).")
            continue

    # Se temos modelo, fazemos a previsão
    artefactos = modelos_carregados[modelo_nome]
    model = artefactos['model']
    train_cols = artefactos['cols']
    
    # Preparar X (garantir mesmas colunas do treino)
    # Removemos colunas de info e targets que possam existir
    X = df_grupo.drop(columns=cols_info + ['preco_m2', 'target_habitacional', 'target_terreno', 'target_box', 'url_id', 'last_crawled', 'data_publicacao', 'descricao_bruta'], errors='ignore')
    X_final = X.reindex(columns=train_cols, fill_value=0).fillna(0)
    
    # Prever Target (Preço/m2 ou Preço/Lote dependendo do modelo)
    pred_target = model.predict(X_final)
    
    df_grupo = df_grupo.copy()
    df_grupo['valor_target_previsto'] = pred_target
    
    # Calcular Valor Total Estimado (Lógica inversa do treino)
    if modelo_nome == 'habitacional':
        df_grupo['valor_justo'] = pred_target * df_grupo['area_bruta_m2']
    elif modelo_nome == 'terreno':
        # Usa a mesma lógica de área calculada do treino
        area_calc = np.where(df_grupo['area_terreno_m2'] > 10, df_grupo['area_terreno_m2'], df_grupo['area_bruta_m2'])
        df_grupo['valor_justo'] = pred_target * area_calc
    elif modelo_nome == 'garagem':
        area_calc = np.where(df_grupo['area_util_m2'] > 5, df_grupo['area_util_m2'], df_grupo['area_bruta_m2'])
        df_grupo['valor_justo'] = pred_target * area_calc
        
    dfs_avaliados.append(df_grupo)
    print(f"✅ Avaliados {len(df_grupo)} imóveis com modelo '{modelo_nome}'.")

# =========================================================================
# 4. RELATÓRIO DE OPORTUNIDADES
# =========================================================================

if not dfs_avaliados:
    print("❌ Nenhum imóvel pôde ser avaliado.")
    sys.exit()

df_final = pd.concat(dfs_avaliados)

# Cálculo do Lucro Potencial
df_final['diferenca'] = df_final['valor_justo'] - df_final['preco_atual']
df_final['lucro_potencial_perc'] = (df_final['diferenca'] / df_final['valor_justo']) * 100

# Filtros de Qualidade para o Relatório
# 1. Ignorar lucros absurdos (> 80% geralmente é erro de dados ou ruína não detetada)
# 2. Ignorar lucros negativos
filtro_oportunidade = (df_final['lucro_potencial_perc'] > 20) & (df_final['lucro_potencial_perc'] < 80)
oportunidades = df_final[filtro_oportunidade].sort_values(by='lucro_potencial_perc', ascending=False)

pd.options.display.float_format = '{:,.0f} €'.format

print("\n=================================================================================")
print(f"🏆 TOP 20 OPORTUNIDADES REAIS (Lucro Estimado > 20%)")
print("=================================================================================\n")

if oportunidades.empty:
    print("Nenhuma oportunidade encontrada com estes critérios.")
else:
    cols_show = ['listing_type', 'freguesia', 'area_bruta_m2', 'preco_atual', 'valor_justo', 'diferenca', 'lucro_potencial_perc', 'link']
    
    # Formatação
    display = oportunidades[cols_show].head(20).copy()
    display['lucro_potencial_perc'] = display['lucro_potencial_perc'].apply(lambda x: f"{x:.1f}%")
    
    print(display.to_markdown(index=False))

print(f"\nℹ️ Total analisado: {len(df_final)} | Oportunidades: {len(oportunidades)}")