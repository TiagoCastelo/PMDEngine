import sys
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Setup de caminhos
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
from common.processing import get_data_from_db, feature_engineering

# Diretoria para guardar os modelos
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def treinar_especialista(df_total, tipo_nome, filtros_tipo, features_cols, target_col):
    """Função genérica para treinar um modelo especializado com filtros robustos."""

    print(f"\n🤖 A treinar Especialista: {tipo_nome.upper()}...")

    # Filtrar registos que correspondem ao tipo de imóvel usando contains para maior flexibilidade
    pattern = '|'.join(filtros_tipo)
    df = df_total[df_total['listing_type'].str.contains(pattern, case=False, na=False)].copy()

    print(f"ℹ️ {tipo_nome}: {len(df)} registos antes do filtro de target.")

    # Aplicar filtro inteligente de área dependendo do tipo
    if tipo_nome == 'habitacional':
        df = df[df['area_bruta_m2'] > 0]
    elif tipo_nome == 'terreno':
        df = df[df['area_terreno_calc'] > 0]
    elif tipo_nome == 'garagem':
        df = df[(df['area_bruta_m2'] > 0) | (df['area_util_m2'] > 0)]

    # Limpeza do target, permitindo valores muito baixos mas positivos
    if target_col in df.columns:
        df = df[(df[target_col] > 0.1) & (df[target_col] < 20000)]
        print(f"ℹ️ {tipo_nome}: {len(df)} registos após filtro de target ({target_col}).")
    else:
        print(f"⚠️ Coluna {target_col} não encontrada no DataFrame. Saltando este especialista.")
        return

    if len(df) < 50:
        print(f"⚠️ Dados insuficientes para {tipo_nome} ({len(df)} registos). A saltar.")
        return

    # Seleção de Features
    cols_disponiveis = [c for c in df.columns if c in features_cols or c.startswith('freg_')]
    X = df[cols_disponiveis].fillna(0)
    y = df[target_col]

    # Treino
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # Avaliação
    score = model.score(X_test, y_test)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"✅ {tipo_nome}: R²={score:.2%} | Erro Médio=€{mae:.2f}/m² ({len(df)} imóveis)")

    # Guardar modelo e colunas
    joblib.dump(model, os.path.join(MODEL_DIR, f'modelo_{tipo_nome}.pkl'))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, f'columns_{tipo_nome}.pkl'))

# ============================
# EXECUÇÃO DO PIPELINE MULTI-MODELO
# ============================

# Carregar dados
df_raw = get_data_from_db()
df_full = feature_engineering(df_raw)

print("\nℹ️ Distribuição de tipos de imóvel após feature engineering:")
print(df_full['listing_type'].value_counts())

# Definição das Features por tipo
feats_habitacao = [
    'area_bruta_m2', 'area_util_m2', 'num_quartos', 'num_wc',
    'idade', 'tem_elevador', 'tem_estacionamento', 'flag_ruina', 'flag_novo'
]

feats_terreno = [
    'area_total_lote', 'flag_urbano', 'flag_rustico', 'flag_viabilidade'
]

feats_garagem = [
    'area_util_m2', 'area_bruta_m2'
]

# Treinar Habitacional -> target = preco_m2
treinar_especialista(
    df_full,
    'habitacional',
    ['apartamento', 'moradia', 'duplex', 'predio'],
    feats_habitacao,
    'preco_m2'
)

# Treinar Terreno -> target = target_terreno
treinar_especialista(
    df_full,
    'terreno',
    ['terreno', 'quinta'],
    feats_terreno,
    'target_terreno'
)

# Treinar Garagem -> target = target_box
treinar_especialista(
    df_full,
    'garagem',
    ['garagem', 'arrecadacao'],
    feats_garagem,
    'target_box'
)

print(f"\n🏁 Processo concluído. Modelos guardados em: {MODEL_DIR}")
