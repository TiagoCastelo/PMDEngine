import sys
import os
import pandas as pd
import joblib
import numpy as np

# =========================================================================
# SETUP E CAMINHOS
# =========================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

# Importar o processador centralizado
from common.processing import get_data_from_db, feature_engineering

MODEL_DIR = os.path.join(current_dir, 'models')

# Mapa para saber que modelo usar para cada tipo de imóvel
MODELO_MAPPING = {
    'apartamento': 'habitacional',
    'moradia': 'habitacional',
    'duplex': 'habitacional',
    'predio': 'habitacional',
    'quinta': 'habitacional',
    'terreno': 'terreno',
    'lote': 'terreno',
    'garagem': 'garagem',
    'arrecadacao': 'garagem'
}

def main():
    # 1. CARREGAR DADOS
    print("🚀 A carregar dados da Base de Dados (SQL)...")
    df_raw = get_data_from_db()
    
    if df_raw.empty:
        print("❌ Sem dados. Verifica se o scraper e o enrich_data.py já correram.")
        return

    # 2. PROCESSAMENTO (Calcula Areas Relevantes e Scores IA)
    print("⚙️ A processar features e inteligência artificial...")
    df_features = feature_engineering(df_raw.copy())

    # =========================================================================
    # 3. FILTRO INTERATIVO (FREGUESIAS)
    # =========================================================================
    print("\n📍 FILTRO GEOGRÁFICO")
    lista_freguesias = sorted(df_features['freguesia'].dropna().unique().tolist())
    
    print(f"   0 - Todas as {len(lista_freguesias)} freguesias")
    # Listar apenas as top 10 mais frequentes para não encher o ecrã
    top_freguesias = df_features['freguesia'].value_counts().head(10).index.tolist()
    for i, f in enumerate(top_freguesias, 1):
        print(f"   {i} - {f}")
    
    escolha = input("\n➡️ Escolha (ENTER para todas, ou digite parte do nome): ").strip().lower()

    if escolha and not escolha.isdigit():
        # Filtro por texto (ex: "benfica")
        df_features = df_features[df_features['freguesia_limpa'].str.contains(escolha)]
        print(f"🔍 Filtrado por nome: {len(df_features)} imóveis encontrados.")
    elif escolha.isdigit() and int(escolha) > 0 and int(escolha) <= len(top_freguesias):
        # Filtro por número
        freguesia_nome = top_freguesias[int(escolha)-1]
        df_features = df_features[df_features['freguesia'] == freguesia_nome]
        print(f"🔍 Filtrado por: {freguesia_nome}")
    else:
        print("🌍 Analisando TODO o mercado.")

    if df_features.empty:
        print("❌ Nenhum imóvel encontrado com esse filtro.")
        return

    # =========================================================================
    # 4. AVALIAÇÃO DE MERCADO (PREDIÇÃO)
    # =========================================================================
    print(f"\n🧠 A avaliar {len(df_features)} imóveis com modelos ML...")
    
    dfs_avaliados = []
    df_features['modelo_necessario'] = df_features['listing_type'].map(MODELO_MAPPING).fillna('outros')

    for modelo_nome, df_grupo in df_features.groupby('modelo_necessario'):
        if modelo_nome == 'outros': continue

        # Carregar o Cérebro Especialista
        path_model = os.path.join(MODEL_DIR, f"modelo_{modelo_nome}.pkl")
        path_cols = os.path.join(MODEL_DIR, f"columns_{modelo_nome}.pkl")
        
        if not os.path.exists(path_model):
            print(f"⚠️ Modelo '{modelo_nome}' não encontrado. (Corre o treino_modelo.py primeiro)")
            continue

        try:
            model = joblib.load(path_model)
            train_cols = joblib.load(path_cols)
        except Exception as e:
            print(f"❌ Erro modelo {modelo_nome}: {e}")
            continue

        # Preparar dados para o modelo (Garante as mesmas colunas do treino)
        X = df_grupo.reindex(columns=train_cols, fill_value=0)
        
        # PREDIÇÃO: O modelo devolve o Preço Justo por m²
        pred_preco_m2 = model.predict(X)
        
        # CÁLCULO DO VALOR FINAL
        # Valor = Preço m2 Estimado * Área Relevante (Lote para terrenos, Privativa para apts)
        df_grupo = df_grupo.copy()
        df_grupo['valor_justo'] = pred_preco_m2 * df_grupo['area_relevante_m2']
        
        dfs_avaliados.append(df_grupo)

    if not dfs_avaliados:
        return

    df_final = pd.concat(dfs_avaliados)

    # =========================================================================
    # 5. RELATÓRIO DE OPORTUNIDADES
    # =========================================================================
    # Lucro Potencial = (Valor Justo - Preço Atual)
    df_final['lucro_potencial'] = df_final['valor_justo'] - df_final['preco_atual']
    df_final['margem_perc'] = (df_final['lucro_potencial'] / df_final['preco_atual']) * 100

    # CRITÉRIOS DE OURO PARA FILTRAGEM
    filtro_oportunidade = (
        (df_final['preco_atual'] > 10000) &          # Ignorar lixo/erros
        (
            (df_final['margem_perc'] > 20) |         # Margem financeira alta
            (df_final['flag_urgente'] == 1)          # OU Urgência detetada pela IA
        )
    )

    oportunidades = df_final[filtro_oportunidade].sort_values(by='margem_perc', ascending=False).head(30)

    print("\n" + "="*80)
    print(f"🏆 TOP 30 OPORTUNIDADES DE NEGÓCIO (IA + ML)")
    print("="*80)

    if oportunidades.empty:
        print("Nenhuma oportunidade clara encontrada hoje. Tente mudar os filtros.")
    else:
        # Preparar tabela bonita
        display = oportunidades.copy()
        
        # Formatar colunas
        display['Preço'] = display['preco_atual'].apply(lambda x: f"{x:,.0f}€")
        display['Justo'] = display['valor_justo'].apply(lambda x: f"{x:,.0f}€")
        display['Margem'] = display['margem_perc'].apply(lambda x: f"{x:+.0f}%")
        display['Area'] = display['area_relevante_m2'].apply(lambda x: f"{x:.0f}m2")
        
        # Coluna IA: Combina Estado e Urgência num ícone
        def formata_ia(row):
            icon_est = "🏚️" if row['score_estado'] <= 2 else ("💎" if row['score_estado'] >= 5 else "🏠")
            icon_urg = "🔥URG" if row['flag_urgente'] else ""
            return f"{icon_est} {icon_urg}"
        
        display['IA'] = display.apply(formata_ia, axis=1)

        # Selecionar colunas finais
        cols_finais = ['listing_type', 'freguesia', 'Area', 'Preço', 'Justo', 'Margem', 'IA', 'link']
        
        print(display[cols_finais].to_markdown(index=False))
        
        # Guardar Excel/CSV para análise
        f_name = 'oportunidades_do_dia.csv'
        display[cols_finais].to_csv(f_name, index=False)
        print(f"\n💾 Relatório guardado em: {f_name}")

if __name__ == "__main__":
    main()