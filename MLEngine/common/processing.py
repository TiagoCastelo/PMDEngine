import pandas as pd
from sqlalchemy import create_engine, text
import re
from datetime import datetime
import numpy as np
import os

# =========================================================================
# VARIÁVEIS DE CONEXÃO (Lê do ambiente Docker ou usa defaults locais)
# =========================================================================
DB_USER = os.getenv('PGUSER', 'user')
DB_PASSWORD = os.getenv('PGPASSWORD', 'password')
DB_HOST = os.getenv('PGHOST', 'localhost')
DB_PORT = os.getenv('PGPORT', '5432')
DB_NAME = os.getenv('PGDATABASE', 'imoveis')

def get_data_from_db():
    """
    Conecta ao PostgreSQL e carrega a tabela principal + dados da IA.
    Faz um LEFT JOIN para garantir que trazemos todos os imóveis.
    """
    try:
        db_url = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
        engine = create_engine(db_url)
        
        # QUERY COM JOIN
        # Trazemos o estado e a urgência da tabela satélite
        sql_query = """
            SELECT 
                t1.*, 
                t2.estado_conservacao as ai_estado, 
                t2.venda_urgente as ai_urgente
            FROM imoveis t1
            LEFT JOIN imoveis_ai_data t2 ON t1.url_id = t2.imovel_id
            WHERE t1.preco_atual > 0;
        """
        
        df = pd.read_sql(sql_query, engine)
        
        # Pequeno log para controlo
        total = len(df)
        com_ai = df['ai_estado'].notna().sum()
        print(f"✅ Dados carregados: {total} imóveis ({com_ai} enriquecidos pela IA).")
        
        return df
        
    except Exception as e:
        print(f"❌ Erro de conexão DB: {e}")
        return pd.DataFrame()


def feature_engineering(df):
    """Aplica transformações inteligentes, normalizando áreas e usando a IA."""
    if df.empty:
        return df

    # --- 1. EXTRAÇÃO DO TIPO ---
    def extract_listing_type_from_link(link):
        if not isinstance(link, str): return 'outra'
        match = re.search(r'venda-([a-z]+)|arrendamento-([a-z]+)', link)
        if match: return (match.group(1) or match.group(2) or 'outra').strip()
        return 'outra'

    df['listing_type'] = df['link'].apply(extract_listing_type_from_link)

    # --- 2. NORMALIZAÇÃO DE COLUNAS DE ÁREA ---
    cols_areas = ['area_bruta_privativa_m2', 'area_bruta_m2', 'area_util_m2', 'area_terreno_m2', 'area_total_do_lote_m2']
    for c in cols_areas:
        if c not in df.columns:
            df[c] = np.nan

    # Fusão inteligente de Área de Lote / Terreno
    df['area_lote_calc'] = df['area_total_do_lote_m2'].fillna(df['area_terreno_m2'])

    # --- 3. CÁLCULO DA "ÁREA RELEVANTE" (Lógica Waterfall) ---
    df['area_relevante_m2'] = df['area_bruta_m2']

    # A. Apartamentos: Privativa > Útil > Bruta
    mask_apt = df['listing_type'].isin(['apartamento', 'duplex', 'estudio', 'flat'])
    df.loc[mask_apt, 'area_relevante_m2'] = (
        df.loc[mask_apt, 'area_bruta_privativa_m2']
        .fillna(df.loc[mask_apt, 'area_util_m2'])
        .fillna(df.loc[mask_apt, 'area_bruta_m2'])
    )

    # B. Moradias: Bruta Total > Privativa
    mask_house = df['listing_type'].isin(['moradia', 'vivenda', 'predio', 'quinta'])
    df.loc[mask_house, 'area_relevante_m2'] = (
        df.loc[mask_house, 'area_bruta_m2']
        .fillna(df.loc[mask_house, 'area_bruta_privativa_m2'])
    )

    # C. Terrenos: Lote > Bruta
    mask_land = df['listing_type'].isin(['terreno', 'lote', 'terreno-rustico'])
    df.loc[mask_land, 'area_relevante_m2'] = (
        df.loc[mask_land, 'area_lote_calc']
        .fillna(df.loc[mask_land, 'area_bruta_m2'])
    )

    # Correção final de Segurança
    df['area_relevante_m2'] = df['area_relevante_m2'].replace(0, np.nan)
    
    # --- 4. PREÇO POR M2 ---
    df['preco_m2_relevante'] = df['preco_atual'] / df['area_relevante_m2']
    
    # --- 5. ENGENHARIA DO RESTO (IA + Regex) ---
    if 'descricao_bruta' in df.columns:
        desc = df['descricao_bruta'].fillna('').str.lower()
        
        # Fallback Regex
        regex_ruina = desc.str.contains('ruína|ruina|recuperar|demolir|obras totais').astype(int)
        regex_novo = desc.str.contains('novo|construção|estrear').astype(int)
        fallback_score = 3 - (regex_ruina * 2) + (regex_novo * 2)

        # IA - SCORE ESTADO
        if 'ai_estado' in df.columns:
            df['score_estado'] = df['ai_estado'].fillna(fallback_score).astype(int)
            df['score_estado'] = df['score_estado'].clip(1, 5)
        else:
            df['score_estado'] = fallback_score

        # Flags IA Derivadas
        df['flag_ruina'] = (df['score_estado'] <= 2).astype(int)
        df['flag_novo'] = (df['score_estado'] == 5).astype(int)
        
        # IA - URGÊNCIA (A Correção Nuclear)
        if 'ai_urgente' in df.columns:
            # np.where ignora tipos e warnings: Se for True é 1, tudo o resto (None, NaN, False) é 0
            df['flag_urgente'] = np.where(df['ai_urgente'] == True, 1, 0)
        else:
            df['flag_urgente'] = 0

        # Outras flags
        df['flag_urbano'] = desc.str.contains('urbano|construção|loteamento').astype(int)
        df['flag_rustico'] = desc.str.contains('rústico|rustico').astype(int)
        df['flag_viabilidade'] = desc.str.contains('viabilidade|projecto|aprovado').astype(int)

    else:
        # Defaults
        df['score_estado'] = 3
        for c in ['flag_urgente', 'flag_ruina', 'flag_novo', 'flag_urbano', 'flag_rustico']:
            df[c] = 0

    # --- 6. ONE HOT ENCODING ---
    listing_type_original = df['listing_type'].copy()
    
    # Limpeza de texto
    df['freguesia_limpa'] = df['freguesia'].fillna('desconhecido').str.lower().str.strip()
    df['tipologia_limpa'] = df['tipologia'].fillna('outra').str.lower().str.strip()

    # Binários
    if 'elevador' in df.columns:
        df['tem_elevador'] = df['elevador'].apply(lambda x: 1 if x == 'Sim' else 0)
    if 'estacionamento' in df.columns:
        df['tem_estacionamento'] = df['estacionamento'].apply(lambda x: 1 if x not in ['Não', None] else 0)

    # Encoding
    df_encoded = pd.get_dummies(
        df,
        columns=['tipologia_limpa', 'freguesia_limpa', 'certificado_energetico', 'listing_type'],
        prefix=['tipo', 'freg', 'cert', 'lst_type'],
        dummy_na=False,
        drop_first=True
    )
    
    df_encoded['listing_type'] = listing_type_original
    df_encoded = df_encoded.loc[:, ~df_encoded.columns.duplicated()]

    return df_encoded