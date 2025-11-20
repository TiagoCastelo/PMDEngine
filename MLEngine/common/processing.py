import pandas as pd
from sqlalchemy import create_engine
import re
from datetime import datetime
import numpy as np

# =========================================================================
# VARIÁVEIS DE CONEXÃO
# =========================================================================
DB_USER = 'user'      
DB_PASSWORD = 'password' 
DB_HOST = 'localhost' 
DB_PORT = '5432'
DB_NAME = 'imoveis'   

def get_data_from_db():
    """Conecta ao PostgreSQL e carrega dados limpos."""
    try:
        engine = create_engine(
            f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
        )
        
        sql_query = """
            SELECT * FROM imoveis 
            WHERE preco_atual > 0;
        """
        
        df = pd.read_sql(sql_query, engine)
        print(f"✅ Dados carregados com sucesso: {len(df)} registos.")
        return df
        
    except Exception as e:
        print(f"❌ Erro de conexão DB: {e}")
        return pd.DataFrame()


def feature_engineering(df):
    """Aplica as transformações necessárias para ML e filtros de qualidade."""
    if df.empty:
        return df

    # --- 1. CLASSIFICAÇÃO E FILTRO ---
    def extract_listing_type_from_link(link):
        """Extrai o tipo puro de imóvel (apartamento, moradia, duplex, terreno, etc.)"""
        if not isinstance(link, str): 
            return 'outra'
        match = re.search(r'venda-([a-z]+)|arrendamento-([a-z]+)', link)
        if match:
            return (match.group(1) or match.group(2) or 'outra').strip()
        return 'outra'

    df['listing_type'] = df['link'].apply(extract_listing_type_from_link)

    # Filtros de Qualidade
    if 'descricao_bruta' in df.columns:
        noise_keywords = ['timeshare', 'direito real de habitação periódica', 'vendido']
        df = df[~df['descricao_bruta'].str.contains('|'.join(noise_keywords), case=False, na=False)].copy()

    # --- 2. ENGENHARIA DE FEATURES ---
    # Preço por m2
    df['preco_m2'] = df['preco_atual'] / df['area_bruta_m2']

    # Terreno
    if 'area_terreno_m2' in df.columns:
        df['area_terreno_calc'] = df['area_terreno_m2'].mask(df['area_terreno_m2'] <= 10, df['area_bruta_m2'])
        df['target_terreno'] = df['preco_atual'] / df['area_terreno_calc'].mask(df['area_terreno_calc'] == 0, np.nan)
    else:
        df['target_terreno'] = 0

    # Garagem / Box
    if 'area_util_m2' in df.columns:
        area_box = df['area_util_m2'].mask(df['area_util_m2'] <= 5, df['area_bruta_m2'])
        df['target_box'] = df['preco_atual'] / area_box.mask(area_box == 0, np.nan)
    else:
        df['target_box'] = 0

    # Substituir inf e NaN
    for col in ['preco_m2', 'target_terreno', 'target_box']:
        df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)

    # Idade do imóvel
    if 'ano_construcao' in df.columns:
        ano_atual = datetime.now().year
        df['idade'] = (ano_atual - df['ano_construcao']).clip(lower=0)

    # Limpeza textual
    df['freguesia_limpa'] = df['freguesia'].fillna('desconhecido').str.lower().str.strip()
    df['tipologia_limpa'] = df['tipologia'].fillna('outra').str.lower().str.strip()

    if 'elevador' in df.columns:
        df['tem_elevador'] = df['elevador'].apply(lambda x: 1 if x == 'Sim' else 0)
    if 'estacionamento' in df.columns:
        df['tem_estacionamento'] = df['estacionamento'].apply(lambda x: 1 if x not in ['Não', None] else 0)

    # Flags a partir da descrição
    if 'descricao_bruta' in df.columns:
        desc = df['descricao_bruta'].fillna('').str.lower()
        df['flag_ruina'] = desc.str.contains('ruína|ruina|recuperar').astype(int)
        df['flag_novo'] = desc.str.contains('novo|construção|estrear').astype(int)
        df['flag_urbano'] = desc.str.contains('urbano|construção|loteamento').astype(int)
        df['flag_rustico'] = desc.str.contains('rústico|rustico').astype(int)
        df['flag_viabilidade'] = desc.str.contains('viabilidade|projecto|aprovado').astype(int)
    else:
        df['flag_ruina'] = df['flag_novo'] = df['flag_urbano'] = df['flag_rustico'] = df['flag_viabilidade'] = 0

    # --- 3. ONE-HOT ENCODING ---
    listing_type_original = df['listing_type'].copy()
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