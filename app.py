
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np # Import numpy for handling potential NaN values
from scipy.sparse import hstack # Import hstack for combining sparse matrices

# Import necessary scikit-learn components
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline # Import Pipeline


# Define the flatten_text function (must be available for joblib to unpickle if FunctionTransformer is pickled)
# This function is no longer strictly needed if we don't pickle the full pipeline,
# but keeping it doesn't hurt and makes the code more robust if we change the
# preprocessing steps later.
def flatten_text(x):
    if x is None: # Handle None values explicitly
        return ""
    return x.ravel()


# ==============================================
# 1. Carregar componentes salvos separadamente
# ==============================================
@st.cache_resource
def load_components():
    # Load the trained model
    model = joblib.load("randomforest_model.pkl")
    # Load the preprocessor components
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    onehot_encoder = joblib.load("onehot_encoder.pkl")

    return model, tfidf_vectorizer, onehot_encoder

model, tfidf_vectorizer, onehot_encoder = load_components()

# ==============================================
# 2. Título e descrição
# ==============================================
st.set_page_config(page_title="Netflix das Vagas", layout="wide")
st.title("📊 Netflix das Vagas - Classificação Automática")
st.markdown(
    '''
    Este aplicativo recebe um **CSV com candidatos não classificados**, aplica o modelo de Machine Learning
    treinado (RandomForest + ETL), e gera um **ranking por probabilidade de aprovação**.

    - ✅ Upload do CSV
    - ✅ Previsão com modelo salvo
    - ✅ Ranking dos melhores candidatos
    - ✅ Download do resultado
    '''
)

# ==============================================
# 3. Upload do CSV
# ==============================================
uploaded_file = st.file_uploader("📥 Carregar arquivo CSV dos candidatos", type="csv")

if uploaded_file is not None:
    # Read CSV
    df_inference = pd.read_csv(uploaded_file)
    st.success(f"Base carregada com {df_inference.shape[0]} registros e {df_inference.shape[1]} colunas.")

    # ==============================================
    # 4. Aplicar pré-processador e modelo para prever probabilidades
    # ==============================================
    # Define the columns to be used for prediction
    categorical_features = [
        "perfil_nivel_academico", "perfil_nivel profissional", "perfil_nivel_ingles",
        "perfil_nivel_espanhol", "nivel_academico", "nivel_ingles", "nivel_espanhol"
    ]
    text_features = ["objetivo_profissional", "titulo_profissional"]


    # Ensure the uploaded dataframe has these columns, handle potential missing ones
    required_cols = categorical_features + text_features
    for col in required_cols:
        if col not in df_inference.columns:
            st.error(f"Missing required column in uploaded CSV: {col}")
            st.stop()

    # Manual Preprocessing Steps
    # 1. Handle missing values (replace NaN with empty string for text features, use a placeholder for categorical)
    df_inference[text_features] = df_inference[text_features].fillna("")
    df_inference[categorical_features] = df_inference[categorical_features].fillna("Não informado") # Or use a specific placeholder

    # 2. Apply OneHotEncoder to categorical features
    X_cat = onehot_encoder.transform(df_inference[categorical_features])

    # 3. Apply TfidfVectorizer to text features
    # Need to apply flatten_text before TF-IDF if the input was not already flattened
    # Assuming the input CSV columns are already simple strings based on previous steps
    # If not, we'd need to apply flatten_text here
    X_text_obj = tfidf_vectorizer.transform(df_inference["objetivo_profissional"])
    X_text_titulo = tfidf_vectorizer.transform(df_inference["titulo_profissional"])


    # Combine the processed features (using hstack for sparse matrices)
    X_inference_processed = hstack([X_cat, X_text_obj, X_text_titulo])


    # Make predictions
    probs = model.predict_proba(X_inference_processed)[:, 1]  # class 1 = approved
    df_inference["probabilidade_aprovacao"] = probs

    # Sort ranking
    df_ranked = df_inference.sort_values(by="probabilidade_aprovacao", ascending=False).reset_index(drop=True)

    # ==============================================
    # 5. Filtrar por vaga e exibir ranking
    # ==============================================
    st.subheader("Filtro por Vaga")

    # Create a combined column for the selectbox
    df_ranked['vaga_info'] = df_ranked['id_vaga'].astype(str) + ' - ' + df_ranked['inf_titulo_vaga']

    # Get unique vacancies
    unique_vacancies = df_ranked[['id_vaga', 'inf_titulo_vaga', 'vaga_info']].drop_duplicates().sort_values(by='vaga_info')

    # Add "Todas as Vagas" option
    all_vacancies_option = 'Todas as Vagas'
    vacancy_list = [all_vacancies_option] + unique_vacancies['vaga_info'].tolist()

    selected_vacancy_info = st.selectbox("Selecione a Vaga:", vacancy_list)

    if selected_vacancy_info == all_vacancies_option:
        df_filtered = df_ranked.copy()
        st.subheader("🏆 Ranking Geral de Candidatos")
    else:
        selected_vacancy_id = int(selected_vacancy_info.split(' - ')[0])
        df_filtered = df_ranked[df_ranked['id_vaga'] == selected_vacancy_id].copy()
        st.subheader(f"🏆 Ranking de Candidatos para a Vaga: {selected_vacancy_info}")

    # Slider for number of top candidates
    num_top_candidates = st.slider("Número de Top Candidatos a Exibir:", min_value=5, max_value=min(50, len(df_filtered)), value=10, step=5)

    # Display top N candidates
    top_n_candidates = df_filtered.head(num_top_candidates)

    # Ensure 'nome' and 'inf_cliente' columns exist for display - they were dropped in df_etl but are in df_full
    # We need to ensure the uploaded CSV `candidatos_para_predicao.csv` includes these.
    # Modify cell HqM6NjRwqL2G to include 'nome' and 'inf_cliente' in the saved CSV.
    # For now, add checks and use placeholders if missing.
    display_cols = ['id_candidato', 'nome', 'inf_cliente', 'probabilidade_aprovacao']
    for col in display_cols:
        if col not in top_n_candidates.columns:
            st.warning(f"Coluna '{col}' não encontrada no CSV. Exibindo apenas colunas disponíveis.")
            display_cols.remove(col) # Remove missing column from display list

    # Display candidates in a structured way
    if not top_n_candidates.empty:
        for index, row in top_n_candidates.iterrows():
            with st.container(border=True):
                st.markdown(f"**Nome:** {row.get('nome', 'Não informado')}")
                st.markdown(f"**ID Candidato:** {row.get('id_candidato', 'Não informado')}")
                st.markdown(f"**Empresa:** {row.get('inf_cliente', 'Não informado')}")
                st.markdown(f"**Probabilidade de Aprovação:** {row['probabilidade_aprovacao']:.2f}")
    else:
        st.info("Nenhum candidato encontrado para os filtros selecionados.")


    # Top N graph (optional, adjust as needed)
    if not top_n_candidates.empty:
        st.subheader(f"📈 Top {num_top_candidates} - Probabilidade de aprovação")
        fig, ax = plt.subplots(figsize=(10, 5))
        # Use 'nome' for x-axis if available, otherwise use index
        plot_x_col = "nome" if "nome" in top_n_candidates.columns else top_n_candidates.index.name or "index"

        if "nome" not in top_n_candidates.columns:
             st.warning("Coluna 'nome' não encontrada no CSV. Usando índice para o gráfico.")
             top_n_candidates[plot_x_col] = top_n_candidates.index # Use index as a temporary column

        top_n_candidates.plot(
            x=plot_x_col, y="probabilidade_aprovacao", kind="bar", ax=ax, legend=False, color="skyblue"
        )
        plt.ylabel("Probabilidade de Aprovação")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    else:
         st.info("Não há dados suficientes para gerar o gráfico com os filtros selecionados.")


    # ==============================================
    # 6. Download of the complete ranking
    # ==============================================
    st.subheader("Download")
    csv = df_ranked.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Baixar ranking completo em CSV",
        data=csv,
        file_name="ranking_candidatos.csv",
        mime="text/csv"
    )
