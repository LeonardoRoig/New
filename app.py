
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
st.set_page_config(page_title="Netflix das Vagas", layout="wide") # Modified title
st.title("📊 Netflix das Vagas - Classificação Automática") # Modified title
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
    # 5. Exibir resultados
    # ==============================================
    st.subheader("🏆 Top 20 candidatos mais promissores")
    st.dataframe(df_ranked.head(20))

    # Top 10 graph
    st.subheader("📈 Top 10 - Probabilidade de aprovação")
    fig, ax = plt.subplots(figsize=(10, 5))
    # Ensure 'nome' column exists for plotting - it was dropped in df_etl but is in df_prospects
    # The uploaded CSV `candidatos_nao_classificados.csv` might not have 'nome'.
    # We need to ensure the uploaded CSV has 'nome' or use another identifier.
    # Based on the original df_prospects and df_full, 'nome' and 'id_candidato' exist before ETL.
    # The CSV `candidatos_nao_classificados.csv` saved in cell HqM6NjRwqL2G does NOT include 'nome' or 'id_candidato'.
    # To plot by name, we need to either:
    # 1. Modify cell HqM6NjRwqL2G to save 'id_candidato' and 'nome' in `candidatos_nao_classificados.csv`.
    # 2. Assume the uploaded CSV contains 'nome' or 'id_candidato' and merge back.
    # Option 1 is better for a self-contained example. Let's plan to modify HqM6NjRwqL2G later if necessary.
    # For now, let's add a check and warning if 'nome' is missing, using the index for plotting as a fallback.

    plot_x_col = "nome" if "nome" in df_ranked.columns else df_ranked.index.name or "index"

    if "nome" not in df_ranked.columns:
         st.warning("Column 'nome' not found in the uploaded CSV. Using index for plotting.")
         df_ranked[plot_x_col] = df_ranked.index # Use index as a temporary column


    fig, ax = plt.subplots(figsize=(10, 5))
    df_ranked.head(10).plot(
        x=plot_x_col, y="probabilidade_aprovacao", kind="bar", ax=ax, legend=False, color="skyblue"
    )
    plt.ylabel("Probabilidade de Aprovação")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)


    # ==============================================
    # 6. Download of the complete ranking
    # ==============================================
    csv = df_ranked.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Baixar ranking completo em CSV",
        data=csv,
        file_name="ranking_candidatos.csv",
        mime="text/csv"
    )
