
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Import necessary scikit-learn components
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Define the flatten_text function (must be available for joblib to unpickle if FunctionTransformer is pickled)
def flatten_text(x):
    return x.ravel()

# ==============================================
# 1. Carregar pré-processador e modelo salvos separadamente
# ==============================================
@st.cache_resource
def load_components():
    # Load the fitted TfidfVectorizer
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    # Load the fitted OneHotEncoder
    onehot_encoder = joblib.load("onehot_encoder.pkl")
    # Load the trained model
    model = joblib.load("randomforest_model.pkl")
    return tfidf_vectorizer, onehot_encoder, model

tfidf_vectorizer, onehot_encoder, model = load_components()

# ==============================================
# 2. Título e descrição
# ==============================================
st.set_page_config(page_title="Ranking de Candidatos", layout="wide")
st.title("📊 Ranking de Candidatos - Classificação Automática")
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
    # Define the columns used for prediction in the original training
    inference_cols_for_prediction = [
        "perfil_nivel_academico", "perfil_nivel profissional", "perfil_nivel_ingles",
        "perfil_nivel_espanhol", "nivel_academico", "nivel_ingles", "nivel_espanhol",
        "objetivo_profissional", "titulo_profissional"
    ]

    # Ensure the uploaded dataframe has these columns, handle potential missing ones
    for col in inference_cols_for_prediction:
        if col not in df_inference.columns:
            st.error(f"Missing required column in uploaded CSV: {col}")
            st.stop()

    X_inference = df_inference[inference_cols_for_prediction].copy()

    # Apply OneHotEncoder
    categorical_features = [
        "perfil_nivel_academico", "perfil_nivel profissional", "perfil_nivel_ingles",
        "perfil_nivel_espanhol", "nivel_academico", "nivel_ingles", "nivel_espanhol"
    ]
    X_inference_cat_encoded = onehot_encoder.transform(X_inference[categorical_features])
    cat_feature_names = onehot_encoder.get_feature_names_out(categorical_features)

    # Apply TfidfVectorizer for 'objetivo_profissional'
    # Need to handle potential NaNs before flattening and vectorizing
    X_inference['objetivo_profissional'] = X_inference['objetivo_profissional'].fillna('')
    X_inference_obj_vectorized = tfidf_vectorizer.transform(flatten_text(X_inference[['objetivo_profissional']].values))
    obj_feature_names = [f"objetivo_profissional_tfidf_{i}" for i in range(X_inference_obj_vectorized.shape[1])]

    # Apply TfidfVectorizer for 'titulo_profissional'
    # Need to handle potential NaNs before flattening and vectorizing
    X_inference['titulo_profissional'] = X_inference['titulo_profissional'].fillna('')
    X_inference_titulo_vectorized = tfidf_vectorizer.transform(flatten_text(X_inference[['titulo_profissional']].values))
    titulo_feature_names = [f"titulo_profissional_tfidf_{i}" for i in range(X_inference_titulo_vectorized.shape[1])]


    # Combine features
    import scipy.sparse
    X_inference_processed = scipy.sparse.hstack([
        X_inference_cat_encoded,
        X_inference_obj_vectorized,
        X_inference_titulo_vectorized
    ])

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

    plot_x_col = "nome" # Assuming 'nome' is present in the uploaded CSV

    if plot_x_col not in df_ranked.columns:
         st.warning(f"Column '{plot_x_col}' not found in the uploaded CSV. Using index for plotting.")
         plot_x_col = df_ranked.index.name or "index"
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
