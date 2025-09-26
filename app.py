
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
from sklearn.ensemble import RandomForestClassifier


# Define the flatten_text function (must be available for joblib to unpickle)
def flatten_text(x):
    return x.ravel()

# Define the preprocessor components locally for robust unpickling
categorical_features_app = [
    "perfil_nivel_academico", "perfil_nivel profissional", "perfil_nivel_ingles",
    "perfil_nivel_espanhol", "nivel_academico", "nivel_ingles", "nivel_espanhol"
]

categorical_transformer_app = OneHotEncoder(handle_unknown="ignore")
text_transformer_app = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="")),
    ("flatten", FunctionTransformer(func=flatten_text, validate=False)),
    ("tfidf", TfidfVectorizer(max_features=500))
])

preprocessor_app = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer_app, categorical_features_app),
        ("text_obj", text_transformer_app, ["objetivo_profissional"]),
        ("text_titulo", text_transformer_app, ["titulo_profissional"])
    ],
    remainder="drop"
)


# ==============================================
# 1. Carregar modelo salvo (pipeline completo)
# ==============================================
@st.cache_resource
def load_model():
    # When loading, joblib needs to be able to find the definition
    # of FunctionTransformer and flatten_text.
    # Explicitly defining the preprocessor structure here can help.
    loaded_pipeline = joblib.load("randomforest_pipeline.pkl")

    # While the structure is loaded, ensuring the FunctionTransformer
    # uses the local flatten_text definition might be necessary.
    # This part is often implicitly handled by joblib when the function
    # is defined at the top level, but redefining the preprocessor
    # structure during load can sometimes force the correct linking.
    # However, the primary fix is ensuring the function exists globally
    # in the app's environment, which we already did.
    # The code below re-creates the preprocessor structure but uses
    # the loaded transformers' fitted state. This is complex and often
    # not necessary if environment versions match and the function is global.
    # Let's rely on the global function definition being sufficient with correct imports.
    # The issue might be deeper if exact version matching doesn't work.

    # Reverting to the simpler load as complex reconstruction is prone to errors
    # if joblib's internal state doesn't match.
    return loaded_pipeline


model = load_model()

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
    # 4. Aplicar modelo para prever probabilidades
    # ==============================================
    # Need to ensure the inference data has the same columns as the training data before preprocessing
    # The original df_full columns were:
    # ['id_vaga', 'inf_titulo_vaga', 'inf_cliente', 'inf_vaga_sap', 'perfil_nivel_academico',
    #  'perfil_nivel profissional', 'perfil_nivel_ingles', 'perfil_nivel_espanhol',
    #  'perfil_competencia_tecnicas_e_comportamentais', 'perfil_principais_atividades',
    #  'titulo', 'id_candidato', 'nome', 'data_candidatura', 'recrutador',
    #  'situacao_candidado', 'target', 'objetivo_profissional', 'titulo_profissional',
    #  'area_atuacao', 'conhecimentos_tecnicos', 'qualificacoes', 'certificacoes',
    #  'experiencias', 'nivel_academico', 'nivel_ingles', 'nivel_espanhol',
    #  'cargo_atual', 'nivel_profissional', 'outro_idioma', 'cursos']

    # The df_etl_clean columns were:
    # ['id_vaga', 'inf_titulo_vaga', 'inf_cliente', 'inf_vaga_sap', 'perfil_nivel_academico',
    #  'perfil_nivel profissional', 'perfil_nivel_ingles', 'perfil_nivel_espanhol',
    #  'perfil_competencia_tecnicas_e_comportamentais', 'perfil_principais_atividades',
    #  'titulo', 'situacao_candidado', 'target', 'objetivo_profissional', 'titulo_profissional',
    #  'area_atuacao', 'conhecimentos_tecnicos', 'certificacoes', 'nivel_academico',
    #  'nivel_ingles', 'nivel_espanhol', 'cursos']

    # The df_inference_clean columns used for prediction were:
    # ['id_vaga', 'inf_titulo_vaga', 'inf_cliente', 'inf_vaga_sap', 'perfil_nivel_academico',
    #  'perfil_nivel profissional', 'perfil_nivel_ingles', 'perfil_nivel_espanhol',
    #  'perfil_competencia_tecnicas_e_comportamentais', 'perfil_principais_atividades',
    #  'titulo', 'situacao_candidado', 'objetivo_profissional', 'titulo_profissional',
    #  'area_atuacao', 'conhecimentos_tecnicos', 'certificacoes', 'nivel_academico',
    #  'nivel_ingles', 'nivel_espanhol', 'cursos']

    # The uploaded CSV `candidatos_nao_classificados.csv` has these columns.
    # We need to ensure the columns passed to the model.predict_proba are the same
    # columns that were used to train the preprocessor (which were from df_train_clean
    # before dropping the target).
    # The preprocessor uses:
    # ['perfil_nivel_academico', 'perfil_nivel profissional', 'perfil_nivel_ingles',
    #  'perfil_nivel_espanhol', 'nivel_academico', 'nivel_ingles', 'nivel_espanhol',
    #  'objetivo_profissional', 'titulo_profissional']

    # The uploaded df_inference (from candidatos_nao_classificados.csv) should already
    # have these columns. Let's explicitly select them to be safe.
    inference_cols_for_prediction = [
        "perfil_nivel_academico", "perfil_nivel profissional", "perfil_nivel_ingles",
        "perfil_nivel_espanhol", "nivel_academico", "nivel_ingles", "nivel_espanhol",
        "objetivo_profissional", "titulo_profissional"
    ]

    # Ensure the uploaded dataframe has these columns, handle potential missing ones
    # although based on the notebook saving process, it should.
    for col in inference_cols_for_prediction:
        if col not in df_inference.columns:
            # This should not happen if using the generated CSV, but as a safeguard:
            st.error(f"Missing required column in uploaded CSV: {col}")
            st.stop() # Stop execution if critical column is missing

    X_inference = df_inference[inference_cols_for_prediction]

    probs = model.predict_proba(X_inference)[:, 1]  # class 1 = approved
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
    # The uploaded CSV `candidatos_nao_classificados.csv` does NOT have the 'nome' column.
    # We need to merge back the 'nome' and 'id_candidato' from the original df_prospects or df_full
    # or modify the saved CSVs to include these identifiers.
    # For simplicity in the app, let's assume the uploaded CSV has an identifier column
    # or modify the notebook to save the inference CSV with identifiers.

    # Let's assume the uploaded CSV has 'id_candidato' and 'nome' based on the project plan
    # aiming to rank candidates. The current saving of `df_nao_classificados.csv` in cell
    # HqM6NjRwqL2G does NOT include 'id_candidato' or 'nome'. This needs to be fixed in the notebook.

    # For now, let's use the index for plotting if 'nome' is not available,
    # but the ideal fix is to save the inference CSV with identifiers.
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
