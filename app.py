
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==============================================
# 1. Carregar modelo salvo (pipeline completo)
# ==============================================
@st.cache_resource
def load_model():
    return joblib.load("randomforest_pipeline.pkl")

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
    # Ler CSV
    df_inference = pd.read_csv(uploaded_file)
    st.success(f"Base carregada com {df_inference.shape[0]} registros e {df_inference.shape[1]} colunas.")

    # ==============================================
    # 4. Aplicar modelo para prever probabilidades
    # ==============================================
    probs = model.predict_proba(df_inference)[:, 1]  # classe 1 = aprovado
    df_inference["probabilidade_aprovacao"] = probs

    # Ordenar ranking
    df_ranked = df_inference.sort_values(by="probabilidade_aprovacao", ascending=False).reset_index(drop=True)

    # ==============================================
    # 5. Exibir resultados
    # ==============================================
    st.subheader("🏆 Top 20 candidatos mais promissores")
    st.dataframe(df_ranked.head(20))

    # Gráfico top 10
    st.subheader("📈 Top 10 - Probabilidade de aprovação")
    fig, ax = plt.subplots(figsize=(10, 5))
    df_ranked.head(10).plot(
        x="nome", y="probabilidade_aprovacao", kind="bar", ax=ax, legend=False, color="skyblue"
    )
    plt.ylabel("Probabilidade de Aprovação")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    # ==============================================
    # 6. Download do ranking completo
    # ==============================================
    csv = df_ranked.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Baixar ranking completo em CSV",
        data=csv,
        file_name="ranking_candidatos.csv",
        mime="text/csv"
    )
