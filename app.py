import streamlit as st
import joblib
from googletrans import Translator

# Carrega o modelo
pipeline = joblib.load("final_model.pkl")

# Instancia o tradutor
translator = Translator()

# Função de classificação
def classificar_texto_portugues(texto_pt):
    traduzido = translator.translate(texto_pt, src='pt', dest='en').text
    pred = pipeline.predict([traduzido])[0]
    prob = pipeline.predict_proba([traduzido])[0]
    return {
        "texto_original": texto_pt,
        "traduzido": traduzido,
        "sentimento": "positivo" if pred == 1 else "negativo",
        "probabilidade_positivo": round(prob[1], 3)
    }

# Interface no Streamlit
st.title("Classificador de Sentimento")
st.write("Digite um texto em português e veja se ele é positivo ou negativo.")

input_texto = st.text_area("Seu texto aqui:", "")

if st.button("Analisar"):
    if input_texto.strip() == "":
        st.warning("Digite algum texto para análise.")
    else:
        resultado = classificar_texto_portugues(input_texto)
        st.markdown(f"**Sentimento:** {resultado['sentimento'].upper()}")
        st.markdown(f"**Probabilidade de ser positivo:** {resultado['probabilidade_positivo']}")
        st.markdown(f"**Tradução usada:** {resultado['traduzido']}")
