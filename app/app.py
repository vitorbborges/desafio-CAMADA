import sys
from pathlib import Path
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from langchain_core.documents import Document
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_application import LLMAccountant, doc2lancamento, row2doc

openai_api_key = os.getenv("OPENAI_API_KEY")

if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'accountant' not in st.session_state:
    # Preparation
    df = pd.read_csv("data/input_com_categorias.csv")
    train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)
    
    accountant = LLMAccountant()
    docs = [Document(page_content=row["Descrição da Transação"]+"; Valor: R$"+str(row["Valor"]), metadata={
        "category": row["Conta Contábil"],
        "date": row['Data'],
        "value": row['Valor']
    }) for _, row in train_df.iterrows()]
    accountant.add_source_of_truth(docs)
    
    st.session_state.accountant = accountant
    st.session_state.test_rows = list(test_df.iterrows())


# UI
with st.sidebar:
    if not openai_api_key:
        openai_api_key = st.text_input(
            "OpenAI API Key 🔐", key="langchain_search_api_key_openai", type="password"
        )
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

    st.subheader("🧑‍💻 Workflow")
    try:
        graph_bytes = st.session_state.accountant.plot_graph()
        st.image(graph_bytes, caption="LLM Accountant Workflow")
    except Exception as e:
        st.error(f"Could not load workflow graph: {e}")

st.title("🧾 IA do Contador - MVP")

st.progress((st.session_state.current_index + 1) / len(st.session_state.test_rows), text="Progresso de Revisão")

if st.session_state.current_index < len(st.session_state.test_rows):
    idx, row = st.session_state.test_rows[st.session_state.current_index]
    
    doc = row2doc((idx, row))
    lancamento = doc2lancamento(doc)
    response = st.session_state.accountant.invoke(lancamento)
    category = response['category']
    
    data = {
        'Descrição': [''.join(response["desc"].split(';')[:-1])],
        'Valor': [f"R${response['value']}"],
        'Conta Contábil': [category.category],
        'Justificativa LLM': [category.explanation],
        'Status': [response['status']],
        'Data': [response["date"]],
    }
    
    dataframe = pd.DataFrame(data).set_index('Data')
    st.markdown(dataframe.to_markdown())
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Aprovar Conta Contábil", key="approve_btn", icon="✅", use_container_width=True):
            st.success("Lançamento aprovado!")
            lancamento.status = "Confirmado"
            st.session_state.accountant.add_source_of_truth(doc)
            st.session_state.current_index += 1
            st.rerun()
    
    with col2:
        if st.button("Rejeitar Conta Contábil", key="reject_btn", icon="❌", use_container_width=True):
            st.error("Lançamento rejeitado!")
            st.session_state.show_input = True
            st.rerun()
    
    if st.session_state.get('show_input', False):
        true_category = st.text_input(
            "Por favor, inclua a verdadeira categoria contábil:",
            key="true_category_input"
        )
        if st.button("Confirmar", key="confirm_btn"):
            doc.metadata['category'] = true_category
            lancamento.status = "Alterado"
            st.session_state.accountant.add_source_of_truth(doc)
            st.session_state.show_input = False
            st.session_state.current_index += 1
            st.rerun()