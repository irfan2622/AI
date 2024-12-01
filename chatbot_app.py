import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Fungsi untuk memuat data
def load_data(filepath='chatbot_data.pkl'):
    with open(filepath, 'rb') as f:
        index, sentence_model, sentences, summaries = pickle.load(f)
    return index, sentence_model, sentences, summaries

# Fungsi chatbot
def chatbot(queries, index, sentence_model, sentences, summaries, top_k=3):
    # Encode queries menjadi embeddings
    query_embeddings = sentence_model.encode(queries)

    # Pastikan embeddings berbentuk 2D
    if len(query_embeddings.shape) == 1:
        query_embeddings = np.expand_dims(query_embeddings, axis=0)

    # FAISS search
    try:
        D, I = index.search(query_embeddings, k=top_k)  # Melakukan pencarian di FAISS index
    except Exception as e:
        st.error(f"Terjadi kesalahan saat pencarian FAISS: {e}")
        return [f"Kesalahan saat memproses pertanyaan '{query}'." for query in queries]

    responses = []
    for query, indices in zip(queries, I):
        relevant_sentences = []
        relevant_summaries = []

        for idx in indices:
            if 0 <= idx < len(sentences):
                relevant_sentences.append(sentences[idx])
                summary = summaries[idx] if idx < len(summaries) and summaries[idx] != "Ringkasan tidak tersedia." else ""
                if summary:
                    relevant_summaries.append(summary)

        if relevant_sentences:
            combined_sentences = " ".join(relevant_sentences[:2])
            combined_summaries = " ".join(relevant_summaries[:2]) if relevant_summaries else "Tidak ada ringkasan yang relevan."

            response = (
                f"**Pertanyaan:** {query}\n\n"
                f"**Jawaban:** {combined_sentences}\n\n"
                f"{f'**Ringkasan:** {combined_summaries}' if relevant_summaries else ''}"
            )
        else:
            response = f"**Pertanyaan:** {query}\n\n**Jawaban:** Tidak ada konten relevan yang ditemukan."

        responses.append(response)
    return responses

# Streamlit App
def main():
    st.title("Chatbot AI")
    st.write("Interaksi dengan dokumen Anda menggunakan AI.")

    # Memuat data chatbot
    st.sidebar.title("Konfigurasi")
    data_path = st.sidebar.text_input("Path ke file data (chatbot_data.pkl)", "chatbot_data.pkl")

    try:
        index, sentence_model, sentences, summaries = load_data(data_path)
        st.sidebar.success("Data berhasil dimuat.")
    except Exception as e:
        st.sidebar.error(f"Gagal memuat data: {e}")
        st.stop()

    # Input pertanyaan pengguna
    queries = st.text_area("Masukkan pertanyaan Anda (pisahkan dengan ';' untuk pertanyaan ganda):")
    if st.button("Ajukan Pertanyaan"):
        if not queries.strip():
            st.warning("Masukkan setidaknya satu pertanyaan.")
        else:
            queries_list = queries.split(";")
            responses = chatbot(queries_list, index, sentence_model, sentences, summaries)
            for response in responses:
                st.markdown(response)

if __name__ == '__main__':
    main()
