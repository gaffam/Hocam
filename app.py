import os
import pickle
import streamlit as st
import numpy as np

from speech_utils import text_to_speech, speech_to_text
from score_db import init_db, add_user, add_points, get_scores

from pdf_utils import extract_chunks
from embedder import Embedder
from search import VectorSearch
from answer_generator import AnswerGenerator

def make_paths(grade: int, subject: str):
    subj = subject.lower().replace(" ", "_")
    os.makedirs("index", exist_ok=True)
    os.makedirs("chunks", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)
    index_path = os.path.join("index", f"{grade}_{subj}.index")
    chunk_path = os.path.join("chunks", f"{grade}_{subj}.pkl")
    emb_path = os.path.join("embeddings", f"{grade}_{subj}.npy")
    return index_path, chunk_path, emb_path


def load_chunks(chunk_path: str):
    if os.path.exists(chunk_path):
        with open(chunk_path, "rb") as f:
            return pickle.load(f)
    return None


def load_embeddings(emb_path: str):
    if os.path.exists(emb_path):
        return np.load(emb_path)
    return None


def save_embeddings(embeddings, emb_path: str):
    np.save(emb_path, embeddings)


def save_chunks(chunks, chunk_path: str):
    with open(chunk_path, "wb") as f:
        pickle.dump(chunks, f)


init_db()

st.title("Türkçe Soru-Cevap Uygulaması")
st.markdown(
    "PDF yükleyip indeks oluşturduktan sonra sorularınızı yazabilirsiniz."
)

if "user" not in st.session_state:
    name = st.text_input("Kullanıcı adı")
    role = st.selectbox("Rol", ["Öğrenci", "Veli"])
    if st.button("Giriş") and name:
        st.session_state["user"] = name
        st.session_state["role"] = role
        add_user(name)
        st.experimental_rerun()
    st.stop()

user = st.session_state["user"]
role = st.session_state["role"]
st.sidebar.write(f"Giriş yapan: {user} ({role})")

if role == "Veli":
    st.header("Puan Tablosu")
    for name, pts in get_scores():
        st.write(f"{name}: {pts}")
    st.stop()

grade = st.selectbox("Sınıf", list(range(1, 9)))
subject = st.selectbox(
    "Ders",
    ["Türkçe", "Matematik", "Fen", "Sosyal", "Hayat Bilgisi", "İngilizce"],
)

index_path, chunk_path, emb_path = make_paths(grade, subject)

embedder = Embedder()
chunks = load_chunks(chunk_path)
embeddings = load_embeddings(emb_path)
dim = embeddings.shape[1] if embeddings is not None else 384
vs = VectorSearch(dim=dim, grade=grade, subject=subject)

if chunks is None:
    st.info("Lütfen bir PDF yükleyin ve indeks oluşturun.")
    uploaded = st.file_uploader("PDF Dosyası")
    if uploaded and st.button("İndeks Oluştur"):
        pdf_path = "uploaded.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded.read())
        chunks = extract_chunks(pdf_path, tokenizer_name=embedder.model_name)
        save_chunks(chunks, chunk_path)
        embeddings = embedder.encode_texts(chunks)
        save_embeddings(embeddings, emb_path)
        vs.add_embeddings(embeddings)
        st.success("İndeks kaydedildi.")

question = st.text_input("Sorunuzu yazın")
audio = st.file_uploader("Sesli soru (isteğe bağlı)", type=["wav", "mp3"])
model_choice = st.selectbox(
    "Model seç",
    [
        "dbmdz/gpt2-turkish",
        "AI4Turk/ke-t5-small-tr",
        "cahya/gpt2-small-turkish",
    ],
)

if st.button("Cevapla"):
    if not question and not audio:
        st.warning("Lütfen bir soru girin veya ses yükleyin.")
    elif chunks is None:
        st.error("Önce bir PDF yükleyin ve indeks oluşturun.")
    else:
        if audio:
            audio_path = "temp_audio"
            with open(audio_path, "wb") as f:
                f.write(audio.read())
            question = speech_to_text(audio_path)
            st.info(f"Algılanan soru: {question}")
        query_vec = embedder.encode_query(question)
        if embeddings is not None:
            _, indices = vs.search_with_rerank(query_vec, embeddings, top_k=3)
        else:
            _, indices = vs.search(query_vec, top_k=3)
        selected = [chunks[i] for i in indices if i < len(chunks)]
        generator = AnswerGenerator(model_name=model_choice)
        answer = generator.generate(selected, question, grade=grade, subject=subject)
        st.markdown(f"**Cevap:**\n\n{answer}")
        audio_out = text_to_speech(answer)
        st.audio(audio_out)
        add_points(user)
