import os
import zipfile
import chainlit as cl
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai

# Memuatkan FAISS dan Embeddings secara selamat
@cl.on_chat_start
async def start():
    # Nyahmampat fail ZIP FAISS jika ada
    zip_files = [f for f in os.listdir('.') if f.endswith('.zip')]
    extract_dirs = []
    for z_file in zip_files:
        folder_name = z_file.replace('.zip', '').split(' ')[0]
        if not os.path.exists(folder_name):
            with zipfile.ZipFile(z_file, 'r') as zip_ref:
                zip_ref.extractall(folder_name)
        extract_dirs.append(folder_name)

    # Jalankan proses berat FAISS/HuggingFace dalam worker thread supaya ASGI tidak crash
    def load_vectorstores():
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        vectorstores = []
        for folder in ['.'] + extract_dirs:
            for root, _, filenames in os.walk(folder):
                if "index.faiss" in filenames:
                    db = FAISS.load_local(root, embedding_model, allow_dangerous_deserialization=True)
                    vectorstores.append(db)
        return vectorstores

    vectorstores = await cl.make_async(load_vectorstores)()
    cl.user_session.set("vectorstores", vectorstores)
    await cl.Message(content="Sistem Pembantu Klinikal AFib (Multi-Guideline RAG) sedia untuk digunakan. Sila masukkan soalan anda.").send()


@cl.on_message
async def main(message: cl.Message):
    vectorstores = cl.user_session.get("vectorstores")
    user_query = message.content

    # Cipta tindak balas awal
    msg = cl.Message(content="")
    await msg.send()

    try:
        # Jalankan pencarian FAISS secara async
        def search_docs():
            all_retrieved_docs = []
            for db in vectorstores:
                docs = db.similarity_search(user_query, k=3)
                all_retrieved_docs.extend(docs)

            seen = set()
            unique_docs = []
            for doc in all_retrieved_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_docs.append(doc)
            return unique_docs

        unique_docs = await cl.make_async(search_docs)()

        context_text = "\n".join([
            f"- [{doc.metadata.get('source', 'Guideline')} | {doc.metadata.get('section', 'General')}] {doc.page_content}"
            for doc in unique_docs[:5]
        ])

        system_prompt = f"""
You are a Board-Certified Clinical Specialist Professor in Atrial Fibrillation (AFib) Pharmacotherapy. 

YOUR OBJECTIVE:
Provide structured, highly accurate clinical recommendations in response to the user's query using ONLY the provided official guideline context chunks. You must attribute every clinical statement to its exact source guideline and section header.

==================================================
STRICT CLINICAL SAFETY RULES:
==================================================
1. STRICT GROUNDING: Base every recommendation solely on the CONTEXT DATA below. Do NOT use outside medical knowledge or unvalidated general assumptions.
2. ABSENCE OF EVIDENCE: If the provided CONTEXT DATA does not contain sufficient information to safely answer the user's query, explicitly state: "The retrieved guideline context does not contain sufficient clinical guidance to answer this specific query."
3. MANDATORY IN-LINE CITATIONS: Every clinical recommendation MUST be followed immediately by an inline bracketed citation containing the Guideline ID and Section Name [Guideline_ID | Section_Header].

==================================================
CONTEXT DATA FROM GUIDELINES:
{context_text}

==================================================
USER QUERY: {user_query}
ANSWER:
"""

        # Jalankan panggilan API Gemini secara async
        def call_gemini():
            api_key = os.environ.get("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key)
            return client.models.generate_content(
                model='gemini-3.6-flash',
                contents=system_prompt
            )

        response = await cl.make_async(call_gemini)()
        
        msg.content = response.text
        await msg.update()

    except Exception as e:
        msg.content = f"Ralat berlaku: {str(e)}"
        await msg.update()
