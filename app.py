import os
import zipfile
import asyncio
import chainlit as cl
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai

# Fungsi penyinkronan untuk memuatkan FAISS
def sync_load_vectorstores():
    zip_files = [f for f in os.listdir('.') if f.endswith('.zip')]
    extract_dirs = []
    for z_file in zip_files:
        folder_name = z_file.replace('.zip', '').split(' ')[0]
        if not os.path.exists(folder_name):
            with zipfile.ZipFile(z_file, 'r') as zip_ref:
                zip_ref.extractall(folder_name)
        extract_dirs.append(folder_name)

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


@cl.on_chat_start
async def start():
    # Memproses FAISS dalam thread berasingan supaya ASGI tidak crash
    vectorstores = await asyncio.to_thread(sync_load_vectorstores)
    cl.user_session.set("vectorstores", vectorstores)
    await cl.Message(content="Sistem Pembantu Klinikal AFib (Multi-Guideline RAG) sedia digunakan. Sila kemukakan soalan anda.").send()


# Fungsi penyinkronan untuk pencarian FAISS
def sync_search_docs(vectorstores, user_query):
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


# Fungsi penyinkronan untuk panggilan Gemini API
def sync_call_gemini(system_prompt):
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    return client.models.generate_content(
        model='gemini-3.6-flash',
        contents=system_prompt
    )


@cl.on_message
async def main(message: cl.Message):
    vectorstores = cl.user_session.get("vectorstores")
    user_query = message.content

    msg = cl.Message(content="")
    await msg.send()

    try:
        # Jalankan pencarian FAISS
        unique_docs = await asyncio.to_thread(sync_search_docs, vectorstores, user_query)

        context_text = "\n".join([
            f"- [{doc.metadata.get('source', 'Guideline')} | {doc.metadata.get('section', 'General')}] {doc.page_content}"
            for doc in unique_docs[:5]
        ])

        system_prompt = f"""
You are a Board-Certified Clinical Specialist Professor in Atrial Fibrillation (AFib) Pharmacotherapy. 

YOUR OBJECTIVE:
Provide structured, highly accurate clinical recommendations using ONLY the provided official guideline context chunks.

STRICT CLINICAL SAFETY RULES:
1. STRICT GROUNDING: Base every recommendation solely on CONTEXT DATA below.
2. ABSENCE OF EVIDENCE: If information is missing, state: "The retrieved guideline context does not contain sufficient clinical guidance to answer this specific query."
3. MANDATORY IN-LINE CITATIONS: Every recommendation MUST include [Guideline_ID | Section_Header].

CONTEXT DATA:
{context_text}

USER QUERY: {user_query}
ANSWER:
"""

        # Jalankan panggilan API Gemini
        response = await asyncio.to_thread(sync_call_gemini, system_prompt)
        
        msg.content = response.text
        await msg.update()

    except Exception as e:
        msg.content = f"Ralat berlaku: {str(e)}"
        await msg.update()
