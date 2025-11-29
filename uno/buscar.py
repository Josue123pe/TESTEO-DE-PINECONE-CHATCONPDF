# buscar.py
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Cargar modelo de embeddings
modelo = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Cargar datos ya procesados del PDF
with open("embeddings_pdf.pkl", "rb") as f:
    data = pickle.load(f)

lista_textos = data["textos"]
lista_embeddings = np.array(data["embeddings"])  # Asegurar que es numpy array


# -----------------------------------------
# FUNCION PARA BUSCAR TEXTO EN EL PDF
# -----------------------------------------
def buscar_respuesta(pregunta, top_k=1):
    # Convertimos la pregunta a embedding
    embedding_pregunta = modelo.encode(pregunta)

    # Normalizamos para similitud de coseno
    embedding_pregunta = embedding_pregunta / np.linalg.norm(embedding_pregunta)
    emb_norm = lista_embeddings / np.linalg.norm(lista_embeddings, axis=1, keepdims=True)

    # Calculamos similitudes
    similitudes = np.dot(emb_norm, embedding_pregunta)

    # Índices de los textos más similares
    idx_mejores = np.argsort(similitudes)[::-1][:top_k]

    # Construimos resultados
    resultados = []
    for idx in idx_mejores:
        resultados.append({
            "texto": lista_textos[idx],
            "score": float(similitudes[idx])
        })

    return resultados


# -----------------------------------------
# EJECUCIÓN INTERACTIVA DESDE CONSOLA
# -----------------------------------------
if __name__ == "__main__":
    pregunta = input("❓ Escribe tu pregunta sobre el PDF: ")

    resultados = buscar_respuesta(pregunta, top_k=1)

    print("\n📌 Texto más relevante encontrado en el PDF:\n")
    print(resultados[0]["texto"])
    print(f"\n🔢 Score de similitud: {resultados[0]['score']}")
