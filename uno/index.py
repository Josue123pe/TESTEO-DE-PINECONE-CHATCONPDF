import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# ==========================
# 1. Cargar variables del .env
# ==========================
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")

# ==========================
# 2. Conectar Pinecone
# ==========================
pc = Pinecone(api_key=api_key)
index_name = "chatcitobot"

# ==========================
# 3. Modelo de embeddings FREE
# ==========================
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dimensiones

def generar_embedding(texto):
    return model.encode(texto).tolist()

# ==========================
# 4. Crear índice si no existe
# ==========================
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,    # IMPORTANTE (NO ES 1536)
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )
    print(f"Índice '{index_name}' creado correctamente.")
else:
    print(f"Índice '{index_name}' ya existe.")

# ==========================
# 5. Conectar al índice
# ==========================
index = pc.Index(index_name)
print("Conectado al índice:", index_name)

# ==========================
# 6. Insertar textos
# ==========================
def agregar_textos(lista_textos):
    vectores = []
    for i, txt in enumerate(lista_textos):
        emb = generar_embedding(txt)
        vectores.append((f"vec_{i}", emb, {"texto": txt}))
    index.upsert(vectores)
    print("Textos insertados correctamente!")

# ==========================
# 7. Consultar similitud
# ==========================
def buscar(query):
    emb = generar_embedding(query)
    res = index.query(vector=emb, top_k=3, include_metadata=True)
    return res

# ==========================
# 8. PRUEBA
# ==========================
textos = [
    "El gato duerme en el sillón.",
    "La programación en Python es divertida.",
    "Me gusta entrenar en el gimnasio.",
]

agregar_textos(textos)

resultado = buscar("¿qué animal duerme?")
print("Resultado búsqueda:")
print(resultado)





