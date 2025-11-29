# agregar_pdf.py
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import PyPDF2

# Nombre del PDF
PDF_NAME = "pokemonguia.pdf"

# Leer PDF
def leer_pdf():
    textos = []
    with open(PDF_NAME, "rb") as f:
        lector = PyPDF2.PdfReader(f)
        for pagina in lector.pages:
            texto = pagina.extract_text()
            if texto:
                textos.append(texto)
    return textos

print("📄 Leyendo PDF...")
lista_textos = leer_pdf()

print(f"✔ PDF leído: {len(lista_textos)} páginas")

# Modelo de Embeddings
modelo = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("🔢 Generando embeddings...")
lista_embeddings = modelo.encode(lista_textos)

# Guardar embeddings + textos
with open("embeddings_pdf.pkl", "wb") as f:
    pickle.dump({
        "textos": lista_textos,
        "embeddings": np.array(lista_embeddings)
    }, f)

print("🎉 Listo: embeddings_pdf.pkl creado!")
