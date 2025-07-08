import os
import json
import numpy as np
import faiss
from unidecode import unidecode
from sentence_transformers import SentenceTransformer

# Lista dos arquivos JSON de conceitos
arquivos = [
    'base_consolidada.json',
    'base_empresarial.json',
    'base_gerais.json',
    'contabilidade_conhecimento.json'
]

# Diretório base do arquivo
base_dir = os.path.dirname(os.path.abspath(__file__))
conceitos = []

# Carregar todos os conceitos dos arquivos
for arquivo in arquivos:
    caminho = os.path.join(base_dir, arquivo)
    try:
        with open(caminho, 'r', encoding='utf-8') as f:
            dados = json.load(f)
            if isinstance(dados, list):
                # Garantir que o dado seja dict e tenha os campos 'definicao' e 'termo'
                conceitos.extend([
                    d for d in dados if isinstance(d, dict) and 'definicao' in d and 'termo' in d
                ])
            else:
                print(f"Aviso: arquivo {arquivo} não contém uma lista.")
    except Exception as e:
        print(f"Erro ao carregar {arquivo}: {e}")

if not conceitos:
    raise ValueError("Nenhum conceito carregado")

# Inicializa o modelo para embeddings
modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Gera os textos base para vetorizar (definição + exemplo)
textos_base = [
    unidecode(c['definicao'] + " " + c.get('exemplo', '')).lower() for c in conceitos
]

# Cria embeddings e normaliza
embeddings = modelo.encode(textos_base, normalize_embeddings=True).astype('float32')

# Cria índice FAISS para busca por similaridade (Inner Product)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

def  responder_semantico_faiss(pergunta_usuario: str) -> str:
    """
    Recebe uma pergunta, cria embedding e busca os conceitos mais semelhantes usando FAISS.
    Retorna texto formatado com os 3 conceitos mais próximos.
    """
    pergunta_processada = unidecode(pergunta_usuario.lower())
    embedding = modelo.encode([pergunta_processada], normalize_embeddings=True).astype('float32')

    top_k = min(3, len(conceitos))
    if top_k == 0:
        return "Base de conceitos vazia."

    distancias, indices = index.search(embedding, top_k)

    resposta = ""
    for i in indices[0]:
        conceito = conceitos[i]
        resposta += f"📌 **{conceito['termo']}** ({conceito.get('categoria', 'Sem categoria')})\n"
        resposta += f"🧠 {conceito['definicao']}\n"
        exemplo = conceito.get('exemplo', '')
        if exemplo:
            resposta += f"💡 Exemplo: {exemplo}\n"
        resposta += "\n"
    return resposta.strip()