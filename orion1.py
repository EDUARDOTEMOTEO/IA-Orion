import os
import re
import json
from typing import Optional, List, Dict, Union
import pandas as pd
from flask import Flask, jsonify, request, render_template, session
from rapidfuzz import process, fuzz
from sympy import sympify
from memoria import inicializar_banco, salvar_mensagem
from modelo import carregar_modelo, treinar_ia
from modelo_semantico import responder_semantico_faiss

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Chave para sessões

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limite upload

# Globais IA
modelo_ia = None
encoder_ia = None
vectorizer_ia = None

VARIACOES_QUANTO = {"quato", "quantu", "qunto", "quntu", "quanto"}
TERMS_CHAVE = {"calcule", "calcular", "quanto", "resultado", "valor", "soma", "subtração", "multiplicação", "divisão"}

SUBSTITUIR_OPERADORES = {
    "vezes": "*",
    "mais": "+",
    "menos": "-",
    "dividido por": "/",
    "dividido": "/",
    "multiplicado por": "*"
}

base_perguntas = {
    "o que é investimento": "Investimento é aplicar dinheiro para obter retorno futuro.",
    "como economizar": "Economizar significa gastar menos do que você ganha e guardar a diferença.",
    "dicas financeiras": "Planeje seu orçamento, evite dívidas e invista regularmente.",
    "qual meu gasto total": "Seu gasto total é a soma de todos os seus gastos registrados.",
    "gastos por categoria": "Posso mostrar os gastos separados por categoria se quiser.",
    "economia": "Economia é a ciência que estuda como as pessoas e sociedades utilizam recursos para produzir, distribuir e consumir bens e serviços.",
    "investimentos": "Investimento é aplicar dinheiro para obter retorno futuro."
}

# --- Memória por usuário usando sessão ---

def obter_memoria_usuario() -> Dict[str, Union[List, Dict]]:
    if 'memoria' not in session:
        session['memoria'] = {
            'bases_json': [],
            'historico': [],
            'tabelas': {},
            'memoria': [],
            'modelo': [],
        }
    return session['memoria']

def salvar_memoria_usuario(memoria: Dict[str, Union[List, Dict]]):
    session['memoria'] = memoria

def carregar_bases() -> List[dict]:
    arquivos = [
        'base_consolidada.json',
        'base_empresarial.json',
        'base_gerais.json',
        'contabilidade_conhecimento.json'
    ]
    bases = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for arq in arquivos:
        caminho = os.path.join(base_dir, arq)
        try:
            with open(caminho, 'r', encoding='utf-8') as f:
                dados = json.load(f)
                if isinstance(dados, list):
                    bases.extend(dados)
                    print(f"Carregou {len(dados)} itens de {arq}")
                else:
                    print(f"Aviso: arquivo {arq} não contém uma lista")
        except Exception as e:
            print(f"Erro ao carregar {arq}: {e}")
    return bases

def inicializar_memoria_usuario():
    memoria = obter_memoria_usuario()
    memoria['bases_json'] = carregar_bases()
    salvar_memoria_usuario(memoria)

# --- Funções utilitárias ---

def nome_valido(nome: str) -> bool:
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_-]*$', nome))

def gerar_nome_tabela_temp(prefix: str = "upload") -> str:
    memoria = obter_memoria_usuario()
    i = 1
    while f"{prefix}_{i}" in memoria['tabelas']:
        i += 1
    return f"{prefix}_{i}"

def corrigir_termos(texto: str) -> str:
    for termo, operador in SUBSTITUIR_OPERADORES.items():
        texto = re.sub(rf'\b{re.escape(termo)}\b', operador, texto, flags=re.IGNORECASE)
    palavras = texto.split()
    palavras_corrigidas = []
    for p in palavras:
        if p.lower() in VARIACOES_QUANTO:
            palavras_corrigidas.append("quanto")
        else:
            palavras_corrigidas.append(p)
    return " ".join(palavras_corrigidas)

def extrair_expressao(texto: str) -> str:
    texto_corrigido = corrigir_termos(texto.lower())
    for termo in TERMS_CHAVE:
        texto_corrigido = re.sub(r'\b' + re.escape(termo) + r'\b', '', texto_corrigido)
    texto_corrigido = re.sub(r'[^0-9+\-*/()., ]', '', texto_corrigido)
    texto_corrigido = texto_corrigido.replace(',', '.').strip()
    return texto_corrigido

def formatar_resultado(resultado: float) -> str:
    if resultado == int(resultado):
        return str(int(resultado))
    return f"{resultado:.6f}".rstrip('0').rstrip('.')

# --- Funções principais ---

def buscar_em_bases(pergunta: str) -> Optional[str]:
    memoria = obter_memoria_usuario()
    termos = [item.get('termo', '').lower() for item in memoria['bases_json']]
    print(f"Buscando em bases para a pergunta: '{pergunta}'")
    resultado = process.extract(pergunta.lower(), termos, scorer=fuzz.token_sort_ratio, limit=1)
    print(f"Resultado do fuzzy match: {resultado}")
    print(f"Total de termos carregados: {len(termos)}")

    respostas = []
    for termo, score, idx in resultado:
        if score >= 75:
            item = memoria['bases_json'][idx]
            resposta = f"📘 *{item.get('termo','')}*"
            if item.get('categoria'):
                resposta += f" ({item['categoria']})"
            resposta += f":\n{item.get('definicao', 'Sem definição disponível.')}"
            if item.get('exemplo'):
                resposta += f"\nExemplo: {item['exemplo']}"
            respostas.append(resposta)
    if respostas:
        return "\n\n".join(respostas)
    return None

def responder_pergunta_semantica(pergunta: str) -> Optional[str]:
    matches = process.extract(pergunta.lower(), base_perguntas.keys(), scorer=fuzz.token_sort_ratio, limit=1)
    if matches and matches[0][1] >= 75:
        return base_perguntas[matches[0][0]]
    return None

def criar_tabela(nome_tabela: str, colunas: List[str]) -> str:
    memoria = obter_memoria_usuario()
    nome_tabela = nome_tabela.lower()
    if not nome_valido(nome_tabela):
        return "Nome de tabela inválido. Use letras, números, underline e hífen apenas."
    if nome_tabela in memoria['tabelas']:
        return f"A tabela '{nome_tabela}' já existe. Escolha outro nome ou modifique a tabela existente."
    if not colunas or any(not c for c in colunas):
        return "Erro: A lista de colunas está vazia ou contém valores inválidos. Tente novamente."
    df = pd.DataFrame(columns=colunas)
    memoria['tabelas'][nome_tabela] = df
    salvar_memoria_usuario(memoria)
    return f"Tabela '{nome_tabela}' criada com sucesso! Colunas: {', '.join(colunas)}."

def inserir_na_tabela(nome_tabela: str, valores: List[str]) -> str:
    memoria = obter_memoria_usuario()
    nome_tabela = nome_tabela.lower()
    if nome_tabela not in memoria['tabelas']:
        return f"A tabela '{nome_tabela}' não existe. Crie-a antes de inserir dados."
    df = memoria['tabelas'][nome_tabela]
    if len(valores) != len(df.columns):
        return f"Erro: A tabela '{nome_tabela}' tem {len(df.columns)} colunas, mas você forneceu {len(valores)} valores."
    nova_linha = dict(zip(df.columns, valores))
    df.loc[len(df)] = nova_linha
    memoria['tabelas'][nome_tabela] = df
    salvar_memoria_usuario(memoria)
    return f"Dados inseridos com sucesso na tabela '{nome_tabela}'."

def consultar_tabela(nome_tabela: str, filtro: Optional[str] = None) -> str:
    memoria = obter_memoria_usuario()
    nome_tabela = nome_tabela.lower()
    if nome_tabela not in memoria['tabelas']:
        return f"Tabela '{nome_tabela}' não encontrada."
    df = memoria['tabelas'][nome_tabela]
    if filtro:
        try:
            df_filtrado = df.query(filtro)
        except Exception as e:
            return f"Erro no filtro: {e}"
        return df_filtrado.to_string(index=False)
    return df.to_string(index=False)

def responder_ia(pergunta: str, modelo, encoder, vectorizer) -> Optional[str]:
    # Aqui deve ser implementada a função que irá interagir com o modelo de IA
    pass

def processar_pergunta(pergunta: str) -> str:
    # 1. Criar tabela
    m_criar = re.search(r'criar tabela ([\w\-]+)\s*\((.*?)\)', pergunta, re.I)
    if m_criar:
        nome_tabela = m_criar.group(1).strip().lower()
        colunas = [c.strip() for c in re.split(r',\s*', m_criar.group(2)) if c.strip()]
        return criar_tabela(nome_tabela, colunas)

    # 2. Inserir dados na tabela
    m_inserir = re.search(r'inserir na tabela ([\w\-]+)\s*\((.*?)\)', pergunta, re.I)
    if m_inserir:
        nome_tabela = m_inserir.group(1).strip().lower()
        valores = [v.strip().strip('"\'') for v in re.split(r',\s*', m_inserir.group(2)) if v.strip()]
        return inserir_na_tabela(nome_tabela, valores)

    # 3. Consultar tabela
    m_consultar = re.search(r'consultar tabela ([\w\-]+)(?: onde (.*))?', pergunta, re.I)
    if m_consultar:
        nome_tabela = m_consultar.group(1).strip().lower()
        filtro = m_consultar.group(2)
        return consultar_tabela(nome_tabela, filtro)

    # 4. Cálculos
    expr = extrair_expressao(pergunta)
    if expr:
        try:
            resultado = float(sympify(expr).evalf())
            return f"O resultado do cálculo é: {formatar_resultado(resultado)}"
        except Exception as e:
            print(f"Erro ao calcular expressão: {e}")

    # 5. Buscar nas bases
    resposta = buscar_em_bases(pergunta)
    if resposta:
        return resposta

    # 6. Pergunta semântica
    resposta = responder_pergunta_semantica(pergunta)
    if resposta:
        return resposta

    # 7. IA
    global modelo_ia, encoder_ia, vectorizer_ia
    if modelo_ia and encoder_ia and vectorizer_ia:
        resposta = responder_ia(pergunta, modelo_ia, encoder_ia, vectorizer_ia)
        if resposta:
            return resposta

    return "Desculpe, não consegui encontrar uma resposta para sua pergunta."
@app.route("/")
def index():
    inicializar_memoria_usuario()  # Correto: agora dentro do contexto HTTP
    return render_template("index.html")


@app.route("/treinar", methods=["POST"])
def treinar_modelo():
    data = request.json
    pergunta = data.get('pergunta', '').strip()
    resposta = data.get('resposta', '').strip()
    if not pergunta or not resposta:
        return jsonify({"resposta": "Por favor, forneça tanto uma pergunta quanto uma resposta."})
    
    salvar_mensagem("usuário", pergunta, resposta)
    global modelo_ia, encoder_ia, vectorizer_ia
    modelo_ia, encoder_ia, vectorizer_ia = treinar_ia()
    
    return jsonify({"resposta": f"Modelo treinado com sucesso com a pergunta: '{pergunta}'."})


@app.route("/pergunta", methods=["POST"])
def perguntar():
    data = request.json
    pergunta = data.get('pergunta', '').strip()
    if not pergunta:
        return jsonify({"resposta": "Por favor, faça uma pergunta válida."})

    resposta = processar_pergunta(pergunta)
    salvar_mensagem("usuário", pergunta, resposta)
    return jsonify({"resposta": resposta})


@app.route("/upload", methods=["POST"])
def upload_arquivo():
    arquivo = request.files.get('arquivo')
    if not arquivo:
        return jsonify({"resposta": "Nenhum arquivo enviado."})
    
    filename = arquivo.filename.lower()
    try:
        if filename.endswith('.json'):
            conteudo = arquivo.read().decode('utf-8')
            dados = json.loads(conteudo)
            if isinstance(dados, list):
                memoria = obter_memoria_usuario()
                memoria['bases_json'].extend(dados)
                salvar_memoria_usuario(memoria)
                return jsonify({"resposta": f"Base JSON com {len(dados)} itens adicionada com sucesso!"})
            else:
                return jsonify({"resposta": "Arquivo JSON deve conter uma lista de itens."})

        elif filename.endswith('.csv') or filename.endswith('.xls') or filename.endswith('.xlsx'):
            if filename.endswith('.csv'):
                df = pd.read_csv(arquivo)
            else:
                df = pd.read_excel(arquivo)
            
            nome_tabela = gerar_nome_tabela_temp()
            memoria = obter_memoria_usuario()
            memoria['tabelas'][nome_tabela] = df
            salvar_memoria_usuario(memoria)

            return jsonify({"resposta": f"Arquivo carregado como tabela temporária '{nome_tabela}' com {len(df)} linhas e {len(df.columns)} colunas."})
        
        else:
            return jsonify({"resposta": "Tipo de arquivo não suportado. Use JSON, CSV ou XLS/XLSX."})

    except Exception as e:
        return jsonify({"resposta": f"Erro ao processar arquivo: {str(e)}"})


# Inicializa banco de dados fora do contexto HTTP
if __name__ == "__main__":
    from memoria import inicializar_banco
    inicializar_banco()  # Pode rodar fora do contexto de requisição
    app.run(debug=True, port=5000)