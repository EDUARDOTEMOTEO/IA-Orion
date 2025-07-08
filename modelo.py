import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from memoria import recuperar_todos_exemplos  # Certifique-se que este módulo existe e funcione
import os
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
from flask import Flask, request, jsonify

# Configurações
MAX_FEATURES = 1000
MODELO_PATH = "modelo.pt"
VECTORIZER_PATH = "vectorizer.pkl"
ENCODER_PATH = "label_encoder.pkl"

# Inicialização global (serão carregados/salvos)
vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
label_encoder = LabelEncoder()

# Lemmatizer e Stop Words para pré-processamento
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def lematizar_texto(texto):
    return " ".join([
        lemmatizer.lemmatize(word.lower())
        for word in texto.split()
        if word.lower() not in stop_words and word.isalpha()
    ])

class RedeIA(nn.Module):
    def __init__(self, input_size, num_classes):
        super(RedeIA, self).__init__()
        self.layer1 = nn.Linear(input_size, 256)
        self.relu = nn.LeakyReLU()
        self.layer2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)

def salvar_modelo(modelo):
    torch.save(modelo.state_dict(), MODELO_PATH)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)
    print("Modelo salvo com sucesso.")

def carregar_modelo():
    global vectorizer, label_encoder
    if not (os.path.exists(MODELO_PATH) and os.path.exists(VECTORIZER_PATH) and os.path.exists(ENCODER_PATH)):
        print("Modelos não encontrados para carregar.")
        return None, None, None

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    dados = recuperar_todos_exemplos()
    if not dados:
        print("Nenhum dado para carregar modelo.")
        return None, None, None

    perguntas, respostas = zip(*dados)
    perguntas = [lematizar_texto(p) for p in perguntas]
    X = vectorizer.transform(perguntas).toarray()
    y = label_encoder.transform(respostas)

    input_size = X.shape[1]
    num_classes = len(set(y))

    modelo = RedeIA(input_size, num_classes)
    modelo.load_state_dict(torch.load(MODELO_PATH))
    modelo.eval()
    print("Modelo carregado com sucesso.")
    return modelo, label_encoder, vectorizer

def treinar_ia():
    dados = recuperar_todos_exemplos()
    if not dados:
        print("Nenhum dado para treino.")
        return None, None, None

    perguntas, respostas = zip(*dados)
    perguntas = [lematizar_texto(p) for p in perguntas]
    X = vectorizer.fit_transform(perguntas).toarray()
    y = label_encoder.fit_transform(respostas)

    input_size = X.shape[1]
    num_classes = len(set(y))
    modelo = RedeIA(input_size, num_classes)

    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=0.01)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    ultima_perda = float('inf')
    for epoca in range(100):
        modelo.train()
        otimizador.zero_grad()
        predicoes = modelo(X_tensor)
        perda = criterio(predicoes, y_tensor)
        perda.backward()
        otimizador.step()

        print(f"Época {epoca+1}/100 - Perda: {perda.item():.4f}")

        # Early stopping simples
        if perda.item() > ultima_perda:
            print("Parando treino antecipadamente devido à perda não melhorar.")
            break
        ultima_perda = perda.item()

    salvar_modelo(modelo)
    modelo.eval()
    print("Treinamento concluído.")
    return modelo, label_encoder, vectorizer

def prever_resposta(modelo, pergunta, encoder, vectorizer):
    pergunta = lematizar_texto(pergunta)
    X = vectorizer.transform([pergunta]).toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        saida = modelo(X_tensor)
    indice = torch.argmax(saida, dim=1).item()
    return encoder.inverse_transform([indice])[0]

def prever_resposta_interativa(modelo, pergunta, encoder, vectorizer):
    resposta = prever_resposta(modelo, pergunta, encoder, vectorizer)
    respostas_adicionais = [
        "Isso ajuda você? Gostaria de mais detalhes?",
        "Posso explicar mais sobre isso? Como posso ajudar?",
        "Há algo mais que você gostaria de saber?"
    ]
    return f"A resposta é: '{resposta}'. {random.choice(respostas_adicionais)}"

def prever_resposta_com_erro(modelo, pergunta, encoder, vectorizer):
    try:
        if modelo is None:
            return "Modelo não está carregado. Por favor, treine o modelo primeiro."
        return prever_resposta_interativa(modelo, pergunta, encoder, vectorizer)
    except Exception as e:
        print(f"Erro na predição: {e}")
        return "Desculpe, não consegui entender. Poderia reformular ou perguntar de outra maneira?"

def atualizar_memoria(memoria, pergunta, resposta):
    memoria.append((pergunta, resposta))
    if len(memoria) > 10:
        memoria.pop(0)
    return memoria

# Flask app
app = Flask(__name__)

memoria_usuario = []

@app.route('/perguntar', methods=['POST'])
def perguntar():
    dados = request.json
    pergunta = dados.get('pergunta', '').strip()
    if not pergunta:
        return jsonify({'erro': 'Pergunta vazia'}), 400

    modelo, encoder, vectorizer_local = carregar_modelo()
    if modelo is None:
        modelo, encoder, vectorizer_local = treinar_ia()
        if modelo is None:
            return jsonify({'erro': 'Não há dados para treinar o modelo.'}), 500

    resposta = prever_resposta_com_erro(modelo, pergunta, encoder, vectorizer_local)
    atualizar_memoria(memoria_usuario, pergunta, resposta)
    return jsonify({'resposta': resposta, 'memoria': memoria_usuario[-5:]})

def responder_ia(pergunta, modelo, encoder, vectorizer):
    return prever_resposta_com_erro(modelo, pergunta, encoder, vectorizer)

if __name__ == '__main__':
    app.run(debug=True)