import sqlite3
import csv
from datetime import datetime

DB_NAME = "banco.db"

def conectar():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def inicializar_banco():
    with conectar() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS conversa (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario TEXT,
                pergunta TEXT,
                resposta TEXT,
                categoria TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

def salvar_mensagem(usuario, pergunta, resposta, categoria="geral"):
    with conectar() as conn:
        conn.execute(
            "INSERT INTO conversa (usuario, pergunta, resposta, categoria) VALUES (?, ?, ?, ?)",
            (usuario, pergunta, resposta, categoria)
        )
        conn.commit()

def recuperar_todos_exemplos():
    with conectar() as conn:
        cursor = conn.execute("SELECT pergunta, resposta FROM conversa")
        return cursor.fetchall()

def recuperar_memoria_recente(limite=5):
    with conectar() as conn:
        cursor = conn.execute(
            "SELECT pergunta, resposta FROM conversa ORDER BY timestamp DESC LIMIT ?", (limite,))
        return cursor.fetchall()

def recuperar_memoria_por_categoria(categoria, limite=5):
    with conectar() as conn:
        cursor = conn.execute(
            "SELECT pergunta, resposta FROM conversa WHERE categoria = ? ORDER BY timestamp DESC LIMIT ?", (categoria, limite))
        return cursor.fetchall()

def recuperar_ultima_interacao():
    with conectar() as conn:
        cursor = conn.execute("SELECT pergunta, resposta FROM conversa ORDER BY timestamp DESC LIMIT 1")
        return cursor.fetchone()

def limpar_banco():
    with conectar() as conn:
        conn.execute("DELETE FROM conversa")
        conn.commit()

def realizar_backup():
    with conectar() as conn:
        cursor = conn.execute("SELECT * FROM conversa")
        dados = cursor.fetchall()
        colunas = [desc[0] for desc in cursor.description]
        with open('backup_memoria.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(colunas)
            writer.writerows([tuple(row) for row in dados])
    print("Backup realizado com sucesso.")

def limpar_memoria_antiga(dias=30):
    with conectar() as conn:
        conn.execute("DELETE FROM conversa WHERE timestamp < datetime('now', ?)", (f'-{dias} days',))
        conn.commit()
    print(f"Memória limpa de interações com mais de {dias} dias.")