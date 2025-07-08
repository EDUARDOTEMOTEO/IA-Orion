import sqlite3

DB_PATH = "base.db"

def conectar():
    return sqlite3.connect(DB_PATH)

def criar_tabela():
    with conectar() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memoria (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pergunta TEXT UNIQUE,
                resposta TEXT
            )
        """)
        conn.commit()

def salvar_memoria(pergunta, resposta):
    with conectar() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO memoria (pergunta, resposta) VALUES (?, ?)
        """, (pergunta, resposta))
        conn.commit()

def buscar_resposta(pergunta):
    with conectar() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT resposta FROM memoria WHERE pergunta = ?", (pergunta,))
        resultado = cursor.fetchone()
        return resultado[0] if resultado else None