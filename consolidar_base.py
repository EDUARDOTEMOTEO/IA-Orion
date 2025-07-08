def consolidar_termos(arquivos_json):
    termos_unicos = {}
    definicoes_vistas = set()
    duplicados_termo = []
    duplicados_definicao = []

    for arquivo in arquivos_json:
        if not os.path.isfile(arquivo):
            print(f"Aviso: arquivo {arquivo} não encontrado.")
            continue
        termos = carregar_termos_de_arquivo(arquivo)
        for termo in termos:
            chave = termo['termo'].strip().lower()
            definicao = termo.get('definicao', termo.get('descricao', '')).strip().lower()

            if chave in termos_unicos:
                # Termo já existe
                definicao_existente = termos_unicos[chave].get('definicao', termos_unicos[chave].get('descricao', '')).strip().lower()
                if definicao != definicao_existente:
                    print(f"⚠️ Conflito para termo '{termo['termo']}': definições diferentes encontradas.")
                duplicados_termo.append(termo['termo'])
                continue

            if definicao in definicoes_vistas:
                # Definição já foi usada para outro termo
                duplicados_definicao.append(termo['termo'])
                continue

            # Se passou as verificações, adiciona termo e definição
            termos_unicos[chave] = termo
            definicoes_vistas.add(definicao)

    if duplicados_termo:
        print(f"\n⚠️ Termos duplicados encontrados ({len(duplicados_termo)}):")
        for d in duplicados_termo:
            print(f" - {d}")

    if duplicados_definicao:
        print(f"\n⚠️ Termos com definições repetidas ({len(duplicados_definicao)}):")
        for d in duplicados_definicao:
            print(f" - {d}")

    return list(termos_unicos.values())