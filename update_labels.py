import os

# --- Configuração dos Diretórios ---
LABEL_DIR = "labels"

def update_labels_to_class_zero():
    """Varre todos os arquivos de anotação e muda a classe 1 para 0."""
    print(f"--- Iniciando a atualização de anotações no diretório '{LABEL_DIR}' ---")

    if not os.path.exists(LABEL_DIR):
        print(f"ERRO: Diretório de anotações '{LABEL_DIR}' não encontrado. Abortando.")
        return

    updated_files_count = 0
    processed_files_count = 0

    for filename in os.listdir(LABEL_DIR):
        if not filename.endswith(".txt"):
            continue

        processed_files_count += 1
        file_path = os.path.join(LABEL_DIR, filename)
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            was_modified = False
            for line in lines:
                # Verifica se a linha não está vazia e começa com '1 '
                if line.strip().startswith('1 '):
                    # Substitui o primeiro caractere por '0'
                    new_lines.append('0' + line[1:])
                    was_modified = True
                else:
                    new_lines.append(line)
            
            if was_modified:
                with open(file_path, 'w') as f:
                    f.writelines(new_lines)
                print(f"Atualizado: {filename}")
                updated_files_count += 1

        except Exception as e:
            print(f"ERRO ao processar o arquivo {filename}: {e}")

    print("\n--- Atualização Concluída ---")
    print(f"Total de arquivos processados: {processed_files_count}")
    print(f"Arquivos atualizados (classe 1 para 0): {updated_files_count}")

if __name__ == "__main__":
    update_labels_to_class_zero()
