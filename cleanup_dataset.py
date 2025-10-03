import os
import shutil

# --- Configuração dos Diretórios ---
IMAGE_DIR = "images"
LABEL_DIR = "labels"
LABELED_DIR = "labeled"

VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

def cleanup_dataset():
    """Verifica as anotações, move as imagens e remove anotações órfãs."""
    print("--- Iniciando a limpeza do dataset ---")

    # Garante que os diretórios existem
    if not os.path.exists(LABELED_DIR):
        print(f"Criando diretório de destino: '{LABELED_DIR}'")
        os.makedirs(LABELED_DIR)

    if not os.path.exists(LABEL_DIR):
        print(f"ERRO: Diretório de anotações '{LABEL_DIR}' não encontrado. Abortando.")
        return

    if not os.path.exists(IMAGE_DIR):
        print(f"AVISO: Diretório de imagens '{IMAGE_DIR}' não encontrado.")

    moved_count = 0
    already_sorted_count = 0
    deleted_labels_count = 0

    # Itera sobre todos os arquivos de anotação
    for label_filename in os.listdir(LABEL_DIR):
        if not label_filename.endswith(".txt"):
            continue

        base_name = os.path.splitext(label_filename)[0]
        found_image = False

        # Procura pela imagem correspondente
        for ext in VALID_IMAGE_EXTENSIONS:
            image_filename = base_name + ext
            source_path = os.path.join(IMAGE_DIR, image_filename)
            dest_path = os.path.join(LABELED_DIR, image_filename)

            if os.path.exists(dest_path):
                already_sorted_count += 1
                found_image = True
                break

            if os.path.exists(source_path):
                try:
                    shutil.move(source_path, dest_path)
                    print(f"Movido: {image_filename}")
                    moved_count += 1
                    found_image = True
                    break
                except Exception as e:
                    print(f"ERRO ao mover {source_path}: {e}")
                    found_image = True
                    break
        
        if not found_image:
            label_path_to_delete = os.path.join(LABEL_DIR, label_filename)
            try:
                os.remove(label_path_to_delete)
                print(f"Removida anotação órfã: {label_filename}")
                deleted_labels_count += 1
            except OSError as e:
                print(f"ERRO ao remover {label_path_to_delete}: {e}")

    print("\n--- Limpeza Concluída ---")
    print(f"Imagens movidas: {moved_count}")
    print(f"Já estavam organizadas: {already_sorted_count}")
    if deleted_labels_count > 0:
        print(f"Anotações órfãs removidas: {deleted_labels_count}")

if __name__ == "__main__":
    cleanup_dataset()
