import os
import random
import shutil

import yaml
from ultralytics import YOLO

# --- Parâmetros Configuráveis pelo Usuário ---

# mlflow server --backend-store-uri runs/mlflow

# 1. Diretórios
SOURCE_IMAGE_DIR = "labeled"
SOURCE_LABEL_DIR = "labels"
DATASET_ROOT_DIR = "dataset"

# 2. Modelo e Treinamento
MODEL_CONFIG = "models/yolo11n-obb.pt"
DATASET_YAML = os.path.join(DATASET_ROOT_DIR, "dataset.yaml")
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 8
PATIENCE = 10
# Nome do Experimento no MLflow (será criado se não existir)
MLFLOW_EXPERIMENT_NAME = "YOLOv11n_OBB_Training"
# Nome desta Run específica (mude a cada novo treino para melhor organização)
MLFLOW_RUN_NAME = "treino_run_1"

# 3. Divisão do Dataset (Train/Validation/Test)
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

# 4. Classes do Modelo
CLASS_NAMES = ["pig"]

# ---------------------------------------------


def prepare_dataset():
    """Cria a estrutura de pastas e divide os dados em treino, validação e teste."""
    print("--- Iniciando a preparação do dataset ---")

    if not os.path.exists(SOURCE_IMAGE_DIR) or not os.listdir(SOURCE_IMAGE_DIR):
        print(
            f"ERRO: A pasta de origem '{SOURCE_IMAGE_DIR}' está vazia ou não existe. Abortando."
        )
        return False
    if not os.path.exists(SOURCE_LABEL_DIR):
        print(f"ERRO: A pasta de anotações '{SOURCE_LABEL_DIR}' não existe. Abortando.")
        return False

    if os.path.exists(DATASET_ROOT_DIR):
        print(f"Limpando diretório de dataset existente: '{DATASET_ROOT_DIR}'")
        shutil.rmtree(DATASET_ROOT_DIR)

    paths = {}
    for split in SPLIT_RATIOS.keys():
        paths[split] = {
            "images": os.path.join(DATASET_ROOT_DIR, "images", split),
            "labels": os.path.join(DATASET_ROOT_DIR, "labels", split),
        }
        os.makedirs(paths[split]["images"])
        os.makedirs(paths[split]["labels"])
    print("Estrutura de pastas do dataset criada.")

    all_label_files = [f for f in os.listdir(SOURCE_LABEL_DIR) if f.endswith(".txt")]
    random.shuffle(all_label_files)

    total_files = len(all_label_files)
    train_end = int(total_files * SPLIT_RATIOS["train"])
    val_end = train_end + int(total_files * SPLIT_RATIOS["val"])

    splits_data = {
        "train": all_label_files[:train_end],
        "val": all_label_files[train_end:val_end],
        "test": all_label_files[val_end:],
    }

    for split, filenames in splits_data.items():
        print(f"Copiando {len(filenames)} arquivos para o conjunto de '{split}'...")
        for label_filename in filenames:
            base_name = os.path.splitext(label_filename)[0]
            image_filename = None
            for ext in [".jpg", ".jpeg", ".png"]:
                if os.path.exists(os.path.join(SOURCE_IMAGE_DIR, base_name + ext)):
                    image_filename = base_name + ext
                    break

            if image_filename:
                shutil.copy(
                    os.path.join(SOURCE_LABEL_DIR, label_filename),
                    paths[split]["labels"],
                )
                shutil.copy(
                    os.path.join(SOURCE_IMAGE_DIR, image_filename),
                    paths[split]["images"],
                )
            else:
                print(
                    f"AVISO: Imagem para a anotação '{label_filename}' não encontrada. Pulando."
                )

    yaml_data = {
        "path": os.path.abspath(DATASET_ROOT_DIR),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }

    with open(DATASET_YAML, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    print(f"Arquivo '{DATASET_YAML}' criado com sucesso.")

    print("--- Preparação do dataset concluída ---")
    return True


def train_model():
    """Carrega o modelo e inicia o treinamento, deixando o YOLOv8 gerenciar o MLflow."""
    print("--- Iniciando o treinamento do modelo YOLOv8-OBB ---")

    model = YOLO(MODEL_CONFIG)

    print(f"Parâmetros: Epochs={EPOCHS}, ImgSize={IMAGE_SIZE}, Batch={BATCH_SIZE}")
    print(f"MLflow: Experimento='{MLFLOW_EXPERIMENT_NAME}', Run='{MLFLOW_RUN_NAME}'")

    # A biblioteca ultralytics gerencia a integração automaticamente
    # passando os nomes do projeto (experimento) e da run.
    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        project=MLFLOW_EXPERIMENT_NAME,
        name=MLFLOW_RUN_NAME,
    )

    print("--- Treinamento concluído ---")
    print(f"Resultados salvos na pasta 'runs/obb/' e registrados no MLflow.")


if __name__ == "__main__":
    if prepare_dataset():
        train_model()
