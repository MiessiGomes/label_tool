import os

import cv2
from ultralytics import YOLO

# --- Parâmetros Configuráveis ---

# 1. Caminho para o modelo treinado
MODEL_PATH = "YOLOv11n_OBB_Training/treino_run_1/weights/best.pt"

# 2. Caminho para o vídeo de entrada
INPUT_VIDEO_PATH = "videos/20250923180845629_37.mp4"

# 3. Caminho para o vídeo de saída
OUTPUT_VIDEO_PATH = "videos_output/result.mp4"

# 4. Limiar de confiança
CONFIDENCE_THRESHOLD = 0.6

# 5. Cores e Aparência
BOX_COLOR = (0, 255, 0)  # Verde
TEXT_COLOR = (255, 255, 255)  # Branco
BOX_THICKNESS = 2
FONT_SCALE = 0.5


def run_video_inference():
    """Roda o rastreamento em um vídeo e salva o resultado com as detecções OBB."""
    print("--- Iniciando o rastreamento em vídeo ---")

    # 1. Validação dos caminhos
    if not os.path.exists(MODEL_PATH):
        print(f"ERRO: Arquivo do modelo não encontrado em '{MODEL_PATH}'")
        return
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"ERRO: Vídeo de entrada não encontrado em '{INPUT_VIDEO_PATH}'")
        return

    # 2. Carregar o modelo
    print(f"Carregando modelo de '{MODEL_PATH}'...")
    try:
        model = YOLO(MODEL_PATH)
        class_names = model.names
        print("Modelo carregado com sucesso.")
    except Exception as e:
        print(f"ERRO ao carregar o modelo: {e}")
        return

    # 3. Configurar leitura e escrita do vídeo
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print("ERRO: Não foi possível abrir o vídeo de entrada.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    print(f"Vídeo de saída será salvo em '{OUTPUT_VIDEO_PATH}'")

    # 4. Loop de processamento frame a frame
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processando frame {frame_count}...", end="\r")

        # Roda o rastreamento no frame
        results = model.track(frame, persist=True, verbose=False)

        # Desenha as detecções no frame
        if results[0].obb is not None and results[0].obb.id is not None:
            # Extrai os dados de detecção e rastreamento
            boxes = results[0].obb.xyxyxyxy.cpu().numpy()
            track_ids = results[0].obb.id.int().cpu().tolist()
            confs = results[0].obb.conf.cpu().numpy()
            cls_ids = results[0].obb.cls.cpu().numpy()

            for i in range(len(track_ids)):
                if confs[i] > CONFIDENCE_THRESHOLD:
                    corners = boxes[i].astype(int)
                    track_id = track_ids[i]
                    class_id = int(cls_ids[i])
                    confidence = confs[i]
                    label = f"ID:{track_id} {class_names[class_id]} {confidence:.2f}"

                    # Desenha o polígono OBB
                    cv2.polylines(
                        frame,
                        [corners],
                        isClosed=True,
                        color=BOX_COLOR,
                        thickness=BOX_THICKNESS,
                    )

                    # Posição do texto
                    text_pos = (corners[0][0], corners[0][1] - 10)
                    cv2.putText(
                        frame,
                        label,
                        text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SCALE,
                        TEXT_COLOR,
                        BOX_THICKNESS,
                    )

        # Escreve o frame processado no vídeo de saída
        out.write(frame)

        # Mostra o frame processado
        cv2.imshow("Video Processing", cv2.resize(frame, (1200, 800)))

        # Adiciona uma verificação de tecla para sair (pressione 'q' para fechar)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 5. Libera os recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n--- Processamento concluído. Vídeo salvo com sucesso! ---")


if __name__ == "__main__":
    run_video_inference()
