print("DEBUG: Script iniciado. Importando bibliotecas...")
import json
import os
import sys
import traceback

import cv2
import numpy as np
from ultralytics import YOLO

print("DEBUG: Bibliotecas importadas com sucesso.")

# --- Configurações Globais ---
IMAGE_DIR = "images"
LABEL_DIR = "labels"
LABELED_DIR = "labeled"
CONFIG_FILE = "annotator_config.json"
WINDOW_NAME = "YOLO OBB Annotator"
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720
HANDLE_SIZE = 5

# Cores (B, G, R)
COLOR_BOX = (255, 200, 0)
COLOR_SELECTED_BOX = (0, 255, 255)
COLOR_TEXT = (0, 255, 0)  # Verde para o texto principal
COLOR_ADD_MODE = (0, 165, 255)
COLOR_HANDLE = (0, 0, 255)  # Vermelho para as alças

# Carregar o modelo YOLOv8 OBB pré-treinado
print("DEBUG: Carregando modelo YOLO...")
try:
    MODEL = YOLO("YOLOv11n_OBB_Training/treino_run_1/weights/best.pt")
    print("DEBUG: Modelo YOLO carregado com sucesso.")
except Exception as e:
    print(f"ERRO CRÍTICO AO CARREGAR O MODELO: {e}")
    with open("annotator_crash.log", "w") as f:
        f.write("Erro ao carregar o modelo YOLO:\n")
        f.write(traceback.format_exc())
    sys.exit(1)


class AppState:
    """Classe para gerenciar o estado da aplicação."""

    def __init__(self):
        self.image_paths = []
        self.current_image_index = 0
        self.current_image = None
        self.display_image = None
        self.boxes = []
        self.last_manual_boxes = []  # Para a função de copiar
        self.selected_box_index = -1
        self.is_dragging = False
        self.is_creating_box = False
        self.is_resizing = False
        self.resize_corner_index = -1
        self.creation_start_pos = (0, 0)
        self.drag_start_pos = (0, 0)
        self.scale = 1.0
        self.add_class_id = None

    def load_images(self):
        """Carrega a lista de imagens dos diretórios de imagens e rotulados."""
        print("DEBUG: Carregando lista de imagens...")
        valid_extensions = [".jpg", ".jpeg", ".png"]
        found_images = []
        for directory in [IMAGE_DIR, LABELED_DIR]:
            if not os.path.exists(directory):
                print(f"DEBUG: Diretório '{directory}' não encontrado. Criando...")
                os.makedirs(directory)
            for f in os.listdir(directory):
                if os.path.splitext(f)[1].lower() in valid_extensions:
                    found_images.append(os.path.join(directory, f))

        self.image_paths = sorted(found_images)

        if not self.image_paths:
            print(
                f"DEBUG: Nenhuma imagem encontrada em '{IMAGE_DIR}' ou '{LABELED_DIR}'."
            )
        else:
            print(f"DEBUG: {len(self.image_paths)} imagens encontradas.")

    def get_current_image_path(self):
        """Retorna o caminho da imagem atual."""
        return self.image_paths[self.current_image_index]


# --- Funções Auxiliares de Geometria ---


def rotate_points(points, angle, center):
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    rotated_points_homogeneous = (M @ points_homogeneous.T).T
    return rotated_points_homogeneous


def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon.astype(np.int32), point, False) >= 0


def get_handle_at_pos(corners, pos, scale):
    for i, corner in enumerate(corners):
        handle_pos_scaled = corner * scale
        if np.linalg.norm(np.array(pos) - handle_pos_scaled) < HANDLE_SIZE * 2:
            return i
    return -1


# --- Funções de Callback e Desenho ---


def mouse_callback(event, x, y, flags, param):
    state = param
    current_pos_scaled = (x, y)
    current_pos = (x / state.scale, y / state.scale)

    if event == cv2.EVENT_LBUTTONDOWN:
        state.is_dragging = False
        state.is_resizing = False
        state.is_creating_box = False

        if state.add_class_id is not None:
            state.is_creating_box = True
            state.creation_start_pos = current_pos
            new_box = {
                "class_id": state.add_class_id,
                "corners": np.array([current_pos] * 4),
                "is_manual": True,
            }
            state.boxes.append(new_box)
            state.selected_box_index = len(state.boxes) - 1
            state.add_class_id = None
            return

        if state.selected_box_index != -1:
            selected_box = state.boxes[state.selected_box_index]
            handle_index = get_handle_at_pos(
                selected_box["corners"], current_pos_scaled, state.scale
            )
            if handle_index != -1:
                state.is_resizing = True
                state.resize_corner_index = handle_index
                state.original_corners_on_drag = selected_box["corners"].copy()
                selected_box["is_manual"] = True
                return

        for i in range(len(state.boxes) - 1, -1, -1):
            if is_point_in_polygon(current_pos, state.boxes[i]["corners"]):
                state.selected_box_index = i
                state.is_dragging = True
                state.drag_start_pos = current_pos
                state.original_corners_on_drag = state.boxes[i]["corners"].copy()
                state.boxes[i]["is_manual"] = True
                return

        state.selected_box_index = -1

    elif event == cv2.EVENT_MOUSEMOVE:
        if state.is_resizing:
            dragged_corner_index = state.resize_corner_index
            p_drag_new = np.array(current_pos)
            p_opp_index = (dragged_corner_index + 2) % 4
            p_adj1_index = (dragged_corner_index + 1) % 4
            p_adj2_index = (dragged_corner_index + 3) % 4
            p_opp = state.original_corners_on_drag[p_opp_index]
            p_adj1_orig = state.original_corners_on_drag[p_adj1_index]
            p_adj2_orig = state.original_corners_on_drag[p_adj2_index]
            v1 = p_adj1_orig - p_opp
            v2 = p_adj2_orig - p_opp
            mouse_vec = p_drag_new - p_opp
            proj1 = (
                np.dot(mouse_vec, v1) / (np.dot(v1, v1) if np.dot(v1, v1) != 0 else 1)
            ) * v1
            proj2 = (
                np.dot(mouse_vec, v2) / (np.dot(v2, v2) if np.dot(v2, v2) != 0 else 1)
            ) * v2
            p_adj1_new = p_opp + proj1
            p_adj2_new = p_opp + proj2
            p_drag_new_final = p_adj1_new + p_adj2_new - p_opp
            new_corners = np.zeros((4, 2))
            new_corners[dragged_corner_index] = p_drag_new_final
            new_corners[p_opp_index] = p_opp
            new_corners[p_adj1_index] = p_adj1_new
            new_corners[p_adj2_index] = p_adj2_new
            state.boxes[state.selected_box_index]["corners"] = new_corners
            return

        if state.is_creating_box:
            start_pos = state.creation_start_pos
            end_pos = current_pos
            corners = np.array(
                [
                    [start_pos[0], start_pos[1]],
                    [end_pos[0], start_pos[1]],
                    [end_pos[0], end_pos[1]],
                    [start_pos[0], end_pos[1]],
                ]
            )
            state.boxes[state.selected_box_index]["corners"] = corners
            return

        if state.is_dragging and state.selected_box_index != -1:
            dx = current_pos[0] - state.drag_start_pos[0]
            dy = current_pos[1] - state.drag_start_pos[1]
            state.boxes[state.selected_box_index]["corners"] = (
                state.original_corners_on_drag + np.array([dx, dy])
            )

    elif event == cv2.EVENT_LBUTTONUP:
        state.is_creating_box = False
        state.is_dragging = False
        state.is_resizing = False

    elif event == cv2.EVENT_MOUSEWHEEL:
        if state.selected_box_index != -1 and (flags & cv2.EVENT_FLAG_SHIFTKEY):
            scroll_up = flags > 0
            angle = 2 if scroll_up else -2
            box = state.boxes[state.selected_box_index]
            center = np.mean(box["corners"], axis=0)
            state.boxes[state.selected_box_index]["corners"] = rotate_points(
                box["corners"], angle, tuple(center)
            )
            box["is_manual"] = True


def draw_boxes(image, boxes, selected_index, scale):
    for i, box in enumerate(boxes):
        is_selected = i == selected_index
        color = COLOR_SELECTED_BOX if is_selected else COLOR_BOX
        display_points = (box["corners"] * scale).astype(np.int32)
        cv2.polylines(image, [display_points], isClosed=True, color=color, thickness=2)
        class_id = box["class_id"]
        text_pos = (display_points[0][0], display_points[0][1] - 10)
        cv2.putText(
            image, str(class_id), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
        if is_selected:
            for p in display_points:
                cv2.circle(image, tuple(p), HANDLE_SIZE, COLOR_HANDLE, -1)


def run_yolo_prediction(state):
    print("DEBUG: Executando predição YOLO...")
    results = MODEL.predict(state.current_image, verbose=False)
    state.boxes = []
    if results[0].obb is not None:
        for r in results[0].obb:
            class_id = int(r.cls[0].item())
            corners = r.xyxyxyxy[0].cpu().numpy()
            state.boxes.append(
                {"class_id": class_id, "corners": corners, "is_manual": False}
            )
    print(f"DEBUG: {len(state.boxes)} caixas detectadas.")


# --- Funções de Persistência ---


def get_label_path(image_path):
    filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    return os.path.join(LABEL_DIR, filename)


def save_annotations(state):
    if not state.image_paths:
        return
    print("DEBUG: Salvando anotações...")
    label_path = get_label_path(state.get_current_image_path())
    h, w, _ = state.current_image.shape
    lines = []
    for box in state.boxes:
        class_id = box["class_id"]
        normalized_corners = box["corners"].flatten().copy()
        normalized_corners[0::2] /= w
        normalized_corners[1::2] /= h
        corner_str = " ".join(map(str, normalized_corners))
        lines.append(f"{class_id} {corner_str}")
    with open(label_path, "w") as f:
        f.write("\n".join(lines))
    print(f"DEBUG: Anotações salvas em: {label_path}")


def move_image_if_labeled(state):
    """Move a imagem atual para a pasta de rotulados, se ainda não estiver lá."""
    if not state.image_paths:
        return
    source_path = state.get_current_image_path()
    if os.path.dirname(source_path) == LABELED_DIR:
        return  # Já está na pasta de rotulados

    dest_path = os.path.join(LABELED_DIR, os.path.basename(source_path))
    print(f"DEBUG: Movendo imagem de {source_path} para {dest_path}")
    try:
        os.rename(source_path, dest_path)
        # Atualiza o caminho na lista de imagens do estado
        state.image_paths[state.current_image_index] = dest_path
    except OSError as e:
        print(f"ERRO: Não foi possível mover a imagem: {e}")


def load_annotations(state):
    label_path = get_label_path(state.get_current_image_path())
    if not os.path.exists(label_path):
        return False
    print(f"DEBUG: Carregando anotações de {label_path}...")
    h, w, _ = state.current_image.shape
    state.boxes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            class_id = int(parts[0])
            corners = np.array([float(p) for p in parts[1:]])
            corners[0::2] *= w
            corners[1::2] *= h
            state.boxes.append(
                {
                    "class_id": class_id,
                    "corners": corners.reshape(4, 2),
                    "is_manual": True,
                }
            )
    print(f"DEBUG: {len(state.boxes)} anotações carregadas.")
    return True


def save_config(state):
    if not state.image_paths:
        return
    config_data = {"last_image_index": state.current_image_index}
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_data, f)
    print(f"DEBUG: Configuração salva em {CONFIG_FILE}")


def load_config(state):
    if not os.path.exists(CONFIG_FILE):
        print("DEBUG: Arquivo de configuração não encontrado.")
        return
    try:
        with open(CONFIG_FILE, "r") as f:
            config_data = json.load(f)
        last_index = config_data.get("last_image_index", 0)
        if 0 <= last_index < len(state.image_paths):
            state.current_image_index = last_index
            print(f"DEBUG: Configuração carregada. Iniciando no índice {last_index}")
        else:
            print("DEBUG: Índice salvo é inválido. Iniciando do começo.")
    except (json.JSONDecodeError, KeyError):
        print("DEBUG: Erro ao ler arquivo de configuração. Iniciando do começo.")


def load_and_scale_image(state):
    print("DEBUG: Carregando e redimensionando a imagem...")
    current_path = state.get_current_image_path()
    state.current_image = cv2.imread(current_path)
    if state.current_image is None:
        raise IOError(f"Não foi possível ler a imagem: {current_path}")
    h, w, _ = state.current_image.shape
    state.scale = 1.0
    if h > MAX_DISPLAY_HEIGHT or w > MAX_DISPLAY_WIDTH:
        state.scale = min(MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h)
    new_w = int(w * state.scale)
    new_h = int(h * state.scale)
    state.display_image = cv2.resize(state.current_image, (new_w, new_h))
    print("DEBUG: Imagem carregada e redimensionada com sucesso.")


# --- Loop Principal ---


def main():
    try:
        print("DEBUG: Iniciando a função main().")
        state = AppState()
        print("DEBUG: Criando janela OpenCV...")
        cv2.namedWindow(WINDOW_NAME)
        print("DEBUG: Janela criada.")
        state.load_images()

        if not state.image_paths:
            error_img = np.zeros((200, 800, 3), dtype=np.uint8)
            cv2.putText(
                error_img,
                "Nenhuma imagem encontrada nas pastas 'images' ou 'labeled'.",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                error_img,
                "Adicione arquivos .jpg ou .png e reinicie a aplicacao.",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                error_img,
                "Pressione 'q' para sair.",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            while True:
                cv2.imshow(WINDOW_NAME, error_img)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
            return

        load_config(state)

        print("DEBUG: Configurando callback do mouse...")
        cv2.setMouseCallback(WINDOW_NAME, mouse_callback, state)
        print("DEBUG: Callback do mouse configurado.")
        load_and_scale_image(state)
        if not load_annotations(state):
            run_yolo_prediction(state)

        print("DEBUG: Entrando no loop principal de eventos...")
        while True:
            display_copy = state.display_image.copy()
            draw_boxes(display_copy, state.boxes, state.selected_box_index, state.scale)

            help_text = f"(A)Prev/(D)Next | (W)Copy | (Q)uit | Del(Space)"
            status_text = (
                "0-9 to Add | Select: Drag=Move, Handle=Resize, SHIFT+Scroll=Rotate"
            )

            if state.add_class_id is not None:
                status_text = f"ADD MODE: Class {state.add_class_id}. Click and drag to create. (ESC) to cancel."
                cv2.putText(
                    display_copy,
                    status_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    COLOR_ADD_MODE,
                    2,
                )
            else:
                cv2.putText(
                    display_copy,
                    status_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    COLOR_TEXT,
                    1,
                )

            cv2.putText(
                display_copy,
                help_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_TEXT,
                1,
            )

            cv2.imshow(WINDOW_NAME, display_copy)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                save_annotations(state)
                save_config(state)
                break
            elif key == 27:
                state.add_class_id = None
                if state.is_creating_box:
                    state.is_creating_box = False
                    state.boxes.pop()
                    state.selected_box_index = -1
            elif ord("0") <= key <= ord("9"):
                state.add_class_id = int(chr(key))
                print(
                    f"DEBUG: Modo de adição ativado para a classe: {state.add_class_id}"
                )
            elif key == 32:
                if state.selected_box_index != -1:
                    print(f"DEBUG: Deletando caixa: {state.selected_box_index}")
                    state.boxes.pop(state.selected_box_index)
                    state.selected_box_index = -1
            elif key == ord("w"):
                if state.last_manual_boxes:
                    print(
                        f"DEBUG: Copiando {len(state.last_manual_boxes)} caixas manuais."
                    )
                    state.boxes.extend(state.last_manual_boxes)
                else:
                    print("DEBUG: Nenhuma caixa manual anterior para copiar.")

            image_changed = False
            if key == ord("d"):
                image_changed = True
            elif key == ord("a"):
                image_changed = True

            if image_changed:
                print("DEBUG: Mudando de imagem...")
                save_annotations(state)
                move_image_if_labeled(state)
                state.last_manual_boxes = [
                    box for box in state.boxes if box.get("is_manual", False)
                ]
                print(
                    f"DEBUG: Armazenado {len(state.last_manual_boxes)} caixas manuais."
                )

                if key == ord("d"):
                    state.current_image_index = (state.current_image_index + 1) % len(
                        state.image_paths
                    )
                elif key == ord("a"):
                    state.current_image_index = (
                        state.current_image_index - 1 + len(state.image_paths)
                    ) % len(state.image_paths)

                state.selected_box_index = -1
                state.is_dragging = False
                state.is_resizing = False
                state.add_class_id = None
                load_and_scale_image(state)
                if not load_annotations(state):
                    run_yolo_prediction(state)

    except Exception as e:
        print(f"ERRO CRÍTICO INESPERADO: {e}")
        with open("annotator_crash.log", "w") as f:
            f.write("Um erro crítico e inesperado ocorreu:\n")
            f.write(traceback.format_exc())
        error_img = np.zeros((300, 1000, 3), dtype=np.uint8)
        cv2.putText(
            error_img,
            "ERRO CRITICO. Verifique o arquivo annotator_crash.log",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            error_img,
            str(e),
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.imshow(WINDOW_NAME, error_img)
        cv2.waitKey(0)

    finally:
        print("DEBUG: Encerrando a aplicação e destruindo janelas.")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    if not os.path.exists(LABEL_DIR):
        os.makedirs(LABEL_DIR)
    if not os.path.exists(LABELED_DIR):
        os.makedirs(LABELED_DIR)
    main()
