import cv2
import os
import random
import base64
from pathlib import Path

def draw_yolo_boxes(image_path, label_path, class_names=None):
    """
    Desenha as bounding boxes do formato YOLO na imagem
    """
    # Ler a imagem
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar imagem: {image_path}")
        return None

    height, width = image.shape[:2]

    # Cores para diferentes classes (BGR)
    colors = [
        (0, 255, 0),    # Verde
        (255, 0, 0),    # Azul
        (0, 0, 255),    # Vermelho
        (255, 255, 0),  # Ciano
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Amarelo
    ]

    # Ler o arquivo de labels
    if not os.path.exists(label_path):
        print(f"Label não encontrado: {label_path}")
        return image

    num_boxes = 0
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Desenhar cada bounding box
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(float(parts[0]))
        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        # Converter de YOLO para coordenadas de pixel
        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)

        # Escolher cor baseada na classe
        color = colors[class_id % len(colors)]

        # Desenhar retângulo
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

        # Adicionar label da classe
        if class_names and class_id < len(class_names):
            label = class_names[class_id]
        else:
            label = f"Class {class_id}"

        # Fundo para o texto
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
        cv2.putText(image, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        num_boxes += 1

    return image, num_boxes

def image_to_base64(image):
    """Converte imagem para base64"""
    # Redimensionar se muito grande
    max_width = 800
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_h = int(h * scale)
        image = cv2.resize(image, (max_width, new_h))

    _, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode()
    return img_str

def generate_html_visualization(images_dir, labels_dir, output_html, num_samples=20, class_names=None):
    """
    Gera um arquivo HTML com visualizações das anotações
    """
    # Listar todas as imagens
    image_files = list(Path(images_dir).glob("*.jpg"))

    if len(image_files) == 0:
        print("Nenhuma imagem encontrada!")
        return

    print(f"Total de imagens: {len(image_files)}")

    # Selecionar amostras: originais e augmentadas
    originals = [f for f in image_files if "_aug" not in f.name]
    augmented = [f for f in image_files if "_aug" in f.name]

    print(f"Originais: {len(originals)}, Augmentadas: {len(augmented)}")

    # Selecionar amostras aleatórias (mix de originais e augmentadas)
    num_originals = min(10, len(originals))
    num_aug = min(num_samples - num_originals, len(augmented))

    samples = random.sample(originals, num_originals) + random.sample(augmented, num_aug)
    random.shuffle(samples)

    # Iniciar HTML
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Visualização de Anotações - Dataset Augmentado</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                text-align: center;
                background-color: #4CAF50;
                color: white;
                padding: 20px;
                border-radius: 10px;
            }
            .stats {
                background-color: white;
                padding: 20px;
                margin: 20px 0;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .image-container {
                background-color: white;
                margin: 20px 0;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .image-title {
                font-weight: bold;
                margin-bottom: 10px;
                color: #333;
                font-size: 16px;
            }
            .image-info {
                color: #666;
                font-size: 14px;
                margin-bottom: 10px;
            }
            img {
                max-width: 100%;
                border: 2px solid #ddd;
                border-radius: 5px;
            }
            .augmented {
                border-left: 5px solid #FF9800;
            }
            .original {
                border-left: 5px solid #4CAF50;
            }
            .legend {
                background-color: white;
                padding: 15px;
                margin: 20px 0;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .legend-item {
                display: inline-block;
                margin-right: 20px;
                padding: 5px 10px;
                border-radius: 5px;
            }
            .legend-original {
                background-color: #4CAF50;
                color: white;
            }
            .legend-augmented {
                background-color: #FF9800;
                color: white;
            }
        </style>
    </head>
    <body>
        <h1>🔍 Visualização de Anotações - Dataset Augmentado</h1>

        <div class="stats">
            <h2>Estatísticas do Dataset</h2>
            <p><strong>Total de imagens:</strong> """ + str(len(image_files)) + """</p>
            <p><strong>Imagens originais:</strong> """ + str(len(originals)) + """</p>
            <p><strong>Imagens augmentadas:</strong> """ + str(len(augmented)) + """</p>
            <p><strong>Amostras visualizadas:</strong> """ + str(len(samples)) + """</p>
        </div>

        <div class="legend">
            <span class="legend-item legend-original">Imagem Original</span>
            <span class="legend-item legend-augmented">Imagem Augmentada</span>
        </div>
    """

    # Processar cada amostra
    for i, image_path in enumerate(samples):
        print(f"Processando [{i+1}/{len(samples)}]: {image_path.name}")

        image_name = image_path.name
        label_path = Path(labels_dir) / image_name.replace('.jpg', '.txt')

        # Desenhar boxes
        image_with_boxes, num_boxes = draw_yolo_boxes(str(image_path), str(label_path), class_names)

        if image_with_boxes is not None:
            # Converter para base64
            img_base64 = image_to_base64(image_with_boxes)

            # Determinar se é original ou augmentada
            is_augmented = "_aug" in image_name
            css_class = "augmented" if is_augmented else "original"
            type_label = "🔄 Augmentada" if is_augmented else "📷 Original"

            # Adicionar ao HTML
            html_content += f"""
            <div class="image-container {css_class}">
                <div class="image-title">{type_label}: {image_name}</div>
                <div class="image-info">Bounding boxes detectadas: {num_boxes}</div>
                <img src="data:image/jpeg;base64,{img_base64}" alt="{image_name}">
            </div>
            """

    # Finalizar HTML
    html_content += """
    </body>
    </html>
    """

    # Salvar HTML
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n✓ HTML gerado com sucesso: {output_html}")
    print(f"Abra o arquivo no navegador para visualizar as anotações!")

if __name__ == "__main__":
    # Configuração
    DATASET_DIR = "C:/Users/samir/git/augmentation/dataset_augmented"
    IMAGES_DIR = os.path.join(DATASET_DIR, "images")
    LABELS_DIR = os.path.join(DATASET_DIR, "labels")
    OUTPUT_HTML = "C:/Users/samir/git/augmentation/visualizacao_anotacoes.html"

    # Nomes das classes
    CLASS_NAMES = ["worker"]

    print("=" * 70)
    print("GERANDO VISUALIZAÇÃO HTML DAS ANOTAÇÕES")
    print("=" * 70)

    generate_html_visualization(IMAGES_DIR, LABELS_DIR, OUTPUT_HTML, num_samples=30, class_names=CLASS_NAMES)

    print("\n" + "=" * 70)
    print("CONCLUÍDO!")
    print("=" * 70)
    print(f"\nAbra o arquivo no navegador:")
    print(f"file:///{OUTPUT_HTML}")
