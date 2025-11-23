# Dataset Augmentation Algorithm

Sistema de augmentação de datasets YOLO para detecção de objetos em canteiros de obras.

## Estrutura do Projeto

```
augmentation-algorithm/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   └── augmenter.py          # Classes principais de augmentação
│   ├── utils/
│   │   ├── __init__.py
│   │   └── label_fixer.py        # Utilitário para correção de labels
│   └── visualization/
│       ├── __init__.py
│       ├── visualize.py          # Visualização com matplotlib
│       ├── visualize_annotations.py  # Visualização com OpenCV
│       └── visualize_web.py      # Geração de HTML
├── examples/
│   ├── augment_dataset.py
│   ├── fix_labels.py
│   ├── visualize_example.py
│   ├── visualize_annotations_example.py
│   └── visualize_web_example.py
├── data/
│   ├── input/                    # Coloque seus datasets aqui
│   └── output/                   # Datasets augmentados
└── README.md
```

## Instalação

```bash
pip install albumentations opencv-python matplotlib numpy
```

## Uso

### 1. Augmentação de Dataset

```python
from src.core.augmenter import DatasetAugmenter

augmenter = DatasetAugmenter(
    input_images_dir="data/input/images",
    input_labels_dir="data/input/labels",
    output_images_dir="data/output/images",
    output_labels_dir="data/output/labels",
    augmentation_level='construction'
)

augmenter.augment_dataset(
    num_augmentations=5,
    save_original=True
)
```

### 2. Correção de Labels

```python
from src.utils.label_fixer import fix_all_labels

fix_all_labels("data/input/labels", backup=True)
```

### 3. Visualização

```python
from src.visualization.visualize import visualize_augmentations

visualize_augmentations(
    image_path="path/to/image.jpg",
    label_path="path/to/label.txt",
    num_samples=6,
    augmentation_level='construction',
    save_path='output.png'
)
```

## Níveis de Augmentação

- **light**: Augmentações leves para treino geral
- **medium**: Augmentações médias com mais variação
- **heavy**: Augmentações pesadas para condições extremas
- **construction**: Augmentações específicas para canteiros de obras

## Exemplos

Execute os scripts em `examples/` para ver o sistema em ação:

```bash
python examples/augment_dataset.py
python examples/visualize_example.py
python examples/fix_labels.py
```

## Características

- Suporte completo para formato YOLO
- Validação automática de bounding boxes
- Augmentações específicas para canteiros de obras
- Visualizações interativas e HTML
- Correção automática de labels inválidos