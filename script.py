import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
from pathlib import Path
import json
from typing import List, Dict, Tuple
import random


class ConstructionSiteAugmentation:
    """
    Classe para aplicar augmentations específicas para canteiros de obras
    """
    
    def __init__(self, output_size: Tuple[int, int] = (640, 640)):
        """
        Args:
            output_size: Tamanho final das imagens (height, width)
        """
        self.output_size = output_size
        
        # Augmentations leves - para treino geral
        self.light_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.7
            ),
            A.Rotate(limit=10, p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.GaussNoise(p=0.2),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            clip=True,
            min_area=0,
            min_visibility=0
        ))
        
        # Augmentations médias - mais variação
        self.medium_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.8
            ),
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.3),
            A.Rotate(limit=15, p=0.4),
            A.RandomScale(scale_limit=0.2, p=0.3),
            A.GaussNoise(p=0.3),
            A.RandomFog(p=0.1),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            clip=True,
            min_area=0,
            min_visibility=0
        ))
        
        # Augmentations pesadas - condições extremas
        self.heavy_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.4,
                contrast_limit=0.4,
                p=0.9
            ),
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=7, p=1.0),
            ], p=0.4),
            A.Rotate(limit=20, p=0.5),
            A.RandomScale(scale_limit=0.3, p=0.4),
            A.GaussNoise(p=0.4),
            A.RandomFog(p=0.2),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.3
            ),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            clip=True,
            min_area=0,
            min_visibility=0
        ))
        
        # Augmentations específicas para canteiro de obras
        self.construction_specific = A.Compose([
            A.HorizontalFlip(p=0.5),
            # Simula diferentes condições de iluminação do dia
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.8
            ),
            # Simula poeira/névoa comum em canteiros
            A.RandomFog(p=0.3),
            # Simula vibração de câmera
            A.MotionBlur(blur_limit=5, p=0.2),
            # Diferentes ângulos de visão
            A.Rotate(limit=15, p=0.4),
            A.Perspective(scale=(0.05, 0.1), p=0.2),
            # Variações de cor para diferentes condições de luz
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
            # Adiciona blur ocasional
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            clip=True,
            min_area=0,
            min_visibility=0
        ))
    
    def get_transform(self, augmentation_level: str = 'medium'):
        """
        Retorna o transform apropriado baseado no nível
        
        Args:
            augmentation_level: 'light', 'medium', 'heavy', ou 'construction'
        """
        transforms = {
            'light': self.light_transform,
            'medium': self.medium_transform,
            'heavy': self.heavy_transform,
            'construction': self.construction_specific
        }
        return transforms.get(augmentation_level, self.medium_transform)


class DatasetAugmenter:
    """
    Classe principal para augmentar dataset no formato YOLO
    """
    
    def __init__(
        self,
        input_images_dir: str,
        input_labels_dir: str,
        output_images_dir: str,
        output_labels_dir: str,
        augmentation_level: str = 'medium'
    ):
        """
        Args:
            input_images_dir: Diretório com imagens originais
            input_labels_dir: Diretório com labels YOLO (.txt)
            output_images_dir: Diretório para salvar imagens augmentadas
            output_labels_dir: Diretório para salvar labels augmentadas
            augmentation_level: Nível de augmentation
        """
        self.input_images_dir = Path(input_images_dir)
        self.input_labels_dir = Path(input_labels_dir)
        self.output_images_dir = Path(output_images_dir)
        self.output_labels_dir = Path(output_labels_dir)
        
        # Criar diretórios de output
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        self.augmenter = ConstructionSiteAugmentation()
        self.transform = self.augmenter.get_transform(augmentation_level)
        
    def validate_bbox(self, bbox: List[float]) -> List[float]:
        """
        Valida e corrige bounding box para estar no intervalo [0, 1]

        Args:
            bbox: [x_center, y_center, width, height]

        Returns:
            bbox corrigida
        """
        # Clonar bbox para nao modificar original
        bbox = bbox.copy()

        # Garantir que todos os valores estejam no intervalo [0, 1]
        for i in range(4):
            bbox[i] = max(0.0, min(1.0, bbox[i]))

        return bbox

    def read_yolo_labels(self, label_path: Path) -> Tuple[List[int], List[List[float]]]:
        """
        Lê arquivo de labels YOLO

        Returns:
            class_labels: Lista de IDs das classes
            bboxes: Lista de bounding boxes no formato [x_center, y_center, width, height]
        """
        class_labels = []
        bboxes = []

        if not label_path.exists():
            return class_labels, bboxes

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]
                    # Validar bbox
                    bbox = self.validate_bbox(bbox)
                    class_labels.append(class_id)
                    bboxes.append(bbox)

        return class_labels, bboxes
    
    def write_yolo_labels(
        self,
        label_path: Path,
        class_labels: List[int],
        bboxes: List[List[float]]
    ):
        """
        Escreve arquivo de labels YOLO
        """
        with open(label_path, 'w') as f:
            for class_id, bbox in zip(class_labels, bboxes):
                class_id = int(class_id)
                line = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                f.write(line)
    
    def augment_single_image(
        self,
        image_path: Path,
        label_path: Path,
        num_augmentations: int = 3,
        save_original: bool = True
    ):
        """
        Aplica augmentations em uma única imagem
        
        Args:
            image_path: Caminho da imagem
            label_path: Caminho do label
            num_augmentations: Número de versões augmentadas a gerar
            save_original: Se deve salvar a imagem original
        """
        # Ler imagem
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Erro ao ler imagem: {image_path}")
            return
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ler labels
        class_labels, bboxes = self.read_yolo_labels(label_path)
        
        # Salvar original se solicitado
        if save_original:
            output_image_path = self.output_images_dir / image_path.name
            output_label_path = self.output_labels_dir / label_path.name
            
            cv2.imwrite(
                str(output_image_path),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )
            if bboxes:
                self.write_yolo_labels(output_label_path, class_labels, bboxes)
        
        # Gerar augmentations
        base_name = image_path.stem
        extension = image_path.suffix

        for i in range(num_augmentations):
            try:
                # Validar bboxes antes de aplicar transform
                validated_bboxes = [self.validate_bbox(list(bbox)) for bbox in bboxes] if bboxes else []

                # Aplicar transform
                if validated_bboxes:
                    transformed = self.transform(
                        image=image,
                        bboxes=validated_bboxes,
                        class_labels=class_labels
                    )
                else:
                    transformed = self.transform(image=image, bboxes=[], class_labels=[])

                aug_image = transformed['image']
                aug_bboxes = transformed['bboxes']
                aug_labels = transformed['class_labels']

                # Validar bboxes transformadas
                aug_bboxes = [self.validate_bbox(list(bbox)) for bbox in aug_bboxes]
                
                # Salvar imagem augmentada
                aug_image_name = f"{base_name}_aug{i+1}{extension}"
                aug_label_name = f"{base_name}_aug{i+1}.txt"
                
                output_image_path = self.output_images_dir / aug_image_name
                output_label_path = self.output_labels_dir / aug_label_name
                
                cv2.imwrite(
                    str(output_image_path),
                    cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                )
                
                if aug_bboxes:
                    self.write_yolo_labels(output_label_path, aug_labels, aug_bboxes)
                
            except Exception as e:
                print(f"Erro ao processar {image_path.name} (aug {i+1}): {str(e)}")
                continue
    
    def augment_dataset(
        self,
        num_augmentations: int = 3,
        save_original: bool = True,
        image_extensions: List[str] = ['.jpg', '.jpeg', '.png']
    ):
        """
        Augmenta todo o dataset
        
        Args:
            num_augmentations: Número de versões augmentadas por imagem
            save_original: Se deve copiar imagens originais
            image_extensions: Extensões de imagem válidas
        """
        # Listar todas as imagens
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.input_images_dir.glob(f"*{ext}"))
            image_files.extend(self.input_images_dir.glob(f"*{ext.upper()}"))

        # Remover duplicatas (Windows nao e case-sensitive)
        image_files = list(set(image_files))

        print(f"Encontradas {len(image_files)} imagens")
        print(f"Gerando {num_augmentations} augmentations por imagem...")
        
        for idx, image_path in enumerate(image_files, 1):
            # Encontrar label correspondente
            label_path = self.input_labels_dir / f"{image_path.stem}.txt"
            
            print(f"Processando [{idx}/{len(image_files)}]: {image_path.name}")
            
            self.augment_single_image(
                image_path,
                label_path,
                num_augmentations,
                save_original
            )
        
        print("\n[OK] Augmentation concluida!")
        print(f"Imagens salvas em: {self.output_images_dir}")
        print(f"Labels salvos em: {self.output_labels_dir}")


def main():
    """
    Exemplo de uso
    """
    # Configurar caminhos
    input_images = "C:/Users/samir/OneDrive/Área de Trabalho/augmentation/train/images"
    input_labels = "C:/Users/samir/OneDrive/Área de Trabalho/augmentation/train/labels"
    output_images = "/dataset_augmented/images"
    output_labels = "/dataset_augmented/labels"
    
    # Criar augmenter
    augmenter = DatasetAugmenter(
        input_images_dir=input_images,
        input_labels_dir=input_labels,
        output_images_dir=output_images,
        output_labels_dir=output_labels,
        augmentation_level='construction'  # Opções: light, medium, heavy, construction
    )
    
    # Augmentar dataset
    augmenter.augment_dataset(
        num_augmentations=3,  # 3 versões augmentadas por imagem
        save_original=True     # Copiar imagens originais também
    )


if __name__ == "__main__":
    main()