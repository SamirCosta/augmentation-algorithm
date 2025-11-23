"""
Script para corrigir coordenadas de bounding boxes que estao ligeiramente fora do range [0.0, 1.0]
devido a erros de precisao numerica.
"""
from pathlib import Path
import shutil

def fix_bbox_coordinates(bbox):
    """
    Corrige coordenadas que estao ligeiramente fora do range [0.0, 1.0]

    Args:
        bbox: Lista [x_center, y_center, width, height]

    Returns:
        Lista com coordenadas corrigidas
    """
    x_center, y_center, width, height = bbox

    # Calcular coordenadas min/max
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    # Clipar para o range valido [0.0, 1.0]
    x_min = max(0.0, min(1.0, x_min))
    y_min = max(0.0, min(1.0, y_min))
    x_max = max(0.0, min(1.0, x_max))
    y_max = max(0.0, min(1.0, y_max))

    # Recalcular centro e dimensoes
    width_fixed = x_max - x_min
    height_fixed = y_max - y_min
    x_center_fixed = x_min + width_fixed / 2
    y_center_fixed = y_min + height_fixed / 2

    return [x_center_fixed, y_center_fixed, width_fixed, height_fixed]


def fix_label_file(label_path, backup=True):
    """
    Corrige um arquivo de label YOLO

    Args:
        label_path: Caminho do arquivo de label
        backup: Se True, cria backup do arquivo original

    Returns:
        True se o arquivo foi modificado, False caso contrario
    """
    if not Path(label_path).exists():
        return False

    # Ler labels
    lines = []
    modified = False

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                bbox = [float(x) for x in parts[1:5]]

                # Verificar se precisa corrigir
                x_center, y_center, width, height = bbox
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2

                needs_fix = (x_min < 0 or y_min < 0 or x_max > 1.0 or y_max > 1.0)

                if needs_fix:
                    bbox_fixed = fix_bbox_coordinates(bbox)
                    line_fixed = f"{class_id} {bbox_fixed[0]:.6f} {bbox_fixed[1]:.6f} {bbox_fixed[2]:.6f} {bbox_fixed[3]:.6f}\n"
                    lines.append(line_fixed)
                    modified = True
                else:
                    lines.append(line)
            else:
                lines.append(line)

    if modified:
        # Criar backup se solicitado
        if backup:
            backup_path = Path(str(label_path) + '.backup')
            shutil.copy2(label_path, backup_path)

        # Escrever arquivo corrigido
        with open(label_path, 'w') as f:
            f.writelines(lines)

    return modified


def fix_all_labels(labels_dir, backup=True):
    """
    Corrige todos os arquivos de label em um diretorio

    Args:
        labels_dir: Diretorio com os arquivos .txt de labels
        backup: Se True, cria backup dos arquivos originais
    """
    labels_path = Path(labels_dir)
    label_files = list(labels_path.glob("*.txt"))

    print("=" * 70)
    print("CORRECAO DE LABELS YOLO")
    print("=" * 70)
    print(f"\nDiretorio: {labels_dir}")
    print(f"Total de labels encontrados: {len(label_files)}")
    print(f"Criar backups: {'Sim' if backup else 'Nao'}")
    print("\nProcessando...")
    print("-" * 70)

    fixed_count = 0

    for label_file in label_files:
        was_modified = fix_label_file(label_file, backup=backup)
        if was_modified:
            fixed_count += 1
            print(f"[CORRIGIDO] {label_file.name}")

    print("-" * 70)
    print(f"\nResultado:")
    print(f"  Total de arquivos: {len(label_files)}")
    print(f"  Arquivos corrigidos: {fixed_count}")
    print(f"  Arquivos OK: {len(label_files) - fixed_count}")

    if backup and fixed_count > 0:
        print(f"\n[INFO] Backups salvos com extensao .backup")

    print("\n" + "=" * 70)
    print("[OK] Processo concluido!")
    print("=" * 70)


if __name__ == "__main__":
    # Diretorio com os labels
    LABELS_DIR = r"C:/Users/samir/git/augmentation/reais/train/labels"

    # Corrigir todos os labels (cria backups automaticamente)
    fix_all_labels(LABELS_DIR, backup=True)
