import json
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
FIGURES_DIR = Path('reports/figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CLASES = ('NORMAL', 'PNEUMONIA')

def guardar_matriz_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    titulo: str,
    nombre_archivo: str,
) -> Path:
    directorio = FIGURES_DIR
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm = cm.astype(float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    imagen = ax.imshow(cm_norm, cmap='Blues')
    ax.set_title(titulo)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASES)
    ax.set_yticklabels(CLASES)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', color='black')
    fig.colorbar(imagen, ax=ax)
    plt.tight_layout()
    ruta_salida = directorio / nombre_archivo
    plt.savefig(ruta_salida, dpi=200)
    plt.close(fig)
    return ruta_salida

def guardar_curva_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    titulo: str,
    nombre_archivo: str,
    etiqueta_adicional: str = None,
) -> Path:
    directorio = FIGURES_DIR
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = float(np.trapz(tpr, fpr))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=etiqueta_adicional or 'modelo')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='azar')
    ax.set_title(f'{titulo} (ROC-AUC = {roc_auc:.4f})')
    ax.set_xlabel('Tasa de falsos positivos')
    ax.set_ylabel('Tasa de verdaderos positivos')
    ax.legend()
    plt.tight_layout()
    ruta_salida = directorio / nombre_archivo
    plt.savefig(ruta_salida, dpi=200)
    plt.close(fig)
    return ruta_salida

# ============================================================
# 1. COMPARACIÓN DE ARQUITECTURAS - VALIDACIÓN - UMBRAL 0.5
# ============================================================
print('='*60)
print('1. COMPARACIÓN ARQUITECTURAS - VALIDACIÓN - UMBRAL 0.5')
print('='*60)

for arch_name, arch_label in [('vgg16', 'VGG16'), ('resnet50', 'ResNet50'), ('mobilenetv2', 'MobileNetV2')]:
    print(f'\n--- {arch_label} ---')
    y_true = np.load(f'results/final/validacion_{arch_name}_y_true.npy')
    y_prob = np.load(f'results/final/validacion_{arch_name}_y_prob.npy')
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Matriz de confusión
    guardar_matriz_confusion(
        y_true, y_pred,
        f'Matriz de confusión - {arch_label} sobre validación',
        f'validation_combinado_{arch_name}_confusion_matrix.png'
    )
    
    # Curva ROC
    guardar_curva_roc(
        y_true, y_prob,
        f'Curva ROC - {arch_label} sobre validación',
        f'validation_combinado_{arch_name}_roc_curve.png',
        etiqueta_adicional=f'{arch_label} COMBINADO'
    )
    print(f'{arch_label}: matriz y ROC actualizados')

# También actualizar los genéricos (que usan MobileNetV2)
y_true_mn = np.load('results/final/validacion_mobilenetv2_y_true.npy')
y_prob_mn = np.load('results/final/validacion_mobilenetv2_y_prob.npy')
y_pred_mn = (y_prob_mn >= 0.5).astype(int)

guardar_matriz_confusion(
    y_true_mn, y_pred_mn,
    'Matriz de confusión - MobileNetV2 sobre validación',
    'validation_combinado_confusion_matrix.png'
)

guardar_curva_roc(
    y_true_mn, y_prob_mn,
    'Curva ROC - MobileNetV2 sobre validación',
    'validation_combinado_roc_curve.png',
    etiqueta_adicional='MobileNetV2 COMBINADO'
)
print('Genéricos (MobileNetV2): matriz y ROC actualizados')

# ============================================================
# 2. AJUSTE UMBRAL - VALIDACIÓN - UMBRAL 0.38 (MobileNetV2)
# ============================================================
print('\n' + '='*60)
print('2. AJUSTE UMBRAL - VALIDACIÓN - UMBRAL 0.38 (MobileNetV2)')
print('='*60)

y_true_val = np.load('results/final/validacion_y_true.npy')  # copia de mobilenetv2
y_prob_val = np.load('results/final/validacion_y_prob.npy')
y_pred_038 = (y_prob_val >= 0.38).astype(int)

# Matriz de confusión umbral 0.38
guardar_matriz_confusion(
    y_true_val, y_pred_038,
    'Matriz de confusión - MobileNetV2 sobre validación',
    'validation_combinado_mobilenetv2_confusion_matrix_0.38.png'
)
print('Matriz umbral 0.38 guardada como validation_combinado_mobilenetv2_confusion_matrix_0.38.png')

# También actualizamos la genérica para que coincida (ya es 0.5 en el nombre)
# La figura threshold_selection_validation.png ya existe y es correcta

# ============================================================
# 3. TEST FINAL - UMBRAL 0.38
# ============================================================
print('\n' + '='*60)
print('3. TEST FINAL - UMBRAL 0.38')
print('='*60)

y_true_test = np.load('results/final/test_y_true.npy')
y_prob_test = np.load('results/final/test_y_prob.npy')
y_pred_test = (y_prob_test >= 0.38).astype(int)

# Matriz de confusión test
guardar_matriz_confusion(
    y_true_test, y_pred_test,
    'Matriz de confusión - MobileNetV2 sobre test',
    'test_final_confusion_matrix.png'
)

# Curva ROC test
guardar_curva_roc(
    y_true_test, y_prob_test,
    'Curva ROC - MobileNetV2 sobre test',
    'test_final_roc_curve.png',
    etiqueta_adicional='MobileNetV2 COMBINADO'
)
print('Test final: matriz y ROC actualizados')

print('\n' + '='*60)
print('TODAS LAS FIGURAS ACTUALIZADAS')
print('='*60)