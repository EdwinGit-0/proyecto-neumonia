import json
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ===== 1. VALIDACIÓN - experimento combinado-arquitecturas =====
print('='*80)
print('1. VALIDACION — experimento combinado-arquitecturas')
print('='*80)

for arch in ['vgg16', 'resnet50', 'mobilenetv2']:
    with open(f'results/final/combinado_{arch}_validacion.json') as f:
        data = json.load(f)
    
    y_true = np.load(f'results/final/validacion_{arch}_y_true.npy')
    y_prob = np.load(f'results/final/validacion_{arch}_y_prob.npy')
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    print('\n--- ' + data['model_name'] + ' ---')
    print('JSON: results/final/combinado_' + arch + '_validacion.json')
    print('  validation_n:', data['validation_n'])
    print('  validation_composicion:', data['validation_composicion'])
    print('  umbral_medicion:', data['umbral_medicion'])
    print('  ROC-AUC (JSON):', data['validation']['roc_auc'])
    print('  BA (JSON):', data['validation']['balanced_accuracy'])
    print('  Matriz (JSON):', data['validation_matriz_confusion'])
    print('  figuras:', data['figuras'])
    print('  finalizado:', data['finalizado'])
    print('npy true: results/final/validacion_' + arch + '_y_true.npy')
    print('npy prob: results/final/validacion_' + arch + '_y_prob.npy')
    print('Recalculado de npy:')
    print('  TN=' + str(cm[0,0]) + ', FP=' + str(cm[0,1]) + ', FN=' + str(cm[1,0]) + ', TP=' + str(cm[1,1]) + ', Total=' + str(cm.sum()))
    print('  ROC-AUC (recalc):', roc_auc)
    print('  ROC-AUC redondeado (grafica):', round(roc_auc, 4))
    print('  Archivo matriz:', data['figuras']['matriz_confusion'])
    print('  Archivo ROC:', data['figuras']['curva_roc'])

# ===== 2. TEST FINAL =====
print('\n' + '='*80)
print('2. TEST FINAL — modelo definitivo MobileNetV2 + COMBINADO + train+val')
print('='*80)

with open('results/final/final_test_report.json') as f:
    test_data = json.load(f)

y_true = np.load('results/final/test_y_true.npy')
y_prob = np.load('results/final/test_y_prob.npy')
y_pred = (y_prob >= 0.38).astype(int)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

print('JSON: results/final/final_test_report.json')
print('  test total:', test_data['test']['total'])
print('  test NORMAL:', test_data['test']['NORMAL'])
print('  test PNEUMONIA:', test_data['test']['PNEUMONIA'])
print('  umbral:', test_data['umbral'])
print('  umbral_referencia:', test_data['umbral_referencia'])
print('  ROC-AUC (JSON):', test_data['metricas_umbral_congelado']['roc_auc'])
print('  BA (JSON):', test_data['metricas_umbral_congelado']['balanced_accuracy'])
print('  Accuracy (JSON):', test_data['metricas_umbral_congelado']['accuracy'])
print('  Precision (JSON):', test_data['metricas_umbral_congelado']['precision'])
print('  Recall (JSON):', test_data['metricas_umbral_congelado']['recall'])
print('  Specificity (JSON):', test_data['metricas_umbral_congelado']['specificity'])
print('  F1 (JSON):', test_data['metricas_umbral_congelado']['f1'])
print('  Matriz (JSON):', test_data['matriz_confusion'])
print('  figuras:', test_data['figuras'])
print('npy true: results/final/test_y_true.npy')
print('npy prob: results/final/test_y_prob.npy')
y_true = np.load('results/final/test_y_true.npy')
y_prob = np.load('results/final/test_y_prob.npy')
y_pred = (y_prob >= 0.38).astype(int)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
print('Recalculado de npy (umbral 0.38):')
print('  TN=' + str(cm[0,0]) + ', FP=' + str(cm[0,1]) + ', FN=' + str(cm[1,0]) + ', TP=' + str(cm[1,1]) + ', Total=' + str(cm.sum()))
fpr, tpr, _ = roc_curve(np.load('results/final/test_y_true.npy'), np.load('results/final/test_y_prob.npy'))
roc_auc = auc(fpr, tpr)
print('  ROC-AUC (recalc):', roc_auc)
print('  ROC-AUC redondeado (grafica):', round(roc_auc, 4))
print('  Archivo matriz:', test_data['figuras']['matriz_confusion'])
print('  Archivo ROC:', test_data['figuras']['curva_roc'])

# Modelo definitivo
print('\n--- Modelo definitivo ---')
with open('results/final/final_model_training.json') as f:
    model_data = json.load(f)
print('  model_name:', model_data['model_name'])
print('  splits_entrenamiento:', model_data['splits_entrenamiento'])
print('  composicion_entrenamiento:', model_data['composicion_entrenamiento'])
print('  epochs:', model_data['epochs'])
print('  umbral:', model_data['umbral'])