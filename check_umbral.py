import json
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ===== MobileNetV2 VALIDACIÓN con umbral 0.38 =====
print('='*80)
print('MobileNetV2 - VALIDACION con umbral 0.38')
print('='*80)

# Load probabilities used for umbral selection
y_true = np.load('results/final/validacion_y_true.npy')
y_prob = np.load('results/final/validacion_y_prob.npy')

print('Archivos y_true/y_prob usados para umbral:')
print('  results/final/validacion_y_true.npy')
print('  results/final/validacion_y_prob.npy')
print('(Estos son copias de validacion_mobilenetv2_*.npy tras la seleccion)')

# Apply threshold 0.38
umbral = 0.38
y_pred = (y_prob >= umbral).astype(int)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

tn = cm[0,0]
fp = cm[0,1]
fn = cm[1,0]
tp = cm[1,1]
total = tn + fp + fn + tp

accuracy = (tn + tp) / total
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
ba = (recall + specificity) / 2

fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

print()
print('Matriz de confusion (umbral 0.38):')
print('  TN = ' + str(tn))
print('  FP = ' + str(fp))
print('  FN = ' + str(fn))
print('  TP = ' + str(tp))
print('  Total = ' + str(total))

print()
print('Metricas:')
print('  Balanced Accuracy = ' + str(ba))
print('  Accuracy = ' + str(accuracy))
print('  Precision = ' + str(precision))
print('  Recall = ' + str(recall))
print('  Specificity = ' + str(specificity))
print('  F1-Score = ' + str(f1))
print('  ROC-AUC = ' + str(roc_auc))
print('  ROC-AUC exacto = ' + str(roc_auc))

# Load umbral decision JSON
with open('results/final/umbral_decision.json') as f:
    umbral_data = json.load(f)
print()
print('Archivo JSON de decision de umbral: results/final/umbral_decision.json')
print('  Umbral congelado: ' + str(umbral_data['umbral']))
print('  Criterio: ' + str(umbral_data['criterio']))
print('  BA validacion (umbral 0.38): ' + str(umbral_data['balanced_accuracy_validacion']))
print('  BA validacion (umbral 0.5): ' + str(umbral_data['balanced_accuracy_validacion_umbral_0.5']))
print('  Delta: ' + str(umbral_data['delta_balanced_accuracy_vs_0.5']))
print('  Congelado: ' + str(umbral_data['congelado']))
print('  Conjunto origen: ' + str(umbral_data['conjunto_origen']))
print('  validation_n: ' + str(umbral_data['validation_n']))

# Check the figuras from umbral_decision
print()
print('Archivo de figura de seleccion de umbral:', umbral_data['figura_seleccion_umbral'])

# Validation composition
print()
print('Validation: ' + str(total) + ' imagenes (confirmado)')
print('  NORMAL: ' + str((y_true==0).sum()))
print('  PNEUMONIA: ' + str((y_true==1).sum()))

# ===== TEST FINAL =====
print()
print('='*80)
print('MODELO DEFINITIVO - TEST FINAL con umbral 0.38')
print('='*80)

with open('results/final/final_test_report.json') as f:
    test_data = json.load(f)

y_true_test = np.load('results/final/test_y_true.npy')
y_prob_test = np.load('results/final/test_y_prob.npy')
y_pred_test = (y_prob_test >= 0.38).astype(int)
cm_test = confusion_matrix(y_true_test, y_pred_test, labels=[0, 1])

tn_test = cm_test[0,0]
fp_test = cm_test[0,1]
fn_test = cm_test[1,0]
tp_test = cm_test[1,1]
total_test = tn_test + fp_test + fn_test + tp_test

accuracy_test = (tn_test + tp_test) / total_test
recall_test = tp_test / (tp_test + fn_test) if (tp_test + fn_test) > 0 else 0
specificity_test = tn_test / (tn_test + fp_test) if (tn_test + fp_test) > 0 else 0
precision_test = tp_test / (tp_test + fp_test) if (tp_test + fp_test) > 0 else 0
f1_test = 2 * precision_test * recall_test / (precision_test + recall_test) if (precision_test + recall_test) > 0 else 0
ba_test = (recall_test + specificity_test) / 2

fpr_test, tpr_test, _ = roc_curve(np.load('results/final/test_y_true.npy'), np.load('results/final/test_y_prob.npy'))
roc_auc_test = auc(fpr_test, tpr_test)

print()
print('Matriz de confusion (umbral 0.38):')
print('  TN = ' + str(cm_test[0,0]))
print('  FP = ' + str(cm_test[0,1]))
print('  FN = ' + str(cm_test[1,0]))
print('  TP = ' + str(cm_test[1,1]))
print('  Total = ' + str(total_test))

print()
print('Metricas:')
print('  Balanced Accuracy = ' + str(ba_test))
print('  Accuracy = ' + str(accuracy_test))
print('  Precision = ' + str(precision_test))
print('  Recall = ' + str(recall_test))
print('  Specificity = ' + str(specificity_test))
print('  F1-Score = ' + str(f1_test))
print('  ROC-AUC = ' + str(roc_auc_test))
print('  ROC-AUC exacto = ' + str(roc_auc_test))

# Load test report JSON
with open('results/final/final_test_report.json') as f:
    test_data = json.load(f)
print()
print('Archivo JSON: results/final/final_test_report.json')
print('  test total:', test_data['test']['total'])
print('  test NORMAL:', test_data['test']['NORMAL'])
print('  test PNEUMONIA:', test_data['test']['PNEUMONIA'])
print('  umbral:', test_data['umbral'])
print('  ROC-AUC (JSON):', test_data['metricas_umbral_congelado']['roc_auc'])
print('  BA (JSON):', test_data['metricas_umbral_congelado']['balanced_accuracy'])
print('  Accuracy (JSON):', test_data['metricas_umbral_congelado']['accuracy'])
print('  Precision (JSON):', test_data['metricas_umbral_congelado']['precision'])
print('  Recall (JSON):', test_data['metricas_umbral_congelado']['recall'])
print('  Specificity (JSON):', test_data['metricas_umbral_congelado']['specificity'])
print('  F1 (JSON):', test_data['metricas_umbral_congelado']['f1'])
print('  Matriz (JSON):', test_data['matriz_confusion'])
print('  figuras:', test_data['figuras'])

# Model training info
with open('results/final/final_model_training.json') as f:
    model_data = json.load(f)
print()
print('Archivo de entrenamiento definitivo: results/final/final_model_training.json')
print('  model_name:', model_data['model_name'])
print('  splits_entrenamiento:', model_data['splits_entrenamiento'])
print('  composicion_entrenamiento:', model_data['composicion_entrenamiento'])
print('  epochs:', model_data['epochs'])
print('  umbral:', model_data['umbral'])

# Confirmations
print()
print('Confirmaciones:')
test_y_true = np.load('results/final/test_y_true.npy')
print('  Test original: ' + str(total_test) + ' imagenes (' + str((test_y_true==0).sum()) + ' NORMAL / ' + str((test_y_true==1).sum()) + ' PNEUMONIA)')
print('  Modelo entrenado con train+val:', model_data['composicion_entrenamiento'])
print('  Umbral congelado usado en test:', test_data['umbral'])
print('  Archivo matriz:', test_data['figuras']['matriz_confusion'])
print('  Archivo ROC:', test_data['figuras']['curva_roc'])