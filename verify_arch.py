import json
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os
from datetime import datetime

print('='*80)
print('VERIFICACION COMPARACION ARQUITECTURAS - combinado-arquitecturas')
print('='*80)

for arch in ['vgg16', 'resnet50', 'mobilenetv2']:
    json_path = f'results/final/combinado_{arch}_validacion.json'
    with open(json_path) as f:
        data = json.load(f)
    
    # Load npy files
    y_true = np.load(f'results/final/validacion_{arch}_y_true.npy')
    y_prob = np.load(f'results/final/validacion_{arch}_y_prob.npy')
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Check image files exist and their timestamps
    img_cm = data['figuras']['matriz_confusion']
    img_roc = data['figuras']['curva_roc']
    
    print()
    print('='*80)
    model_name = data['model_name']
    print(model_name)
    print('='*80)
    print('JSON: ' + json_path)
    print('  finalizado: ' + data['finalizado'])
    print('  validation_n: ' + str(data['validation_n']))
    print('  umbral_medicion: ' + str(data['umbral_medicion']))
    print('  Matriz JSON: ' + str(data['validation_matriz_confusion']))
    print('  BA JSON: ' + str(data['validation']['balanced_accuracy']))
    print('  Accuracy JSON: ' + str(data['validation']['accuracy']))
    print('  Precision JSON: ' + str(data['validation']['precision']))
    print('  Recall JSON: ' + str(data['validation']['recall']))
    print('  Specificity JSON: ' + str(data['validation']['specificity']))
    print('  F1 JSON: ' + str(data['validation']['f1']))
    print('  ROC-AUC JSON: ' + str(data['validation']['roc_auc']))
    print('  ROC-AUC exacto: ' + str(data['validation']['roc_auc']))
    print('  Matriz recalc (npy): TN=' + str(cm[0,0]) + ', FP=' + str(cm[0,1]) + ', FN=' + str(cm[1,0]) + ', TP=' + str(cm[1,1]) + ', Total=' + str(cm.sum()))
    print('  ROC-AUC recalc: ' + str(roc_auc))
    print('  ROC-AUC redondeado (imagen): ' + str(round(roc_auc, 4)))
    print('  y_true: results/final/validacion_' + arch + '_y_true.npy')
    print('  y_prob: results/final/validacion_' + arch + '_y_prob.npy')
    print('  Imagen matriz: ' + img_cm)
    print('  Imagen ROC: ' + img_roc)
    if os.path.exists(img_cm):
        mtime = datetime.fromtimestamp(os.path.getmtime(img_cm))
        print('  Imagen matriz timestamp: ' + str(mtime))
    if os.path.exists(img_roc):
        mtime = datetime.fromtimestamp(os.path.getmtime(img_roc))
        print('  Imagen ROC timestamp: ' + str(mtime))
    
    # Verify JSON matches recalc
    json_tn = data['validation_matriz_confusion']['tn']
    json_fp = data['validation_matriz_confusion']['fp']
    json_fn = data['validation_matriz_confusion']['fn']
    json_tp = data['validation_matriz_confusion']['tp']
    if json_tn != cm[0,0] or json_fp != cm[0,1] or json_fn != cm[1,0] or json_tp != cm[1,1]:
        print('  >>> DISCREPANCIA MATRIZ: JSON=(' + str(json_tn) + ',' + str(json_fp) + ',' + str(json_fn) + ',' + str(json_tp) + ') vs NPY=(' + str(cm[0,0]) + ',' + str(cm[0,1]) + ',' + str(cm[1,0]) + ',' + str(cm[1,1]) + ')')
    else:
        print('  >>> Matriz JSON coincide con NPY')
    
    json_roc = data['validation']['roc_auc']
    if abs(json_roc - roc_auc) > 1e-10:
        print('  >>> DISCREPANCIA ROC-AUC: JSON=' + str(json_roc) + ' vs NPY=' + str(roc_auc))
    else:
        print('  >>> ROC-AUC JSON coincide con NPY')