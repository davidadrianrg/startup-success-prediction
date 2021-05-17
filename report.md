# Startup Success Prediction Model

## David Adrián Rodríguez García & Víctor Caínzos López


---
## Clasificación supervisada


---
### Preprocesado: Preparación de los datos

En primer lugar se realizará un análisis del dataset **startup_data.csv** 
            para el cual se realizará la limpieza de los datos espurios o nulos y se procederá 
            al filtrado de las columnas representativas y la recodificación de variables cualitativas a cuantitativas.
#### Data Missing dataframe: Contiene los datos eliminados del dataset original.


---
|    | feature                  |   missing |   (%) of total |
|---:|:-------------------------|----------:|---------------:|
|  0 | closed_at                |       587 |          63.67 |
|  1 | age_first_milestone_year |       152 |          16.49 |
|  2 | age_last_milestone_year  |       152 |          16.49 |
|  3 | state_code.1             |         1 |           0.11 |

#### Data Spurious dataframe: Contiene los datos sin sentido del dataset original.


---
|     |   age_first_funding_year |   age_last_funding_year |   age_first_milestone_year |   age_last_milestone_year |   age |
|----:|-------------------------:|------------------------:|---------------------------:|--------------------------:|------:|
|  88 |                   0.8822 |                  0.8822 |                     0      |                    0      |    -8 |
| 558 |                  -9.0466 |                 -9.0466 |                    -6.0466 |                   -3.8822 |    -4 |
|  73 |                   1.6685 |                  9.337  |                     7.3808 |                   10.474  |    -2 |
| 350 |                   0.3288 |                  0.3288 |                    -0.4192 |                   -0.4192 |     0 |
| 690 |                   0      |                  0.6904 |                     0      |                    0.6904 |     0 |

#### Data Skewness dataframe: Contiene los datos con alta dispersión del dataset original.


---
|    | feature                |   skewness |
|---:|:-----------------------|-----------:|
|  0 | funding_total_usd      |   27.7868  |
|  1 | state_code             |    2.54287 |
|  2 | relationships          |    2.33271 |
|  3 | age_first_funding_year |    2.32396 |
|  4 | avg_participants       |    1.75802 |

#### Boxplot Feature Skewness > 2: Muestra la dispersión de los datos para las características con asimetría mayor que 2.


---
![Boxplot Feature Skewness](./img/boxplot_fskewness.png)
#### Boxplot Norm Features: Muestra la dispersión de los datos para las características normalizadas.


---
![Boxplot Normalized Feature](./img/boxplot_normalized.png)
#### X dataframe: Contiene la matriz de características.


---
|    |   state_code |   age_last_funding_year |   age_first_milestone_year |   age_last_milestone_year |   funding_rounds |
|---:|-------------:|------------------------:|---------------------------:|--------------------------:|-----------------:|
|  0 |            0 |                  3.0027 |                     4.6685 |                    6.7041 |                3 |
|  1 |            0 |                  9.9973 |                     7.0055 |                    7.0055 |                4 |
|  2 |            0 |                  1.0329 |                     1.4575 |                    2.2055 |                1 |
|  3 |            0 |                  5.3151 |                     6.0027 |                    6.0027 |                3 |
|  4 |            0 |                  1.6685 |                     0.0384 |                    0.0384 |                2 |

|    |   milestones |   is_CA |   is_NY |   is_MA |   is_TX |
|---:|-------------:|--------:|--------:|--------:|--------:|
|  0 |            3 |       1 |       0 |       0 |       0 |
|  1 |            1 |       1 |       0 |       0 |       0 |
|  2 |            2 |       1 |       0 |       0 |       0 |
|  3 |            1 |       1 |       0 |       0 |       0 |
|  4 |            1 |       1 |       0 |       0 |       0 |

|    |   is_otherstate |   is_software |   is_web |   is_mobile |   is_enterprise |
|---:|----------------:|--------------:|---------:|------------:|----------------:|
|  0 |               0 |             0 |        0 |           0 |               0 |
|  1 |               0 |             0 |        0 |           0 |               1 |
|  2 |               0 |             0 |        1 |           0 |               0 |
|  3 |               0 |             1 |        0 |           0 |               0 |
|  4 |               0 |             0 |        0 |           0 |               0 |

|    |   is_advertising |   is_gamesvideo |   is_ecommerce |   is_biotech |   is_consulting |
|---:|-----------------:|----------------:|---------------:|-------------:|----------------:|
|  0 |                0 |               0 |              0 |            0 |               0 |
|  1 |                0 |               0 |              0 |            0 |               0 |
|  2 |                0 |               0 |              0 |            0 |               0 |
|  3 |                0 |               0 |              0 |            0 |               0 |
|  4 |                0 |               1 |              0 |            0 |               0 |

|    |   is_othercategory |   has_VC |   has_angel |   has_roundA |   has_roundB |
|---:|-------------------:|---------:|------------:|-------------:|-------------:|
|  0 |                  1 |        0 |           1 |            0 |            0 |
|  1 |                  0 |        1 |           0 |            0 |            1 |
|  2 |                  0 |        0 |           0 |            1 |            0 |
|  3 |                  0 |        0 |           0 |            0 |            1 |
|  4 |                  0 |        1 |           1 |            0 |            0 |

|    |   has_roundC |   has_roundD |   avg_participants |   is_top500 |   age |
|---:|-------------:|-------------:|-------------------:|------------:|------:|
|  0 |            0 |            0 |             1      |           0 |     7 |
|  1 |            1 |            1 |             4.75   |           1 |    14 |
|  2 |            0 |            0 |             4      |           1 |     5 |
|  3 |            1 |            1 |             3.3333 |           1 |    12 |
|  4 |            0 |            0 |             1      |           1 |     2 |

|    |   norm_funding_total_usd |   norm_age_first_funding_year |   norm_relationships |
|---:|-------------------------:|------------------------------:|---------------------:|
|  0 |                 0.268198 |                      0.376383 |             0.333333 |
|  1 |                 0.623283 |                      0.57891  |             0.553655 |
|  2 |                 0.415358 |                      0.226596 |             0.430827 |
|  3 |                 0.623093 |                      0.453101 |             0.430827 |
|  4 |                 0.36268  |                      0        |             0.26416  |

#### t dataframe: Contiene el vector de etiquetas.


---
|    |   labels |
|---:|---------:|
|  0 |        1 |
|  1 |        1 |
|  2 |        1 |
|  3 |        1 |
|  4 |        0 |

### Entrenamiento: Comparativa de modelos de aprendizaje automático

Se procederá a comparar los resultados obtenidos de diferentes modelos de aprendizaje automático
            variando tanto el tipo de modelo como los hiperparámetros de los que depende con el objetivo
            de obtener el mejor modelo que prediga el éxito o fracaso de las diferentes startups
#### Results dataframe: Muestra los resultados de los mejores modelos obtenidos


---
|   Folds |   LR_train_accuracy |   LR_val_accuracy |   LDA_train_accuracy |   LDA_val_accuracy |   KNN_train_accuracy |
|--------:|--------------------:|------------------:|---------------------:|-------------------:|---------------------:|
|       1 |            0.847176 |          0.80597  |             0.818937 |           0.835821 |             0.770764 |
|       2 |            0.833887 |          0.865672 |             0.830565 |           0.80597  |             0.77907  |
|       3 |            0.853821 |          0.776119 |             0.820598 |           0.820896 |             0.759136 |
|       4 |            0.842193 |          0.895522 |             0.82392  |           0.80597  |             0.784053 |
|       5 |            0.850498 |          0.850746 |             0.825581 |           0.850746 |             0.775748 |

|   Folds |   KNN_val_accuracy |   SVC_train_accuracy |   SVC_val_accuracy |   DNN_train_accuracy |   DNN_val_accuracy |
|--------:|-------------------:|---------------------:|-------------------:|---------------------:|-------------------:|
|       1 |           0.791045 |             0.863787 |           0.865672 |             0.80495  |           0.769478 |
|       2 |           0.716418 |             0.863787 |           0.80597  |             0.814311 |           0.757537 |
|       3 |           0.746269 |             0.853821 |           0.820896 |             0.766329 |           0.76306  |
|       4 |           0.701493 |             0.872093 |           0.746269 |             0.777932 |           0.77306  |
|       5 |           0.776119 |             0.873754 |           0.716418 |             0.789693 |           0.791791 |

|   Folds |   LR_train_recall |   LR_val_recall |   LDA_train_recall |   LDA_val_recall |   KNN_train_recall |
|--------:|------------------:|----------------:|-------------------:|-----------------:|-------------------:|
|       1 |          0.914573 |        0.911111 |           0.906329 |         0.909091 |             0.965  |
|       2 |          0.904523 |        0.977778 |           0.911392 |         0.886364 |             0.97   |
|       3 |          0.91206  |        0.933333 |           0.906329 |         0.909091 |             0.9625 |
|       4 |          0.914787 |        0.931818 |           0.916456 |         0.863636 |             0.965  |
|       5 |          0.927318 |        0.863636 |           0.908861 |         0.931818 |             0.9625 |

|   Folds |   KNN_val_recall |   SVC_train_recall |   SVC_val_recall |   DNN_train_recall |   DNN_val_recall |
|--------:|-----------------:|-------------------:|-----------------:|-------------------:|-----------------:|
|       1 |         1        |           0.918575 |         0.953488 |           0.80495  |         0.769478 |
|       2 |         0.933333 |           0.916031 |         0.906977 |           0.814311 |         0.757537 |
|       3 |         0.933333 |           0.918575 |         0.860465 |           0.766329 |         0.76306  |
|       4 |         0.933333 |           0.928571 |         0.863636 |           0.777932 |         0.77306  |
|       5 |         0.955556 |           0.928571 |         0.818182 |           0.789693 |         0.791791 |

|   Folds |   LR_train_specificity |   LR_val_specificity |   LDA_train_specificity |   LDA_val_specificity |   KNN_train_specificity |
|--------:|-----------------------:|---------------------:|------------------------:|----------------------:|------------------------:|
|       1 |               0.715686 |             0.590909 |                0.652174 |              0.695652 |                0.386139 |
|       2 |               0.696078 |             0.636364 |                0.676329 |              0.652174 |                0.40099  |
|       3 |               0.740196 |             0.454545 |                0.657005 |              0.652174 |                0.356436 |
|       4 |               0.699507 |             0.826087 |                0.647343 |              0.695652 |                0.425743 |
|       5 |               0.699507 |             0.826087 |                0.666667 |              0.695652 |                0.405941 |

|   Folds |   KNN_val_specificity |   SVC_train_specificity |   SVC_val_specificity |   DNN_train_specificity |   DNN_val_specificity |
|--------:|----------------------:|------------------------:|----------------------:|------------------------:|----------------------:|
|       1 |              0.363636 |                0.760766 |              0.708333 |                0.805547 |              0.77534  |
|       2 |              0.272727 |                0.76555  |              0.625    |                0.813461 |              0.75875  |
|       3 |              0.363636 |                0.732057 |              0.75     |                0.765211 |              0.772583 |
|       4 |              0.227273 |                0.766667 |              0.521739 |                0.777461 |              0.773525 |
|       5 |              0.409091 |                0.771429 |              0.521739 |                0.79075  |              0.78819  |

|   Folds |   LR_train_precision |   LR_val_precision |   LDA_train_precision |   LDA_val_precision |   KNN_train_precision |
|--------:|---------------------:|-------------------:|----------------------:|--------------------:|----------------------:|
|       1 |             0.862559 |           0.82     |              0.832558 |            0.851064 |              0.756863 |
|       2 |             0.853081 |           0.846154 |              0.843091 |            0.829787 |              0.762279 |
|       3 |             0.872596 |           0.777778 |              0.834499 |            0.833333 |              0.747573 |
|       4 |             0.856808 |           0.911111 |              0.832184 |            0.844444 |              0.768924 |
|       5 |             0.858469 |           0.904762 |              0.838785 |            0.854167 |              0.762376 |

|   Folds |   KNN_val_precision |   SVC_train_precision |   SVC_val_precision |   DNN_train_precision |   DNN_val_precision |
|--------:|--------------------:|----------------------:|--------------------:|----------------------:|--------------------:|
|       1 |            0.762712 |              0.878345 |            0.854167 |              0.80495  |            0.769478 |
|       2 |            0.724138 |              0.880196 |            0.8125   |              0.814311 |            0.757537 |
|       3 |            0.75     |              0.865707 |            0.860465 |              0.766329 |            0.76306  |
|       4 |            0.711864 |              0.881356 |            0.77551  |              0.777932 |            0.77306  |
|       5 |            0.767857 |              0.883495 |            0.765957 |              0.789693 |            0.791791 |

|   Folds |
|--------:|
|       1 |
|       2 |
|       3 |
|       4 |
|       5 |

#### Boxplot models: Muestra los valores de exactitud de los diferentes modelos


---
![Boxplot Models Accuracy](./img/boxplot_models_accuracy.png)
#### Contraste de hipótesis: Comparación de modelos mediante el test de Kruskal-Wallis


---

```no-format
p-valor KrusW:0.01136941395786562
Hypotheses are being rejected: the models are different
  Multiple Comparison of Means - Tukey HSD, FWER=0.05  
=======================================================
 group1   group2  meandiff p-adj   lower  upper  reject
-------------------------------------------------------
modelDNN modelKNN  -0.0151    0.9 -0.0763 0.0461  False
modelDNN modelLDA   0.0357 0.4718 -0.0255 0.0969  False
modelDNN  modelLR   0.0417 0.3131 -0.0195 0.1029  False
modelDNN modelSVC   0.0582 0.0698  -0.003 0.1194  False
modelKNN modelLDA   0.0508 0.1462 -0.0104  0.112  False
modelKNN  modelLR   0.0568 0.0804 -0.0044  0.118  False
modelKNN modelSVC   0.0733 0.0117  0.0121 0.1345   True
modelLDA  modelLR   0.0061    0.9 -0.0551 0.0673  False
modelLDA modelSVC   0.0225 0.8122 -0.0387 0.0837  False
 modelLR modelSVC   0.0164    0.9 -0.0448 0.0776  False
-------------------------------------------------------
```

#### Matrices de confusión: Compara los valores reales con los valores predichos para cada modelo


---
![Matriz de confusión LR](./img/confusion_matrix_LR.png)
![Matriz de confusión LDA](./img/confusion_matrix_LDA.png)
![Matriz de confusión KNN](./img/confusion_matrix_KNN.png)
![Matriz de confusión SVC](./img/confusion_matrix_SVC.png)


**Matriz de confusión DNN**

|                  |   Clase Real 0 |   Clase Real 1 |
|:-----------------|---------------:|---------------:|
| Clase Predicha 0 |             42 |             14 |
| Clase Predicha 1 |             10 |            102 |

#### Curva ROC: Compara el ajuste entre la especificidad y la sensibilidad para cada modelo


---
![Curva ROC de LR](./img/roc_curve_LR.png)
![Curva ROC de LDA](./img/roc_curve_LDA.png)
![Curva ROC de KNN](./img/roc_curve_KNN.png)
![Curva ROC de SVC](./img/roc_curve_SVC.png)
![Curva ROC de DNN](./img/roc_curve_dnn.png)
#### Informe de clasificación: Compara los resultados de cada modelo de clasificación


---

```no-format
Classification report for model LR:

              precision    recall  f1-score   support

           0       0.75      0.69      0.72        58
           1       0.84      0.88      0.86       110

    accuracy                           0.82       168
   macro avg       0.80      0.79      0.79       168
weighted avg       0.81      0.82      0.81       168

```


```no-format
Classification report for model LDA:

              precision    recall  f1-score   support

           0       0.71      0.60      0.65        58
           1       0.81      0.87      0.84       110

    accuracy                           0.78       168
   macro avg       0.76      0.74      0.75       168
weighted avg       0.77      0.78      0.77       168

```


```no-format
Classification report for model KNN:

              precision    recall  f1-score   support

           0       0.77      0.34      0.48        58
           1       0.73      0.95      0.83       110

    accuracy                           0.74       168
   macro avg       0.75      0.65      0.65       168
weighted avg       0.75      0.74      0.70       168

```


```no-format
Classification report for model SVC:

              precision    recall  f1-score   support

           0       0.74      0.67      0.70        58
           1       0.83      0.87      0.85       110

    accuracy                           0.80       168
   macro avg       0.79      0.77      0.78       168
weighted avg       0.80      0.80      0.80       168

```


```no-format
Classification report for model DNN:

              precision    recall  f1-score   support

           0       0.81      0.75      0.78        56
           1       0.88      0.91      0.89       112

    accuracy                           0.86       168
   macro avg       0.84      0.83      0.84       168
weighted avg       0.86      0.86      0.86       168

```

#### Exactitud media: Compara la exactitud de cada modelo en función de sus hiperparámetros


---
![Exactitud media de LR](./img/mean_accuracy_LR.png)
![Exactitud media de LDA](./img/mean_accuracy_LDA.png)
![Exactitud media de KNN](./img/mean_accuracy_KNN.png)
![Exactitud media de SVC](./img/mean_accuracy_SVC.png)
#### Curva de validación: Compara los resultados del modelo en función de sus hiperparámetros


---
![Validation Curve](./img/validation_curve_dnn.png)
#### Hiperparámetros: Muestra el dataframe con los hiperparámetros usados en el entrenamiento


---
##### Hiperparámetros del modelo LR

|             |       C |   max_iter | penalty   | solver   |
|:------------|--------:|-----------:|:----------|:---------|
| hyperparams | 7.12741 |        575 | l2        | lbfgs    |

##### Hiperparámetros del modelo LDA

|             | shrinkage   | solver   |     tol |
|:------------|:------------|:---------|--------:|
| hyperparams | auto        | lsqr     | 0.00099 |

##### Hiperparámetros del modelo KNN

|             |   n_neighbors | weights   |
|:------------|--------------:|:----------|
| hyperparams |            64 | uniform   |

##### Hiperparámetros del modelo SVC

|             |       C | decision_function_shape   | gamma   | kernel   | probability   |
|:------------|--------:|:--------------------------|:--------|:---------|:--------------|
| hyperparams | 8.51396 | ovr                       | auto    | linear   | True          |

##### Hiperparámetros de la red neuronal

|         |   neurons | activation   |
|:--------|----------:|:-------------|
| layer 0 |        11 | relu         |
| layer 1 |        17 | sigmoid      |
| layer 2 |        14 | relu         |
| layer 3 |        11 | relu         |
| layer 4 |         2 | softmax      |

|          | optimizer   |      lr |
|:---------|:------------|--------:|
| compiler | Adam        | 0.00017 |

## Clasificación No Supervisada


---
#### Reducción de la dimensionalidad

![PCA](./img/pca.png)
#### Clustering

##### K-means clustering

![KMeans Clustering](./img/kmeans.png)
##### DBSCAN clustering

![DBSCAN Clustering](./img/dbscan.png)
#### Detección de anomalías

##### Isolation Forest - Classification Report


```no-format
              precision    recall  f1-score   support

          -1       0.00      0.00      0.00         0
           0       0.00      0.00      0.00        28
           1       0.80      0.99      0.88       110

    accuracy                           0.79       138
   macro avg       0.27      0.33      0.29       138
weighted avg       0.63      0.79      0.70       138

```

##### Local Oulier Factor (LOF) - Classification Report


```no-format
              precision    recall  f1-score   support

          -1       0.00      0.00      0.00         0
           0       0.00      0.00      0.00        28
           1       0.80      0.99      0.88       110

    accuracy                           0.79       138
   macro avg       0.27      0.33      0.29       138
weighted avg       0.63      0.79      0.70       138

```

##### Autoencoding


---
![Autoencoder Validation](./img/autoencoder_validation.png)
![Autoencoder Threshold](./img/autoencoder_threshold.png)
![Autoencoder Error](./img/autoencoder_error.png)
##### Autoencoding Classification Report


```no-format
              precision    recall  f1-score   support

           0       0.00      0.00      0.00        28
           1       0.80      1.00      0.89       110

    accuracy                           0.80       138
   macro avg       0.40      0.50      0.44       138
weighted avg       0.64      0.80      0.71       138

```

## Registro de tiempos de entrenamiento


---
El entrenamiento se ha realizado en computación multihilo
#### Tiempos del entrenamiento de los modelos de clasificación supervisada

|             |   LR |   LDA |   KNN |   SVC |    DNN |
|:------------|-----:|------:|------:|------:|-------:|
| Time Models | 2.32 |  2.04 |  8.99 |  9.46 | 740.15 |

