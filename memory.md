# <center> Startup Succes Prediction Model <center>

## <center> Aprendizaje Automático I  
## <center>Máster en Informática Industrial y Robótica
---
###  <center> _Víctor Caínzos López & David Adrián Rodríguez García_
<br>

## Introducción
<p style='text-align: justify;'> El objetivo de nuestro modelo de aprendizaje automático será predecir si una empresa puede tener éxito o
no, clasificándola en 0 o 1 en función de si el sistema recomienda invertir o no en la empresa. Además,
podrá obtenerse la probabilidad dentro de ese rango [0,1] que da una idea de lo más o menos viable que
puede llegar a ser la inversión en esa compañía, y ordenar las diferentes compañías por orden de viabilidad.
Para cumplir este objetivo se probarán y se compararán diferentes clasificadores empleando Validación
Cruzada y utilizando diferentes métricas de evaluación.
Por último se seleccionará el modelo más óptimo, el cual permitiría ser llevado a producción por empresas
del tipo Venture Capital para hacer sus estudios de viabilidad de inversión en startups.
  
<p style='text-align: justify;'>En esta memoria se explican las instrucciones principales del programa para que este se ejecute correctamente, especialmente las sentencias definidas en el archivo:

[```main.py```](https://github.com/davidadrianrg/startup-success-prediction/blob/master/main.py), cuya documentación se puede estudiar en [```main```](https://davidadrianrg.com/startup-success-prediction/app.html).

<p style='text-align: justify;'>No obtante, se recomienda consultar toda la documentación para conocer en profundidad la funcionalidad de los módulos internos en los que se basa su funcionamiento, respondiendo a la siguiente estructura:

<center>

| [```preprocessing```](https://davidadrianrg.com/startup-success-prediction/util.html#module-preprocessing.preprocessing)   |      [```models```](https://davidadrianrg.com/startup-success-prediction/util.html#module-preprocessing.preprocessing)      |  [```postprocessing```](https://davidadrianrg.com/startup-success-prediction/classes.html) |
|----------|:-------------:|------:|
| [preprocessing.py](https://github.com/davidadrianrg/startup-success-prediction/blob/master/preprocessing/preprocessing.py) |  [customized_metrics.py](https://github.com/davidadrianrg/startup-success-prediction/blob/master/models/customized_metrics.py) | [report.py](https://github.com/davidadrianrg/startup-success-prediction/blob/master/postprocessing/report.py) |
|  |    [hyperparametersTunning.py](https://github.com/davidadrianrg/startup-success-prediction/blob/master/models/hyperparametersTunning.py)   |    |
|  | [hyperparametersDNN.py](https://github.com/davidadrianrg/startup-success-prediction/blob/master/models/hyperparametersDNN.py) |     |
|  | [models_evaluation.py](https://github.com/davidadrianrg/startup-success-prediction/blob/master/models/models_evaluation.py) |     

</center>
<br>

De la misma forma, se aconseja revisar el documento:
  
[```README.md```](https://github.com/davidadrianrg/startup-success-prediction/blob/master/README.md)

<p style='text-align: justify;'>Conforme se tienen instalados todos los paquetes necesarios para la ejecución del programa.

```python
# Importing required modules

import numpy as np
import pandas as pd

from models import models_evaluation as mdleval
from postprocessing.report import Report
from preprocessing import preprocessing as prp
```

## Índice
1. [Descripción de los datos empleados](#id1)
2. [Experimentos realizados](#id2)
2. [Análisis de los resultados](#id3)
2. [Conclusiones obtenidas](#id4)

<div id='id1' />

## 1. Descripción de los datos empleados
<p style='text-align: justify;'> El conjunto de datos sobre el que se realizarán los modelos de aprendizaje automático es el
correspondiente al DataSet obtenido de Kaggle, denominado Startup Success Prediction.
El conjunto de datos consistirá en un total de 48 columnas con datos de tipo cuantitativo y categórico que
reflejan las diferentes tendencias de la industria, perspectivas de inversión e información de compañías
individuales de Estados Unidos, algunas de las columnas más interesantes tratan temas como las diferentes
rondas de financiación de las startups, el año de fundación, el estado donde está ubicada la empresa, si ha
sido comprada por otra empresa, el tipo de sector al que pertenece, entre otras.

<p style='text-align: justify;'> El preprocesado y limpieza del conjunto de datos se describe en los siguientes subapartados y se lleva a cabo en el módulo:  

- [*```preprocessing```*](https://github.com/davidadrianrg/startup-success-prediction/tree/master/preprocessing)
    - [*```preprocessing.py```*](https://github.com/davidadrianrg/startup-success-prediction/blob/master/preprocessing/preprocessing.py)

Inicialmente, se lee el dataset descargado previamente del repositorio: 

- [StartUp Succes Prediction Dataset](https://www.kaggle.com/manishkc06/startup-success-prediction).

```python
def make_preprocessing(filepath: str) -> pd.DataFrame:
    """Clean and prepare the given dataframe to train ML models."""
    # Load a dataframe to preprocess from the given filepath
    data = prp.read_dataset(filepath)
```

### 1.1. Eliminación de datos espurios
<p style='text-align: justify;'> En primer lugar se procede a analizar el conjunto de muestras de las variables de entrada de nuestro problema, prescidiendo de todas aquellas que no aporten información lo suficientemente relevante como para tenerse en cuenta, además de limpiar las duplicadas:  

```python
# Clean the dataset of lost values
    drop_duplicates = ["name"]
    to_drop = [
        "Unnamed: 0",
        "Unnamed: 6",
        "latitude",
        "longitude",
        "zip_code",
        "object_id",
        "status",
    ]
    data, data_missing = prp.drop_values(data, drop_duplicates, to_drop) 
```
<p style='text-align: justify;'> Además, se obtiene un DataFrame con las variables que contienen valores perdidos cuyo contenido se muestra en el report.    
  
- _data_missing_

### 1.2 Imputación de valores
<p style='text-align: justify;'>Una vez se han separado las variables que no son de utilidad para el estudio, es preciso continuar con la preparación de las que sí lo son. En este caso estudiando la relación de las variables que contienen datos perdidos con el fin de estimar un valor coherente de acuerdo a su significado.  

<p style='text-align: justify;'> En el caso concreto de las características:  
  
- *age_first_milestone_year*
- *age_last_milestone_year* 

<p style='text-align: justify;'> Se imputan los valores nulos, considerando que si no se tienen datos, quiere decir que no se han registrado hitos.  
  
Por otra parte, las variables correspondientes al inicio y cierre de la empresa, se codifican a formato fecha y se corelacionan entre sí para definir una nueva característica que cuantifica los años de vida de la empresa. Para acotar el estudio temporalmente, se supone como fecha máxima de cierre el último día del año más tardío que tiene constancia en el DataSet:
  
- *age*

```python
# Fill empty values with zeros
    empty_labels = ["age_first_milestone_year", "age_last_milestone_year"]
    data = prp.fill_empty_values(data, empty_labels)

    # Transform data values to datatime
    date_labels = ["closed_at", "founded_at"]
    data = prp.to_datetime(data, date_labels)

    # Use the last date of the last year registered considered as the end of the study period
    data = prp.to_last_date(data, "closed_at")

    # Define the ages of the startups being active as a new feature
    data["age"] = data["last_date"] - data["founded_at"]
    data["age"] = round(data.age / np.timedelta64(1, "Y"))

    # Clean the dataset from spurious data, negative years and non-sense data
    data, data_spurious = prp.eliminate_spurious_data(data, "age")
```
<p style='text-align: justify;'> De lo anterior, también se puede extraer un DataFrame de carácter informñativo que describe un resumen de las muestras que carecen de sentido, en concreto aquellas que registran años negativos. 

- *data_spurious*

<p style='text-align: justify;'>Estas muestras son eliminadas de acuerdo al subapartado anterior.  

### 1.3. Recodificación de variables no numéricas
<p style='text-align: justify;'> Recordando que se trata de un análisis de clasificación supervisada en el cual, se facilitan las etiquetas o variables de salida reales al modelo de aprendizaje para el conjunto de datos en cuestión, resulta de gran interés poder contar no sólo con el mayor número de muestras posible sino también con todas aquellas variables que sí sean relevantes por sí mismas o por combinación con otras de cara a la generalización del sistema.
  
<p style='text-align: justify;'>Este es el caso de la característica:

- *state_code*
  
<p style='text-align: justify;'>Se trata de una variable categórica que ha de ser recodificada a formato numérico, que aporta información sobre el estado territorial de la empresa.

```python
# Recode non numerical columns to numerical ones
    non_numerical = "state_code"
    data = prp.non_numerical_recoding(data, non_numerical)
```
### 1.4 Normalización
<p style='text-align: justify;'>Para terminar con este apartado de preprocesado y limpieza de datos, es necesario no pasar por alto la distribución de los mismos, pretendiendo que se adapten a una normal N(0,1).

<p style='text-align: justify;'>A modo de preparación, se estudia qué variables presentan una asimetría estadística superior, siendo estas las candidatas a ser normalizadas. 

```python
# Normalize quantitative data
    data, data_skewness, features, norm_features = prp.data_normalization(data)
```
<p style='text-align: justify;'>De la función anterior se obtiene junto con el DataSet preprocesado, un DataFrame descriptivo en función la asimetría de la distribución y las variables cadidatas antes y después de la normalización.
  
- *data_skewness*
- *features*
- *norm_features*

<div id='id2' />

## 2. Experimentos realizados
<p style='text-align: justify;'> Se emplean en el proyecto diferente técnicas de aprendizaje automático implementadas en Scikit-Learn y una red de neuronas profunda (deep network) implementada en Keras.  
  
<p style='text-align: justify;'> Los experimetos realizados se explican en los siguientes subapartados y se llevan a cabo utilizando las funciones definidas en los módulo:  

- [*```models```*](https://github.com/davidadrianrg/startup-success-prediction/tree/master/models), la documentación completa se puede consultar en [```models```](https://davidadrianrg.com/startup-success-prediction/util.html#module-models.models_evaluation):
    - [*```customized_metrics.py```*](https://github.com/davidadrianrg/startup-success-prediction/blob/master/models/customized_metrics.py)
    - [*```hyperparametersTunning.py```*](https://github.com/davidadrianrg/startup-success-prediction/blob/master/models/hyperparametersTunning.py)
    - [*```hyperparametersDNN.py```*](https://github.com/davidadrianrg/startup-success-prediction/blob/master/models/hyperparametersDNN.py)
    - [*```models_evaluation.py```*](https://github.com/davidadrianrg/startup-success-prediction/blob/master/models/models_evaluation.py)

La función principal en ```main.py```:

```python
def train_models(X: np.ndarray, t: np.ndarray):
    """Train different models using the dataset and its labels.

    :param X: Input values of the dataset
    :type X: numpy.ndarray
    :param t: Label values for the exit of the dataset
    :type t: numpy.ndarray
    """
```

### 2.1. Optimización de los modelos
<p style='text-align: justify;'>Una vez se han escogido los modelos que se van a estudiar, es importante conocer cuál es la combinación óptima de sus hiperparámetros que ofrecen su mejor versión de cara a la resolución del problema. Para ello se definen una serie de funciones en los módulos que se citan a continuación:

[```customized_metrics.py```](https://github.com/davidadrianrg/startup-success-prediction/blob/master/models/customized_metrics.py)

<p style='text-align: justify;'>Se definen aquellas métricas que no tienen una función implementada en sus correspondiente librerías.
  
[```hyperparametersTunning.py```](https://github.com/davidadrianrg/startup-success-prediction/blob/master/models/hyperparametersTunning.py)

<p style='text-align: justify;'>Este es un módulo específico para los modelos de la librería de Scikit-Learn.  

<p style='text-align: justify;'>Se incluyen funciones para la selección, creación y proceso de optimización en función del rango de hiperparámetros establecido.

[```hyperparametersDNN.py```](https://github.com/davidadrianrg/startup-success-prediction/blob/master/models/hyperparametersDNN.py)
<p style='text-align: justify;'>De forma análoga al módulo anterior, se definen una serie de funciones orientadas para trabajar con redes de neuronas de la librería Keras. Una función principal se encarga de realizar el proceso de optimización, comparando distintas redes creadas de forma aleatoria de acuerdo con un rango de hiperparámetros determinado.

[```models_evaluation.py```](https://github.com/davidadrianrg/startup-success-prediction/blob/master/models/models_evaluation.py)
<p style='text-align: justify;'> Con el fin de automatizar el proceso de optimización de todos los modelos empleados, se define este módulo contenedor, de manera que facilite las instrucciones de operación del programa.

### 2.2. Validación cruzada
<p style='text-align: justify;'>Tanto para el entrenamiento de los modelos de Scikit-Learn como para la red de neuronas, se realiza una validación cruzada K-Fold, especificando el número de subconjuntos de validación, el número de iteraciones (epochs para el caso de la red) y el tamaño de los lotes de muestras para la actualización de los pesos de la red de neuronas.

<p style='text-align: justify;'>Este proceso de entrenamiento, se realiza para todos los modelos probados como potencialmente mejores. Calificando como óptimo el que obtiene puntuaciones más adecuadas para la métrica de error escogida. Generalmente, se estudiará la exactitud de los modelos como filtro de análisis, no obstante, el error también puede utilizarse en según qué casos.

<p style='text-align: justify;'>A continuación se muestra la función principal que obtiene los mejores modelos para los parámetros seleccionados:

```python
# Wrapper function of optimize_models and optimize_DNN functions in hyperparameters modules
    # Return a tuple with a dict with the best models validated and the train size and the best DNN model
    # Using the hyperperameters ranges given in the arguments
    best_models, best_DNN = mdleval.get_best_models(X, t)
``` 

<p style='text-align: justify;'>Después de este proceso de optimización definido por la creación de modelos en base a sus hiperparámetros de forma aleatoria, el entrenamiento mediante validación cruzada y la actualización, se conocerían cuáles son los mejores modelos y por consiguiente, qué combinación de hiperparámetros responde mejor. Es importante tener en cuenta el hecho de particionar el conjunto de muestras de forma que los modelos utilicen una parte para entrenamiento y validación, definida por:

- *train_size*

<p style='text-align: justify;'>y otra (que no haya sido utilizada previamente por el modelo) para el análisis del rendimiento. Antes de analizar el comportamiento de cada modelo de forma individual, se guardan de forma ordenada los resultados obtenidos en el proceso de entrenamiento:

```python
# Return a dataframe with validation results of the models in visutalization mode.
    results = mdleval.get_results(best_models, best_DNN)

    return results, best_models, best_DNN
```
<p style='text-align: justify;'>Utilizando estos resultados se pueden obtener las curvas de exactitud media por fold para cada modelo, así como la curva de validación para la red de neuronas profunda.

### 2.3. Análisis del rendimiento

<p style='text-align: justify;'>Después de obtener los mejores modelos en base a cuál de ellos generaliza mejor que los demás, estos pueden ser evaluados en términos de rendimiento utilizando las métricas correspondientes y el conjunto de datos reservado para tal cometido:

- *test_size*

<p style='text-align: justify;'>Utilizando las funciones definidas en el módulo:

- [*```models```*](https://github.com/davidadrianrg/startup-success-prediction/blob/master/models)
    - [*```models_evaluation.py```*](https://github.com/davidadrianrg/startup-success-prediction/blob/master/models/models_evaluation.py)


 <p style='text-align: justify;'>Se obtienen las salidas predichas y sus probabilidades por clase, posteriormente utilizadas para calcular la matriz de cofusión y la curva ROC y AUC.

### 2.4. Estudio comparativo
<p style='text-align: justify;'>Se emplean contrastes de hipótesis para determinar si las diferencias de rendimiento entre los modelos son estadísticamente significativas y determinar de esta forma cuál sería el mejor modelo para resolver el reto planteado.
  
<p style='text-align: justify;'>En primer lugar se estudia si los resultados son iguales y en caso contrario se emplea el método de comparación múltiple, respectivamente:

- *Kruskal-Wallis*
- *Tukey*

<div id='id3' />

## 3. Análisis de los resultados
<p style='text-align: justify;'>Todos los resultados obtenidos para los experimentos realizados a lo largo del proyecto, descritos en el apartado anterior, se recogen en un report, donde se pueden visualizar de forma ordenada y comparativa.
  
<p style='text-align: justify;'>Para la generación de este documento, se crea un módulo específico encargado de construír y elaborar DataFrames explicativos de los datos empleados y los resultados obtenidos de la validación cruzada para cada modelo, además de los correspondientes diagramas de cajas y gráficas de evaluación como las matrices de confusión o las curvas ROC y AUC. Este se encuentra en:

- [*```postprocessing```*](https://github.com/davidadrianrg/startup-success-prediction/tree/master/postprocessing)
    - [*```report.py```*](https://github.com/davidadrianrg/startup-success-prediction/blob/master/postprocessing/report.py), la documentación se encuentra en [```report```](https://davidadrianrg.com/startup-success-prediction/classes.html)

La función principal en [```main.py```](https://github.com/davidadrianrg/startup-success-prediction/blob/master/main.py), cuya documentación está en [```main```](https://davidadrianrg.com/startup-success-prediction/app.html):

```python
def make_report(df_dict: dict, features_list: list, results: pd.DataFrame, best_models: tuple, best_DNN: tuple):
    """Generate a report taking into account the given data."""
    # Processing the data to be passed to the Report class in the right format
    results_tags = []
    for i in results.columns:
        if len(i.split("_val_accuracy")) > 1:
            results_tags.append(i.split("_val_accuracy")[0])

    results_data = {}
    results_labels = []
    for tag in results_tags:
        results_data.update({tag: results[tag + "_val_accuracy"]})
        results_labels.append(tag)
```
Y se generan las cabeceras del documento:
```python
with Report(generate_pdf=True) as report:
        # Generating the header
        report.print_title("Startup Success Prediction Model")
        report.print_title("David Adrián Rodríguez García & Víctor Caínzos López", 2)
        report.print_line()
```

### 3.1. Datos empleados
<p style='text-align: justify;'>Se incluyen los DataFrames explicativos referentes al conjunto de datos y al preprocesado de los mismos, presentando los valores perdidos, asimetría estadística y normalización comparativa utilizando diagramas de cajas.

```python
# Generating preprocessing report chapter
        report.print_title("Preprocesado: Preparación de los datos", 3)
        report.print(
            """En primer lugar se realizará un análisis del dataset **startup_data.csv** 
            para el cual se realizará la limpieza de los datos espurios o nulos y se procederá 
            al filtrado de las columnas representativas y la recodificación de variables cualitativas a cuantitativas."""
        )

        report.print_title("Data Missing dataframe: Contiene los datos eliminados del dataset original.", 4)
        report.print_line()
        report.print_dataframe(df_dict["Data Missing"])

        report.print_title("Data Spurious dataframe: Contiene los datos sin sentido del dataset original.", 4)
        report.print_line()
        report.print_dataframe(df_dict["Data Spurious"])

        report.print_title("Data Skewness dataframe: Contiene los datos con alta dispersión del dataset original.", 4)
        report.print_line()
        report.print_dataframe(df_dict["Data Skewness"])

        report.print_title("Boxplot Feature Skewness > 2: Muestra la dispersión de los datos para las características con asimetría mayor que 2.", 4)
        report.print_line()
        report.print_boxplot(df_dict["Data"], features_list[0].tolist(), filename="boxplot_fskewness.png", img_title="Boxplot Feature Skewness")

        report.print_title("Boxplot Norm Features: Muestra la dispersión de los datos para las características normalizadas.", 4)
        report.print_line()
        report.print_boxplot(df_dict["Data"], features_list[1], filename="boxplot_normalized.png", img_title="Boxplot Normalized Feature")

        report.print_title("X dataframe: Contiene la matriz de características.", 4)
        report.print_line()
        report.print_dataframe(df_dict["X"])

        report.print_title("t dataframe: Contiene el vector de etiquetas.", 4)
        report.print_line()
        report.print_dataframe(df_dict["t"])
```

### 3.2. Resultados de la validación cruzada
<p style='text-align: justify;'>Se recopilan los resultados de todas las métricas de evaluación para entrenamiento y validación y se organizan para cada modelo óptimo:

```python
# Generating training report chapter
        report.print_title("Entrenamiento: Comparativa de modelos de aprendizaje automático", 3)
        report.print(
            """Se procederá a comparar los resultados obtenidos de diferentes modelos de aprendizaje automático
            variando tanto el tipo de modelo como los hiperparámetros de los que depende con el objetivo
            de obtener el mejor modelo que prediga el éxito o fracaso de las diferentes startups"""
        )

        report.print_title("Results dataframe: Muestra los resultados de los mejores modelos obtenidos", 4)
        report.print_line()
        report.print_dataframe(results)
```
<p style='text-align: justify;'> Además de los resultados, también se anotan los hiperparámetros que mejor puntuación han obtenido tanto para los modelos implementados en Scikit-Learn como para la red de neuronas profunda en Keras.

```python
report.print_title(
            "Hiperparámetros: Muestra el dataframe con los hiperparámetros usados en el entrenamiento", 4
        )
        report.print_line()
        for model in best_models:
            report.print_title("Hiperparámetros del modelo " + model, 5)
            report.print_dataframe(mdleval.get_hyperparams(best_models[model][1].steps[1][1], model))
        report.print_title("Hiperparámetros de la red neuronal", 5)
        dfparams, dfcomp = mdleval.get_hyperparams_DNN(best_DNN[0][0])
        report.print_dataframe(dfparams)
        report.print_dataframe(dfcomp)
```
<p style='text-align: justify;'>En base a los resultados anteriores, se pueden dibujar una serie de gráficas interpretables como la exactitud media de los modelos para cada fold.

```python
report.print_title("Exactitud media: Compara la exactitud de cada modelo en función de sus hiperparámetros", 4)
        report.print_line()
        for model in best_models:
            report.print_mean_acc_model(
                best_models,
                model,
                filename="mean_accuracy_" + model + ".png",
                img_title="Exactitud media de " + model,
            )
```
<p style='text-align: justify;'> Del mismo modo para la red neuronal se puede elaborar la curva de validación calculando tantas medias como orden tenga la k-fold empleada en la validación cruzada y valores como número de epochs o iteraciones se registren por cada entrenamiento.

```python
report.print_title(
            "Curva de validación: Compara los resultados del modelo en función de sus hiperparámetros", 4
        )
        report.print_line()
        report.print_val_curve_dnn(best_DNN)
```

### 3.3. Resultados de rendimiento y comparativa
Se generan diagramas de cajas para comparar la exactitud de los modelos.
```python        
report.print_title("Boxplot models: Muestra los valores de exactitud de los diferentes modelos", 4)
        report.print_line()
        report.print_boxplot(
            pd.DataFrame(results_data),
            results_labels,
            filename="boxplot_models_accuracy.png",
            img_title="Boxplot Models Accuracy",
            figsize=(10, 7),
            same_scale=True,
        )
```

<p style='text-align: justify;'>Y se presentan los resultados del contraste de hipótesis realizado para determinar el mejor modelo.

```python
report.print_title("Contraste de hipótesis: Comparación de modelos mediante el test de Kruskal-Wallis", 4)
        report.print_line()
        report.print_hpcontrast(list(results_data.values()), results_labels)
```

<p style='text-align: justify;'>También se incluyen en el report todos aquellos valores obtenidos en el estudio del rendimiento de los mejores modelos, esto es, matrices de confusión, curvas ROC y AUC e informes de clasificación.

<p style='text-align: justify;'>Para ello, se evaluan los modelos sobre el conjunto reservado de test. Estas funciones se encuentran definidas en el modulo:

[```models_evaluation.py```](https://github.com/davidadrianrg/startup-success-prediction/blob/master/models/models_evaluation.py), la documentación está en [```models_evaluation```](https://davidadrianrg.com/startup-success-prediction/util.html#module-models.models_evaluation).

```python
# To analize the models is needed to fit them using analize_performance_models function from models_evaluation module
    best_models, X_test, t_test, y_pred, y_score = mdleval.analize_performance_models(
        best_models, df_dict.get("X"), df_dict.get("t")
    )
    # Same function calling for the DNN models
    _, _, t_test_dnn, y_pred_dnn, y_pred_proba_dnn = mdleval.analize_performance_DNN(best_DNN)
```
<p style='text-align: justify;'>Con las predicciones extraídas de las funciones anteriores, se pueden calcular las matrices de confusión teniendo en cuenta las salidas binarizadas.

```python
report.print_title(
            "Matrices de confusión: Compara los valores reales con los valores predichos para cada modelo", 4
        )
        report.print_line()
        for model in best_models:
            report.print_confusion_matrix(
                best_models[model][1],
                X_test.values,
                t_test.values,
                filename="confusion_matrix_" + model + ".png",
                img_title="Matriz de confusión " + model,
                xlabel="Clase Predicha",
                ylabel="Clase Real",
            )
```
Y de la misma forma para la red de neuronas.

```python
report.print("\n")
        report.print_confusion_matrix_DNN(
            t_test_dnn, y_pred_dnn, xlabel="Clase Predicha", ylabel="Clase Real", title="Matriz de confusión DNN"
        )
```
<p style='text-align: justify;'>Obteniendo las probabilidades por clase se pueden dibujar las curvas ROC y AUC para cada modelo y la red neuronal.

```python
report.print_title("Curva ROC: Compara el ajuste entre la especificidad y la sensibilidad para cada modelo", 4)
        report.print_line()
        for model in best_models:
            report.print_roc_curve(
                t_test, y_score[model], filename="roc_curve_" + model + ".png", img_title="Curva ROC de " + model
            )
        report.print_roc_curve(t_test_dnn, y_pred_proba_dnn, filename="roc_curve_dnn.png", img_title="Curva ROC de DNN")
```

<p style='text-align: justify;'>Finalmente, también se incluyen los informes de los mejores clasificadores para las principales métricas de evaluación

```python
report.print_title("Informe de clasificación: Compara los resultados de cada modelo de clasificación", 4)
        report.print_line()
        for model in best_models:
            report.print_clreport(t_test, y_pred[model], title="Classification report for model " + model)
        report.print_clreport(t_test_dnn, y_pred_dnn, title="Classification report for model DNN")
```

<div id='id4' />

## 4. Conclusiones obtenidas
<p style='text-align: justify;'>En base a los resultados obtenidos pueden extraerse una serie de ideas que pueden servir de apoyo a futuros estudios en este término.

<p style='text-align: justify;'>En primer lugar, resulta de gran importancia no pasar por alto la necesidad de realizar un preprocesamiento del conjunto de datos. En el caso de este proyecto, existen una gran cantidad de datos perdidos que podrían interpretarse como que las startups no alcanzaron una fecha de cierre durante el estudio o no tuvieron lugar hitos históricos en la empresa. A pesar de que esto puede cobrar sentido, sigue siendo necesario realizar una imputación de los valores para poder tener en cuenta las muestras en el proceso de aprendizaje. En otros casos, puede resultar más interesante prescindir de aquellas variables que no aportan un grado de significación considerable de cara a la resolución del problema o bien, obtener mejores características por corelación como por ejemplo, los años de vida de la empresa teniendo en cuenta la fundación y el cierre o en su defecto el último año registrado en el estudio. Además, merece la pena mencionar la importancia de la recodificación de las variables categóricas a formato numérico de manera que sean interpretables y la normalización, principalmente de las que presentan una mayor asimetría estadística para facilitar el proceso de optimización. Analizando los diagramas de cajas del report se puede observar como tras aplicar una transformación logarítmica y normalización se obtienen distribuciones más centradas.

<p style='text-align: justify;'>Durante el proceso de obtención de los mejores modelos, resultaría de gran ayuda en términos de coste computacional, escoger un rango de hiperparámetros prometedor para obtener el modelo óptimo. Sin embargo, esto no es así, puesto que el orden y combinación de los mismos es de antemano desconocido y probablemente distinto para cada modelo. Por este motivo, una buena práctica sería realizar un barrido amplio inicialmente y después con el riego de incrementar el tiempo de compilación del programa, llevar a cabo una segunda búsqueda en un rango más reducido. Existen una seríe de métodos implementados en Scikit-Learn y Keras para este cometido, no obstante, en el proyecto se utiliza un función definida específicamente considerando la aleatoriedad del proceso de búsqueda.

<p style='text-align: justify;'>Observando los valores de los resultados tanto en entrenamiento como en validación para las mejores versiones de cada uno de los modelos, se aprecia como para cada k-fold, se tienen resultados aproximadamente parecidos, lo cual resulta lógico dado que se realizan distintos entrenamientos utilizando diferentes conjuntos de datos. Esto es apreciable en las curvas de exactitud media de los modelos implementados en Scikit-Learn. Por otra parte, la curva de validación para la red de neuronas profunda indica como evoluciona el proceso de aprendizaje y el grado de sobreajuste en su caso.

<p style='text-align: justify;'>Para analizar el rendimiento de los modelos se han definido una serie de métricas de evaluación durante el proceso de validación cruzada y además se estudian otras metodologías como las matrices de confusión, diagramas de bloques y curvas ROC-AUC por clase, todas ellas recogidas en el report, de forma automática al ejecutar el programa.

<p style='text-align: justify;'>En base a los resultados obtenidos tanto en la etapa de validación como en el análisis de rendimiento individual para cada modelo se pueden extraer ideas para escoger el modelo que mejor se adapta a las expectativas del problema. Por ejemplo, analizando las matrices de confusión. Partiendo como hipótesis nula, que las startups no van a tener éxito, puede resultar interesante minimizar el error de tipo I o nivel de significación, reduciendo los falsos positivos, de forma que no se pronostiquen erroneamente empresas que no tienen un buen pronóstico para invertir en ellas.

<p style='text-align: justify;'>Por otra parte, con el fin de realizar un estudio comparativo con detalle para aquellos modelos que a simple vista no presentan diferencias estaditicamente significativas, se realiza un contraste de hipótesis para determinar si se se pueden considerar iguales o no utilizando el método de Kruskal-Wallis, y en caso de no serlo, el análisis múltiple de Tukey. Una vez realizado este estudio, si los modelos son iguales, lo recomendable sería escoger el más sencillo para ahorrar coste computacional, si por el contrario no lo son, podría escogerse el que mejor responda a los intereses de problema como se mencionaba en el apartado anterior.






