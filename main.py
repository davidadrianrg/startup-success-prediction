"""Main file of the program to train a model for startup success prediction."""

# Importing required modules
import numpy as np
import pandas as pd
from postprocessing.report import Report
from preprocessing import preprocessing as prp
from models import models_evaluation as mdleval

def make_preprocessing(filepath: str) -> pd.DataFrame:
    """Clean and prepare the given dataframe to train ML models."""
    # Load a dataframe to preprocess from the given filepath
    data = prp.read_dataset(filepath)

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

    # Recode non numerical columns to numerical ones
    non_numerical = "state_code"
    data = prp.non_numerical_recoding(data, non_numerical)

    # Normalize quantitative data
    data, data_skewness, features, norm_features = prp.data_normalization(data)

    # Get X and t from the dataset using the column tags
    x_tags = [
        "state_code",
        "age_last_funding_year",
        "age_first_milestone_year",
        "age_last_milestone_year",
        "funding_rounds",
        "milestones",
        "is_CA",
        "is_NY",
        "is_MA",
        "is_TX",
        "is_otherstate",
        "is_software",
        "is_web",
        "is_mobile",
        "is_enterprise",
        "is_advertising",
        "is_gamesvideo",
        "is_ecommerce",
        "is_biotech",
        "is_consulting",
        "is_othercategory",
        "has_VC",
        "has_angel",
        "has_roundA",
        "has_roundB",
        "has_roundC",
        "has_roundD",
        "avg_participants",
        "is_top500",
        "age",
        "norm_funding_total_usd",
        "norm_age_first_funding_year",
        "norm_relationships",
    ]
    label_tag = "labels"
    X, t = prp.split_X_t(data, x_tags, label_tag)

    # Make a dictionary with the dataframes to be returned

    dataframes_dict = {
        "X": X,
        "t": t.to_frame(),
        "Data Missing": data_missing,
        "Data Spurious": data_spurious,
        "Data Skewness": data_skewness,
        "Data": data
    }

    # Make a features list to be returned

    features_list = [features, norm_features]

    return dataframes_dict, features_list


def train_models(X: np.ndarray, t: np.ndarray):
    """Train different models using the dataset and its labels.

    :param X: Input values of the dataset
    :type X: numpy.ndarray
    :param t: Label values for the exit of the dataset
    :type t: numpy.ndarray
    """
    # Wrapper function of optimize_models and optimize_DNN functions in hyperparameters modules

    # Ask user for the training trials, epochs and folds
    trials = int(input("Introduce el número de intentos aleatorios en la generación de los modelos: "))
    epochs = int(input("Introduce el número de iteraciones máximo: "))
    cv = int(input("Introduce el número de folds para la validación cruzada: "))
    # Return a tuple with a dict with the best models validated and the train size and the best DNN model
    # Using the hyperperameters ranges given in the arguments
    best_models, best_DNN = mdleval.get_best_models(X, t, cv=cv, trials=trials, epochs=epochs)

    # Return a dataframe with validation results of the models in visutalization mode.
    results = mdleval.get_results(best_models, best_DNN)

    return results, best_models, best_DNN


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
    
    # To analize the models is needed to fit them using analize_performance_models function from models_evaluation module
    best_models, X_test, t_test, y_pred, y_score = mdleval.analize_performance_models(best_models, df_dict.get("X"), df_dict.get("t"))
    # Same function calling for the DNN models
    results_DNN, X_test_dnn, t_test_dnn, y_pred_dnn, y_pred_proba_dnn = mdleval.analize_performance_DNN(best_DNN)


    with Report(generate_pdf=True) as report:
        # Generating the header
        report.print_title("Startup Success Prediction Model")
        report.print_title("David Adrián Rodríguez García & Víctor Caínzos López", 2)
        report.print_line()

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

        report.print_title("Boxplot models: Muestra los valores de exactitud de los diferentes modelos", 4)
        report.print_line()
        report.print_boxplot(pd.DataFrame(results_data), results_labels,filename="boxplot_models_accuracy.png" , img_title="Boxplot Models Accuracy",figsize= (10,7), same_scale=True)

        report.print_title("Contraste de hipótesis: Comparación de modelos mediante el test de Kruskal-Wallis", 4)
        report.print_line()
        report.print_hpcontrast(list(results_data.values()), results_labels)

        report.print_title("Matrices de confusión: Compara los valores reales con los valores predichos para cada modelo", 4)
        report.print_line()
        for model in best_models:
            report.print_confusion_matrix(best_models[model][1], X_test.values, t_test.values, filename="confusion_matrix_" + model + ".png", img_title="Matriz de confusión " + model, xlabel="Clase Predicha", ylabel="Clase Real")
        report.print("\n")
        report.print_confusion_matrix_DNN(t_test_dnn, y_pred_dnn, xlabel="Clase Predicha", ylabel="Clase Real", title="Matriz de confusión DNN")

        report.print_title("Curva ROC: Compara el ajuste entre la especificidad y la sensibilidad para cada modelo", 4)
        report.print_line()
        for model in best_models:
            report.print_roc_curve(t_test, y_score[model], filename="roc_curve_" + model + ".png", img_title="Curva ROC de " + model)
        report.print_roc_curve(t_test_dnn,y_pred_proba_dnn, filename="roc_curve_dnn.png", img_title="Curva ROC de DNN")
        
        report.print_title("Informe de clasificación: Compara los resultados de cada modelo de clasificación", 4)
        report.print_line()
        for model in best_models:
            report.print_clreport(t_test, y_pred[model], title="Classification report for model " + model)
        report.print_clreport(t_test_dnn, y_pred_dnn, title="Classification report for model DNN")

        report.print_title("Exactitud media: Compara la exactitud de cada modelo en función de sus hiperparámetros", 4)
        report.print_line()
        for model in best_models:
            report.print_val_curve_model(best_models, model, filename="validation_curve_" + model + ".png", img_title="Curva de validación de " + model)
        report.print_title("Curva de validación: Compara los resultados del modelo en función de sus hiperparámetros", 4)
        report.print_line()
        report.print_val_curve_dnn(best_DNN)

        report.print_title("Hiperparámetros: Muestra el dataframe con los hiperparámetros usados en el entrenamiento", 4)
        report.print_line()
        for model in best_models:
            report.print_title("Hiperparámetros del modelo " + model, 5)
            report.print_dataframe(mdleval.get_hyperparams(best_models[model][1].steps[1][1], model))
        report.print_title("Hiperparámetros de la red neuronal", 5)
        dfparams, dfcomp = mdleval.get_hyperparams_DNN(best_DNN[0][0])
        report.print_dataframe(dfparams)
        report.print_dataframe(dfcomp)


if __name__ == "__main__":

    # Loading and preprocessing the startup dataframe
    df_dict, features_list = make_preprocessing("data/startup_data.csv")

    # Training different models using the previous dataset
    results, best_models, best_DNN = train_models(df_dict.get("X").values, df_dict.get("t").values)

    # Generating the report with the results of the training
    make_report(df_dict, features_list, results, best_models, best_DNN)
