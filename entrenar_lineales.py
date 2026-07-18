import pandas as pd
import time
import gc
import joblib
from memory_profiler import memory_usage
from IPython.display import display

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Carga de datos
input_path = 'AvancesTesisCronograma/data/train_val_set.csv'
df_train = pd.read_csv(input_path, sep=';')
df_train['texto_lineal'] = df_train['texto_lineal'].fillna('')

X = df_train[['texto_lineal']]
y = df_train['label']

print(f"Dataset de entrenamiento cargado: {len(X)} registros.")

# Configuración del Diseño Experimental (Actividad 1)
# 1. Validación Cruzada Estratificada K=5
cv_estratificado = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. Pipeline Dinámico: Submuestreo -> Vectorización TF-IDF
undersampler = RandomUnderSampler(random_state=42)
preprocessor = ColumnTransformer(
    transformers=[('tfidf', TfidfVectorizer(), 'texto_lineal')]
)

# Pipeline Base (el clasificador final se inserta dinámicamente)
pipeline_base = ImbPipeline(steps=[
    ('undersampler', undersampler),
    ('vectorizer', preprocessor),
    ('classifier', LogisticRegression()) # Placeholder
])

print("Diseño Experimental configurado: CV K=5 con balanceo y vectorización aislados por pliegue.")

# Funciones auxiliares de perfilado
def entrenar_con_metricas(grid_search_obj, X_data, y_data, nombre_modelo):
    print(f"\nIniciando entrenamiento: {nombre_modelo} (K=5)")
    
    def _fit():
        grid_search_obj.fit(X_data, y_data)

    t_inicio = time.time()
    muestras_ram = memory_usage(
        (_fit, [], {}), interval=0.1, include_children=True, max_usage=False, retval=False
    )
    t_fin = time.time()

    pico_ram_gb = max(muestras_ram) / 1024
    tiempo_min  = (t_fin - t_inicio) / 60
    best_score  = grid_search_obj.best_score_

    print(f"  Finalizado. Mejor F1-Score: {best_score:.4f} | Tiempo: {tiempo_min:.2f} min | RAM: {pico_ram_gb:.2f} GB")
    return best_score, tiempo_min, pico_ram_gb

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    # Malla de parámetros SVM
    from sklearn.base import clone

    param_grid_svm = [
        {
            'vectorizer__tfidf__max_features': [5000, 10000, 20000],
            'classifier': [LinearSVC(max_iter=2000, random_state=42)],
            'classifier__C': [0.1, 1.0, 10.0, 100.0]
        }
    ]

    grid_svm = GridSearchCV(
        estimator=clone(pipeline_base),
        param_grid=param_grid_svm,
        cv=cv_estratificado,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    f1_svm, t_svm, ram_svm = entrenar_con_metricas(grid_svm, X, y, "SVM - LinearSVC")
    gc.collect()

    # Malla de parámetros Regresión Logística
    param_grid_log = [
        {
            'vectorizer__tfidf__max_features': [5000, 10000, 20000],
            'classifier': [LogisticRegression(max_iter=1000, random_state=42)],
            'classifier__C': [0.1, 1.0, 10.0, 100.0]
        }
    ]

    grid_log = GridSearchCV(
        estimator=clone(pipeline_base),
        param_grid=param_grid_log,
        cv=cv_estratificado,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    f1_log, t_log, ram_log = entrenar_con_metricas(grid_log, X, y, "Regresión Logística")
    gc.collect()

    # Entregable: Guardar modelos optimizados
    joblib.dump(grid_svm.best_estimator_, 'AvancesTesisCronograma/data/modelo_optimizado_svm.joblib')
    joblib.dump(grid_log.best_estimator_, 'AvancesTesisCronograma/data/modelo_optimizado_lr.joblib')
    print("Modelos optimizados guardados en disco.")
