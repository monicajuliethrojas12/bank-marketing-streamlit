"""
Funciones de preparación, modelado y evaluación del proyecto integrador
(Bank Marketing, UCI Machine Learning Repository).

El módulo lo usan la aplicación Streamlit (app.py) y la sección 11 del
cuaderno de Colab. Así, ambos producen los mismos resultados con las mismas
especificaciones, semillas y reglas de preparación documentadas en el informe.
"""

import io

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Estructura del conjunto de datos
# ---------------------------------------------------------------------------
COLUMNAS = ["age", "job", "marital", "education", "default", "balance",
            "housing", "loan", "contact", "day", "month", "duration",
            "campaign", "pdays", "previous", "poutcome", "y"]
NUMERICAS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
CATEGORICAS = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "poutcome"]
# Columnas necesarias para puntuar clientes nuevos (sin duration ni y).
ENTRADAS_PREDICCION = [c for c in COLUMNAS if c not in ("duration", "y")]

DICCIONARIO = pd.DataFrame([
    ("age", "Edad del cliente", "Numérica entera", "Años", "Predictora"),
    ("job", "Ocupación del cliente", "Categórica (12 categorías)", "-", "Predictora"),
    ("marital", "Estado civil (divorced incluye viudez)", "Categórica (3)", "-", "Predictora"),
    ("education", "Nivel educativo", "Categórica (4)", "-", "Predictora"),
    ("default", "Tiene crédito en mora", "Binaria (yes/no)", "-", "Predictora; se excluye en la logística reducida"),
    ("balance", "Saldo medio anual de la cuenta", "Numérica entera", "Euros", "Respuesta de la regresión lineal; predictora en clasificación"),
    ("housing", "Tiene crédito de vivienda", "Binaria (yes/no)", "-", "Predictora"),
    ("loan", "Tiene préstamo personal", "Binaria (yes/no)", "-", "Predictora"),
    ("contact", "Medio del último contacto", "Categórica (3)", "-", "Predictora"),
    ("day", "Día del mes del último contacto", "Numérica entera (1-31)", "Día", "Predictora"),
    ("month", "Mes del último contacto", "Categórica (12)", "-", "Predictora"),
    ("duration", "Duración del último contacto", "Numérica entera", "Segundos", "Excluida: fuga de información (se conoce al terminar la llamada)"),
    ("campaign", "Contactos en la campaña actual, incluido el último", "Numérica entera", "Contactos", "Predictora"),
    ("pdays", "Días desde el último contacto de una campaña anterior (-1 = sin contacto previo)", "Numérica entera", "Días", "Predictora; en los árboles se divide en contactado_antes y pdays_real"),
    ("previous", "Contactos realizados antes de esta campaña", "Numérica entera", "Contactos", "Predictora"),
    ("poutcome", "Resultado de la campaña anterior", "Categórica (4)", "-", "Predictora"),
    ("y", "El cliente suscribió el depósito a plazo", "Binaria (yes/no)", "-", "Objetivo de clasificación"),
], columns=["Variable", "Significado", "Tipo", "Unidad", "Rol analítico"])

# ---------------------------------------------------------------------------
# Especificaciones de los modelos (secciones 7 a 10 del informe)
# ---------------------------------------------------------------------------
COSTO_FN = 10   # cliente interesado al que no se llama
COSTO_FP = 1    # llamada a un cliente que no contrata
UMBRALES = np.linspace(0.01, 0.99, 100)

VARIABLES_LINEAL = {
    "Completo": ["age", "job", "marital", "education", "default", "housing",
                 "loan", "contact", "day", "month", "campaign", "pdays",
                 "previous", "poutcome"],
    "Reducido": ["age", "job", "marital", "education", "default", "housing",
                 "loan", "contact", "month"],
}

VARIABLES_LOGISTICA = {
    "completa": ["age", "job", "marital", "education", "default", "balance",
                 "housing", "loan", "contact", "day", "month", "campaign",
                 "pdays", "previous", "poutcome"],
    "reducida": ["job", "marital", "education", "balance", "housing", "loan",
                 "contact", "day", "month", "campaign", "previous", "poutcome"],
}

ARBOL_CATEGORICAS = CATEGORICAS
ARBOL_NUMERICAS = ["age", "balance", "day", "campaign", "previous",
                   "pdays_real", "contactado_antes"]

LOGISTICA_REDUCIDA = "Regresión logística (reducida)"
LOGISTICA_COMPLETA = "Regresión logística (completa)"
ARBOL_PODADO = "Árbol de decisión (podado)"
ARBOL_LIBRE = "Árbol de decisión (sin límite de profundidad)"
RANDOM_FOREST = "Random Forest"

CLASIFICADORES = [LOGISTICA_REDUCIDA, LOGISTICA_COMPLETA, ARBOL_PODADO,
                  ARBOL_LIBRE, RANDOM_FOREST]
# Especificaciones recomendadas en las secciones 9 y 10.
RECOMENDADOS = [LOGISTICA_REDUCIDA, ARBOL_PODADO, RANDOM_FOREST]

DESCRIPCION_MODELOS = {
    LOGISTICA_REDUCIDA: "12 variables (sin age, default ni pdays, descartadas por razón de verosimilitud); "
                        "StandardScaler + OneHotEncoder(drop='first'); class_weight='balanced'.",
    LOGISTICA_COMPLETA: "15 variables (todas salvo duration y y); StandardScaler + "
                        "OneHotEncoder(drop='first'); class_weight='balanced'.",
    ARBOL_PODADO: "Gini, max_depth=10, min_samples_leaf=50, class_weight='balanced' "
                  "(elegido con validación cruzada y la regla de un error estándar).",
    ARBOL_LIBRE: "Gini sin límite de profundidad; sirve para mostrar el sobreajuste.",
    RANDOM_FOREST: "300 árboles, min_samples_leaf=5, sin límite de profundidad, "
                   "class_weight='balanced_subsample'.",
}

# Valores publicados en las secciones 9 y 10 del informe técnico.
VALORES_INFORME = [
    ("9.1", "Regresión lineal (completa)", 42, "R² prueba", 0.047, 3),
    ("9.1", "Regresión lineal (completa)", 42, "MAE prueba (euros)", 1487, 0),
    ("9.1", "Regresión lineal (completa)", 42, "RMSE prueba (euros)", 3056, 0),
    ("9.1", "Regresión lineal (reducida)", 42, "MAE prueba (euros)", 1489, 0),
    ("9.2", LOGISTICA_COMPLETA, 1234, "ROC-AUC", 0.774, 3),
    ("9.2", LOGISTICA_COMPLETA, 1234, "Umbral por costo", 0.446, 3),
    ("9.2", LOGISTICA_COMPLETA, 1234, "Costo total", 5580, 0),
    ("9.2", LOGISTICA_REDUCIDA, 1234, "ROC-AUC", 0.774, 3),
    ("9.2", LOGISTICA_REDUCIDA, 1234, "Umbral por costo", 0.485, 3),
    ("9.2", LOGISTICA_REDUCIDA, 1234, "Costo total", 5653, 0),
    ("10", ARBOL_PODADO, 42, "ROC-AUC", 0.778, 3),
    ("10", ARBOL_PODADO, 42, "PR-AUC", 0.418, 3),
    ("10", ARBOL_PODADO, 42, "Verdaderos positivos (umbral 0,5)", 629, 0),
    ("10", RANDOM_FOREST, 42, "ROC-AUC", 0.803, 3),
    ("10", RANDOM_FOREST, 42, "PR-AUC", 0.455, 3),
]


# ---------------------------------------------------------------------------
# Lectura y preparación
# ---------------------------------------------------------------------------
def leer_csv(fuente):
    """Lee un CSV separado por ';' o ','. Acepta una ruta, bytes o un archivo abierto."""
    if isinstance(fuente, (bytes, bytearray)):
        contenido = bytes(fuente)
    elif hasattr(fuente, "read"):
        contenido = fuente.read()
        if isinstance(contenido, str):
            contenido = contenido.encode("utf-8")
    else:
        with open(fuente, "rb") as archivo:
            contenido = archivo.read()
    if not contenido.strip():
        raise ValueError("El archivo está vacío.")

    lineas = contenido[:20000].decode("utf-8", errors="ignore").splitlines()
    encabezado = lineas[0] if lineas else ""
    separador = ";" if encabezado.count(";") >= encabezado.count(",") else ","
    try:
        datos = pd.read_csv(io.BytesIO(contenido), sep=separador, encoding="utf-8")
    except UnicodeDecodeError:
        datos = pd.read_csv(io.BytesIO(contenido), sep=separador, encoding="latin-1")
    except pd.errors.ParserError as error:
        raise ValueError(f"No fue posible leer el archivo como CSV: {error}") from error
    datos.columns = [str(c).strip().strip('"').lower() for c in datos.columns]
    return datos


def preparar_datos(datos, con_objetivo=True):
    """
    Aplica las reglas de preparación de la sección 3 y devuelve
    (datos preparados, registro de cambios). El índice conserva la posición
    de cada fila en el archivo original.
    """
    requeridas = COLUMNAS if con_objetivo else ENTRADAS_PREDICCION
    faltan = [c for c in requeridas if c not in datos.columns]
    if faltan:
        raise ValueError("Faltan columnas obligatorias: " + ", ".join(faltan))

    columnas = [c for c in COLUMNAS if c in datos.columns]
    prep = datos[columnas].copy()
    registro = []

    texto = [c for c in CATEGORICAS + ["y"] if c in prep.columns]
    espacios = 0
    for col in texto:
        serie = prep[col].astype(object)
        limpia = serie.map(lambda v: v.strip() if isinstance(v, str) else v)
        espacios += int((serie.ne(limpia) & serie.notna()).sum())
        prep[col] = limpia
    registro.append(("Variables de texto", f"Se retiraron espacios externos en {espacios} celdas."
                     if espacios else "Sin espacios externos que corregir."))

    no_numericos = 0
    for col in [c for c in NUMERICAS if c in prep.columns]:
        convertida = pd.to_numeric(prep[col], errors="coerce")
        no_numericos += int((convertida.isna() & prep[col].notna()).sum())
        prep[col] = convertida
    if no_numericos:
        registro.append(("Variables numéricas", f"{no_numericos} valores no numéricos se trataron como faltantes."))

    if "y" in prep.columns:
        prep["y"] = prep["y"].map(lambda v: v.lower() if isinstance(v, str) else v)
        invalidos = sorted(set(prep["y"].dropna().astype(str)) - {"yes", "no"})
        if invalidos:
            raise ValueError("La columna 'y' solo admite 'yes' y 'no'. Valores encontrados: "
                             + ", ".join(invalidos[:10]))
        sin_objetivo = int(prep["y"].isna().sum())
        if sin_objetivo:
            prep = prep[prep["y"].notna()]
            registro.append(("Variable objetivo", f"Se eliminaron {sin_objetivo} filas sin valor en 'y'."))

    duplicados = int(prep.duplicated().sum())
    if duplicados:
        prep = prep.drop_duplicates(keep="first")
    registro.append(("Duplicados", f"Se eliminaron {duplicados} filas duplicadas."
                     if duplicados else "No hay filas duplicadas."))

    presentes = prep.notna().mean(axis=1) * 100
    incompletas = int((presentes < 80).sum())
    if incompletas:
        prep = prep[presentes >= 80]
    registro.append(("Valores faltantes", f"Se eliminaron {incompletas} filas con menos del 80 % de datos."
                     if incompletas else "Ninguna fila tiene menos del 80 % de datos."))

    imputados = []
    for col in prep.columns:
        nulos = int(prep[col].isna().sum())
        if nulos and col != "y":
            if col in NUMERICAS:
                prep[col] = prep[col].fillna(prep[col].median())
                imputados.append(f"{col}: {nulos} con mediana")
            else:
                prep[col] = prep[col].fillna(prep[col].mode(dropna=True).iloc[0])
                imputados.append(f"{col}: {nulos} con moda")
    registro.append(("Imputación", "; ".join(imputados) if imputados else "No fue necesaria."))

    if len(prep) == 0:
        raise ValueError("No quedan filas válidas después de la preparación.")
    if con_objetivo and prep["y"].nunique() < 2:
        raise ValueError("La columna 'y' debe contener las dos clases ('yes' y 'no').")
    return prep, pd.DataFrame(registro, columns=["Etapa", "Resultado"])


def preparar_nuevos(datos):
    """
    Prepara un archivo de clientes por puntuar: limpia el texto, convierte las
    columnas numéricas y separa las filas incompletas, que no se puntúan.
    Devuelve (filas válidas, número de filas excluidas).
    """
    faltan = [c for c in ENTRADAS_PREDICCION if c not in datos.columns]
    if faltan:
        raise ValueError("Faltan columnas obligatorias: " + ", ".join(faltan))
    nuevos = datos[ENTRADAS_PREDICCION].copy()
    for col in CATEGORICAS:
        nuevos[col] = nuevos[col].astype(object).map(lambda v: v.strip() if isinstance(v, str) else v)
    for col in [c for c in NUMERICAS if c in nuevos.columns]:
        nuevos[col] = pd.to_numeric(nuevos[col], errors="coerce")
    completas = nuevos.notna().all(axis=1)
    if not completas.any():
        raise ValueError("Ninguna fila tiene completas las columnas obligatorias.")
    return nuevos[completas], int((~completas).sum())


def objetivo_binario(datos):
    return datos["y"].eq("yes").astype(int).to_numpy()


def variables_arbol(datos):
    """Divide pdays en contactado_antes (0/1) y pdays_real, como en la sección 10."""
    X = datos[ARBOL_CATEGORICAS + ["age", "balance", "day", "campaign", "previous"]].copy()
    X["pdays_real"] = datos["pdays"].where(datos["pdays"] != -1, 0)
    X["contactado_antes"] = (datos["pdays"] != -1).astype(int)
    return X[ARBOL_CATEGORICAS + ARBOL_NUMERICAS]


def variables_modelo(datos, nombre):
    if nombre == LOGISTICA_REDUCIDA:
        return datos[VARIABLES_LOGISTICA["reducida"]]
    if nombre == LOGISTICA_COMPLETA:
        return datos[VARIABLES_LOGISTICA["completa"]]
    return variables_arbol(datos)


def construir_clasificador(nombre, semilla):
    if nombre in (LOGISTICA_REDUCIDA, LOGISTICA_COMPLETA):
        variables = VARIABLES_LOGISTICA["reducida" if nombre == LOGISTICA_REDUCIDA else "completa"]
        num = [c for c in variables if c in NUMERICAS]
        cat = [c for c in variables if c in CATEGORICAS]
        transformador = ColumnTransformer([
            ("num_scaler", StandardScaler(), num),
            ("cat_encoder", OneHotEncoder(drop="first", handle_unknown="ignore"), cat),
        ], remainder="passthrough")
        modelo = LogisticRegression(class_weight="balanced", max_iter=1000,
                                    random_state=semilla, solver="lbfgs")
        return Pipeline([("transformer", transformador), ("classifier", modelo)])

    preparacion = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ARBOL_CATEGORICAS),
        ("num", "passthrough", ARBOL_NUMERICAS),
    ])
    if nombre == ARBOL_PODADO:
        modelo = DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_leaf=50,
                                        class_weight="balanced", random_state=semilla)
    elif nombre == ARBOL_LIBRE:
        modelo = DecisionTreeClassifier(criterion="gini", class_weight="balanced",
                                        random_state=semilla)
    elif nombre == RANDOM_FOREST:
        modelo = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=5,
                                        class_weight="balanced_subsample", oob_score=True,
                                        random_state=semilla, n_jobs=-1)
    else:
        raise ValueError(f"Modelo no reconocido: {nombre}")
    return Pipeline([("prep_arb", preparacion), ("modelo", modelo)])


def particion_estratificada(y, semilla=42, prop_prueba=0.2):
    """Índices de entrenamiento y prueba; coinciden con train_test_split(X, y, stratify=y)."""
    posiciones = np.arange(len(y))
    return train_test_split(posiciones, test_size=prop_prueba, stratify=y, random_state=semilla)


def entrenar_clasificador(datos, nombre, semilla=42, prop_prueba=0.2):
    """
    Entrena el clasificador con la partición estratificada indicada y devuelve
    las probabilidades en prueba y las probabilidades fuera de muestra del
    entrenamiento (validación cruzada de 5 grupos u out-of-bag en el Random
    Forest), que se usan para elegir el umbral sin mirar la prueba.
    """
    y = objetivo_binario(datos)
    X = variables_modelo(datos, nombre)
    idx_tr, idx_te = particion_estratificada(y, semilla, prop_prueba)
    modelo = construir_clasificador(nombre, semilla)
    modelo.fit(X.iloc[idx_tr], y[idx_tr])
    prob_prueba = modelo.predict_proba(X.iloc[idx_te])[:, 1]
    if nombre == RANDOM_FOREST:
        prob_seleccion = modelo.named_steps["modelo"].oob_decision_function_[:, 1]
        metodo = "probabilidades out-of-bag del entrenamiento"
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=semilla)
        prob_seleccion = cross_val_predict(clone(modelo), X.iloc[idx_tr], y[idx_tr],
                                           cv=cv, method="predict_proba")[:, 1]
        metodo = "validación cruzada estratificada de 5 grupos en entrenamiento"
    return {
        "nombre": nombre, "modelo": modelo, "semilla": semilla, "prop_prueba": prop_prueba,
        "idx_entrenamiento": idx_tr, "idx_prueba": idx_te,
        "id_prueba": datos.index.to_numpy()[idx_te],
        "y_entrenamiento": y[idx_tr], "y_prueba": y[idx_te],
        "prob_prueba": prob_prueba, "prob_seleccion": prob_seleccion,
        "metodo_umbral": metodo,
    }


# ---------------------------------------------------------------------------
# Umbral, métricas y tablas
# ---------------------------------------------------------------------------
def costos_por_umbral(y, prob, costo_fn=COSTO_FN, costo_fp=COSTO_FP, umbrales=UMBRALES):
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = prob[None, :] >= np.asarray(umbrales)[:, None]
    fp = (pred & (y == 0)).sum(axis=1)
    fn = (~pred & (y == 1)).sum(axis=1)
    return pd.DataFrame({"Umbral": umbrales, "Falsos positivos": fp,
                         "Falsos negativos": fn, "Costo total": costo_fn * fn + costo_fp * fp})


def umbral_por_costo(y, prob, costo_fn=COSTO_FN, costo_fp=COSTO_FP):
    """Umbral de la malla 0,01-0,99 que minimiza costo_fn*FN + costo_fp*FP (primer mínimo)."""
    tabla = costos_por_umbral(y, prob, costo_fn, costo_fp)
    return float(tabla.loc[tabla["Costo total"].idxmin(), "Umbral"])


def metricas_clasificacion(y, prob, umbral=0.5, costo_fn=COSTO_FN, costo_fp=COSTO_FP):
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= umbral).astype(int)
    vn, fp, fn, vp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())

    def division(a, b):
        return a / b if b else 0.0

    sensibilidad = division(vp, vp + fn)
    precision = division(vp, vp + fp)
    return {
        "Umbral": float(umbral), "VP": vp, "FP": fp, "FN": fn, "VN": vn,
        "Sensibilidad": sensibilidad,
        "Especificidad": division(vn, vn + fp),
        "Precisión": precision,
        "F1": division(2 * precision * sensibilidad, precision + sensibilidad),
        "Exactitud": division(vp + vn, len(y)),
        "ROC-AUC": float(roc_auc_score(y, prob)),
        "PR-AUC": float(average_precision_score(y, prob)),
        "Costo total": costo_fn * fn + costo_fp * fp,
    }


def tabla_ganancias(y, prob, grupos=10):
    """Ordena a los clientes por probabilidad y resume cada decil (1 = mayor probabilidad)."""
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    orden = np.argsort(-prob, kind="mergesort")
    total_pos = max(int(y.sum()), 1)
    tasa_base = y.mean()
    filas, acum_clientes, acum_pos = [], 0, 0
    for g, bloque in enumerate(np.array_split(orden, grupos), start=1):
        positivos = int(y[bloque].sum())
        acum_clientes += len(bloque)
        acum_pos += positivos
        filas.append({
            "Decil": g, "Clientes": len(bloque), "Contrataciones": positivos,
            "Tasa de contratación (%)": 100 * positivos / len(bloque),
            "Probabilidad mínima": float(prob[bloque].min()),
            "Clientes acumulados (%)": 100 * acum_clientes / len(y),
            "Contrataciones capturadas (%)": 100 * acum_pos / total_pos,
            "Lift acumulado": (acum_pos / acum_clientes) / tasa_base if tasa_base else np.nan,
        })
    return pd.DataFrame(filas)


def tabla_predicciones(datos, resultado, umbral):
    """Tabla de predicciones del grupo de prueba para un clasificador entrenado."""
    prueba = datos.iloc[resultado["idx_prueba"]]
    prob = resultado["prob_prueba"]
    pred = (prob >= umbral).astype(int)
    real = resultado["y_prueba"]
    tipo = np.select([(pred == 1) & (real == 1), (pred == 1) & (real == 0),
                      (pred == 0) & (real == 1)], ["VP", "FP", "FN"], default="VN")
    tabla = pd.DataFrame({
        "id_registro": prueba.index.to_numpy(),
        "y_real": real,
        "probabilidad": np.round(prob, 4),
        "umbral": round(float(umbral), 4),
        "prediccion": pred,
        "resultado": tipo,
    })
    for col in ["age", "job", "balance", "housing", "contact", "poutcome"]:
        tabla[col] = prueba[col].to_numpy()
    return tabla


def puntuar_clientes(resultado, nuevos, umbral):
    """Probabilidad, decisión y prioridad de llamada para clientes nuevos."""
    X = variables_modelo(nuevos, resultado["nombre"])
    prob = resultado["modelo"].predict_proba(X)[:, 1]
    salida = nuevos[[c for c in ENTRADAS_PREDICCION if c in nuevos.columns]].copy()
    salida.insert(0, "id_registro", nuevos.index.to_numpy())
    salida["probabilidad"] = np.round(prob, 4)
    salida["decision"] = np.where(prob >= umbral, "Llamar", "No llamar")
    orden = np.argsort(-prob, kind="mergesort")
    prioridad = np.empty(len(prob), dtype=int)
    prioridad[orden] = np.arange(1, len(prob) + 1)
    salida["prioridad"] = prioridad
    salida["decil"] = np.ceil(prioridad / len(prob) * 10).astype(int)
    primeras = ["id_registro", "prioridad", "decil", "probabilidad", "decision"]
    salida = salida[primeras + [c for c in salida.columns if c not in primeras]]
    return salida.sort_values("prioridad").reset_index(drop=True)


def categorias_desconocidas(resultado, nuevos):
    """Categorías de los datos nuevos que no aparecieron en el entrenamiento."""
    pasos = resultado["modelo"].named_steps
    transformador = pasos.get("transformer", pasos.get("prep_arb"))
    for nombre, codificador, columnas in transformador.transformers_:
        if isinstance(codificador, OneHotEncoder):
            avisos = {}
            for col, categorias in zip(columnas, codificador.categories_):
                nuevas = sorted(set(nuevos[col].astype(str)) - set(map(str, categorias)))
                if nuevas:
                    avisos[col] = nuevas
            return avisos
    return {}


# ---------------------------------------------------------------------------
# Regresión lineal y PCA
# ---------------------------------------------------------------------------
def entrenar_regresion(datos, especificacion="Reducido", semilla=42, prop_prueba=0.2):
    variables = VARIABLES_LINEAL[especificacion]
    X = datos[variables]
    y = datos["balance"].astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=prop_prueba, random_state=semilla)
    num = [c for c in variables if c in NUMERICAS]
    cat = [c for c in variables if c in CATEGORICAS]
    preprocesador = ColumnTransformer([
        ("num", Pipeline([("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))]), cat),
    ])
    modelo = Pipeline([("preprocessor", preprocesador), ("regressor", LinearRegression())])
    modelo.fit(X_tr, y_tr)
    pred_tr = modelo.predict(X_tr)
    pred_te = modelo.predict(X_te)
    nombres = [n.split("__", 1)[-1] for n in preprocesador.get_feature_names_out()]
    coeficientes = pd.DataFrame({"Variable": nombres,
                                 "Coeficiente": modelo.named_steps["regressor"].coef_})
    coeficientes = coeficientes.reindex(coeficientes["Coeficiente"].abs()
                                        .sort_values(ascending=False).index)
    metricas = {
        "Especificación": especificacion, "Variables": len(variables),
        "R² entrenamiento": r2_score(y_tr, pred_tr), "R² prueba": r2_score(y_te, pred_te),
        "MAE prueba (euros)": mean_absolute_error(y_te, pred_te),
        "RMSE prueba (euros)": float(np.sqrt(mean_squared_error(y_te, pred_te))),
        "MAE con la mediana (euros)": mean_absolute_error(y_te, np.full(len(y_te), y_tr.median())),
    }
    predicciones = pd.DataFrame({
        "id_registro": X_te.index.to_numpy(), "balance_real": y_te.to_numpy(),
        "balance_predicho": np.round(pred_te, 2), "residuo": np.round(y_te.to_numpy() - pred_te, 2),
    })
    return {"modelo": modelo, "metricas": metricas, "coeficientes": coeficientes,
            "intercepto": float(modelo.named_steps["regressor"].intercept_),
            "predicciones": predicciones}


VARIABLES_PCA = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]


def resultados_pca(datos, minimo_varianza=80):
    Z = StandardScaler().fit_transform(datos[VARIABLES_PCA])
    pca = PCA().fit(Z)
    etiquetas = [f"CP{i + 1}" for i in range(len(VARIABLES_PCA))]
    varianza = pd.DataFrame({
        "Componente": etiquetas,
        "Varianza explicada (%)": pca.explained_variance_ratio_ * 100,
        "Varianza acumulada (%)": np.cumsum(pca.explained_variance_ratio_) * 100,
    })
    retenidas = int(np.argmax(varianza["Varianza acumulada (%)"].to_numpy() >= minimo_varianza - 1e-9) + 1)
    cargas = pd.DataFrame(pca.components_.T, index=VARIABLES_PCA, columns=etiquetas)
    return {"varianza": varianza, "cargas": cargas, "retenidas": retenidas}


# ---------------------------------------------------------------------------
# Integración de resultados (sección 11)
# ---------------------------------------------------------------------------
def comparar_clasificadores(datos, nombres=None, semilla=42, prop_prueba=0.2,
                            costo_fn=COSTO_FN, costo_fp=COSTO_FP, resultados=None):
    """Evalúa varios clasificadores en la misma partición de prueba."""
    nombres = nombres or RECOMENDADOS
    resultados = resultados if resultados is not None else {}
    filas = []
    for nombre in nombres:
        if nombre not in resultados:
            resultados[nombre] = entrenar_clasificador(datos, nombre, semilla, prop_prueba)
        r = resultados[nombre]
        umbral = umbral_por_costo(r["y_entrenamiento"], r["prob_seleccion"], costo_fn, costo_fp)
        for regla, u in [("Umbral 0,5", 0.5), ("Umbral por costo", umbral)]:
            fila = {"Modelo": nombre, "Regla": regla}
            fila.update(metricas_clasificacion(r["y_prueba"], r["prob_prueba"], u, costo_fn, costo_fp))
            filas.append(fila)
    return pd.DataFrame(filas), resultados


def verificar_reproducibilidad(datos, resultados_previos=None):
    """Recalcula las cifras publicadas en las secciones 9 y 10 y las compara."""
    previos = resultados_previos or {}
    cache = {}

    def clasificador(nombre, semilla):
        clave = (nombre, semilla)
        if clave not in cache:
            previo = previos.get(nombre)
            if previo is not None and previo["semilla"] == semilla and previo["prop_prueba"] == 0.2:
                cache[clave] = previo
            else:
                cache[clave] = entrenar_clasificador(datos, nombre, semilla, 0.2)
        return cache[clave]

    lineal = {"Regresión lineal (completa)": entrenar_regresion(datos, "Completo", 42)["metricas"],
              "Regresión lineal (reducida)": entrenar_regresion(datos, "Reducido", 42)["metricas"]}
    filas = []
    for seccion, modelo, semilla, medida, publicado, decimales in VALORES_INFORME:
        if modelo in lineal:
            valor = lineal[modelo][medida]
        else:
            r = clasificador(modelo, semilla)
            if medida == "Umbral por costo":
                valor = umbral_por_costo(r["y_entrenamiento"], r["prob_seleccion"])
            elif medida == "Costo total":
                u = umbral_por_costo(r["y_entrenamiento"], r["prob_seleccion"])
                valor = metricas_clasificacion(r["y_prueba"], r["prob_prueba"], u)["Costo total"]
            elif medida.startswith("Verdaderos positivos"):
                valor = metricas_clasificacion(r["y_prueba"], r["prob_prueba"], 0.5)["VP"]
            else:
                valor = metricas_clasificacion(r["y_prueba"], r["prob_prueba"], 0.5)[medida]
        redondeado = round(float(valor), decimales)
        if redondeado == round(float(publicado), decimales):
            estado = "Coincide"
        elif abs(valor - publicado) <= 0.01 * abs(publicado):
            estado = "Diferencia menor al 1 %"
        else:
            estado = "No coincide"
        filas.append({"Sección": seccion, "Modelo": modelo, "Semilla": semilla, "Medida": medida,
                      "Valor del informe": publicado, "Valor reproducido": redondeado, "Estado": estado})
    return pd.DataFrame(filas)


def exportar_csv(tabla):
    """CSV para Excel en español: separador ';', decimal ',' y codificación UTF-8 con BOM."""
    return tabla.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
