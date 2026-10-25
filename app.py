"""
Aplicación Streamlit del proyecto integrador de Analítica de Datos 2.
Conjunto de datos: Bank Marketing (UCI), archivo bank-full.csv.

Ejecución local:  streamlit run app.py
"""

import hashlib
import io
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve
from sklearn.tree import export_text, plot_tree

import modelos as mdl

st.set_page_config(page_title="Bank Marketing | Analítica 2", layout="wide")

RUTA_DATOS = Path(__file__).resolve().parent / "data" / "bank-full.csv"

# Paleta y estilo de los gráficos
AZUL, NARANJA, AQUA, ROJO = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
TINTA, TINTA_2, GRIS = "#0b0b0b", "#52514e", "#898781"
REJILLA, EJE = "#e1e0d9", "#c3c2b7"
COLOR_MODELO = {mdl.LOGISTICA_REDUCIDA: AZUL, mdl.ARBOL_PODADO: NARANJA, mdl.RANDOM_FOREST: AQUA}
SECUENCIAL = LinearSegmentedColormap.from_list(
    "azules", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGENTE = LinearSegmentedColormap.from_list("divergente", [AZUL, "#f0efec", ROJO])
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": EJE, "axes.labelcolor": TINTA_2, "axes.titlecolor": TINTA,
    "axes.titlesize": 11, "axes.labelsize": 10, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": REJILLA,
    "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False, "font.size": 10,
})

MODULOS = [
    "Inicio",
    "Carga y análisis de datos",
    "Análisis de componentes principales",
    "Agrupamiento K-means",
    "Regresión lineal múltiple",
    "Modelos de clasificación",
    "Comparación y priorización",
    "Predicción de nuevos clientes",
    "Reproducibilidad y despliegue",
]
COLORES_CLUSTER = [AZUL, NARANJA, AQUA, "#eda100", "#e87ba4", "#008300", "#6250d6", ROJO]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def num(valor, decimales=3):
    """Formato numérico en español: punto de miles y coma decimal."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return ""
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


def pct(valor, decimales=1):
    return f"{num(100 * valor, decimales)} %"


def mostrar_tabla(tabla, decimales=3, sin_miles=()):
    formatos = {}
    for col in tabla.columns:
        if pd.api.types.is_bool_dtype(tabla[col]):
            continue
        if pd.api.types.is_float_dtype(tabla[col]):
            formatos[col] = lambda v, d=decimales: num(v, d)
        elif pd.api.types.is_integer_dtype(tabla[col]) and col not in sin_miles:
            formatos[col] = lambda v: num(v, 0)
    st.dataframe(tabla.style.format(formatos), hide_index=True)


def boton_descarga(tabla, nombre_archivo, etiqueta, clave):
    st.download_button(etiqueta, data=mdl.exportar_csv(tabla), file_name=nombre_archivo,
                       mime="text/csv", key=clave)


def mostrar_figura(fig):
    """Muestra la figura a su tamaño natural (sin estirarla al ancho de la página)."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    ancho = int(fig.get_figwidth() * 92)
    plt.close(fig)
    st.image(buffer.getvalue(), width=ancho)


def nombres_limpios(nombres):
    return [str(n).split("__", 1)[-1] for n in nombres]


def nombre_corto(nombre):
    return (nombre.replace("Regresión logística", "logistica").replace("Árbol de decisión", "arbol")
            .replace("Random Forest", "random_forest").replace(" (", "_").replace(")", "")
            .replace(" ", "_").lower())


# ---------------------------------------------------------------------------
# Datos y modelos en caché
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar_datos(contenido):
    original = mdl.leer_csv(contenido)
    preparado, registro = mdl.preparar_datos(original)
    return original, preparado, registro


@st.cache_resource(show_spinner=False, max_entries=12)
def clasificador_entrenado(clave, nombre, semilla, prop_prueba, _datos):
    return mdl.entrenar_clasificador(_datos, nombre, semilla, prop_prueba)


@st.cache_resource(show_spinner=False, max_entries=12)
def regresion_entrenada(clave, especificacion, semilla, prop_prueba, _datos):
    return mdl.entrenar_regresion(_datos, especificacion, semilla, prop_prueba)


@st.cache_data(show_spinner=False)
def pca_calculado(clave, minimo, _datos):
    return mdl.resultados_pca(_datos, minimo)


@st.cache_data(show_spinner=False)
def curva_k_calculada(clave, _datos):
    return mdl.curva_k(_datos)


@st.cache_resource(show_spinner=False, max_entries=8)
def kmeans_calculado(clave, k, _datos):
    return mdl.kmeans_final(_datos, k)


@st.cache_resource(show_spinner=False, max_entries=8)
def sensibilidad_calculada(clave, k, _datos, _etiquetas):
    return mdl.sensibilidad_kmeans(_datos, _etiquetas, k)


@st.cache_resource(show_spinner=False, max_entries=8)
def estabilidad_calculada(clave, k, remuestreos, _datos, _etiquetas):
    return mdl.estabilidad_kmeans(_datos, _etiquetas, k, remuestreos=remuestreos)


def obtener_clasificador(nombre):
    with st.spinner(f"Entrenando {nombre}..."):
        return clasificador_entrenado(CLAVE, nombre, SEMILLA, PROP_PRUEBA, DATOS)


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def fig_histograma(datos, variable, recortar, por_respuesta):
    serie = datos[variable]
    vista = datos
    if recortar:
        p1, p99 = serie.quantile([0.01, 0.99])
        vista = datos[serie.between(p1, p99)]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    if por_respuesta:
        sns.histplot(data=vista, x=variable, hue="y", hue_order=["no", "yes"], bins=35,
                     stat="percent", common_norm=False, element="step", fill=False,
                     palette={"no": AZUL, "yes": NARANJA}, linewidth=2, ax=ax)
        ax.set_ylabel("Clientes de cada grupo (%)")
    else:
        sns.histplot(vista[variable], bins=35, color=AZUL, edgecolor="white", linewidth=0.6, ax=ax)
        ax.set_ylabel("Clientes")
    ax.set_xlabel(variable)
    titulo = f"Distribución de {variable}"
    ax.set_title(titulo + (" (vista P1-P99)" if recortar else ""), loc="left")
    fig.tight_layout()
    return fig


def fig_categoria(datos, variable):
    tabla = (datos.assign(contrato=datos["y"].eq("yes").astype(int))
             .groupby(variable)["contrato"].agg(["size", "mean"]).sort_values("mean"))
    fig, ejes = plt.subplots(1, 2, figsize=(11, max(2.6, 0.38 * len(tabla) + 1.2)), sharey=True)
    ejes[0].barh(tabla.index.astype(str), tabla["size"], color=AZUL, height=0.65)
    ejes[0].set(title="Clientes por categoría", xlabel="Clientes")
    ejes[1].barh(tabla.index.astype(str), tabla["mean"] * 100, color=AZUL, height=0.65)
    tasa = datos["y"].eq("yes").mean() * 100
    ejes[1].axvline(tasa, color=TINTA_2, linestyle="--", linewidth=1.2, label=f"Tasa global ({num(tasa, 1)} %)")
    ejes[1].legend(loc="lower right", fontsize=8)
    ejes[1].set(title="Tasa de contratación (%)", xlabel="Clientes que contratan (%)")
    for valor, posicion in zip(tabla["mean"] * 100, range(len(tabla))):
        ejes[1].text(valor + 0.4, posicion, num(valor, 1), va="center", fontsize=8, color=TINTA_2)
    for ax in ejes:
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def fig_correlacion(datos):
    matriz = datos[mdl.NUMERICAS].assign(contrato=datos["y"].eq("yes").astype(int)).corr()
    fig, ax = plt.subplots(figsize=(7, 5.4))
    sns.heatmap(matriz, annot=True, fmt=".2f", cmap=DIVERGENTE, vmin=-1, vmax=1, linewidths=2,
                linecolor="white", cbar_kws={"label": "Correlación de Pearson"}, ax=ax)
    ax.set_title("Matriz de correlación (variables numéricas y contratación 0/1)", loc="left")
    ax.grid(False)
    fig.tight_layout()
    return fig


def fig_varianza_pca(varianza, minimo):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(1, len(varianza) + 1)
    ax.bar(x, varianza["Varianza explicada (%)"], color="#86b6ef", width=0.6, label="Varianza explicada")
    ax.plot(x, varianza["Varianza acumulada (%)"], marker="o", markersize=8, color=AZUL, label="Varianza acumulada")
    ax.axhline(minimo, color=TINTA_2, linestyle="--", linewidth=1.2, label=f"Criterio del {minimo} %")
    ax.set(xticks=x, xticklabels=varianza["Componente"], ylabel="Varianza (%)", ylim=(0, 105),
           title="Varianza explicada por componente")
    ax.legend(loc="center right")
    fig.tight_layout()
    return fig


def fig_cargas(cargas):
    fig, ax = plt.subplots(figsize=(1.3 * cargas.shape[1] + 2.5, 4.2))
    sns.heatmap(cargas, annot=True, fmt=".2f", cmap=DIVERGENTE, vmin=-1, vmax=1, linewidths=2,
                linecolor="white", cbar_kws={"label": "Carga"}, ax=ax)
    ax.set_title("Cargas de las componentes retenidas", loc="left")
    ax.grid(False)
    fig.tight_layout()
    return fig


def fig_curva_k(tabla, k_elegido):
    fig, ejes = plt.subplots(1, 2, figsize=(12, 3.9))
    ejes[0].plot(tabla["k"], tabla["Inercia"], marker="o", markersize=8, color=AZUL)
    ejes[0].set(title="Método del codo (Elbow)", xlabel="Número de clústeres (k)", ylabel="Inercia (WCSS)")
    ejes[1].plot(tabla["k"], tabla["Silhouette"], marker="o", markersize=8, color=AQUA)
    ejes[1].set(title="Silhouette promedio", xlabel="Número de clústeres (k)", ylabel="Silhouette")
    for ax in ejes:
        ax.axvline(k_elegido, color=NARANJA, linestyle="--", linewidth=1.5, label=f"k elegido = {k_elegido}")
        ax.set_xticks(tabla["k"])
        ax.legend(loc="best")
    fig.tight_layout()
    return fig


def fig_silueta_obs(sil, lab, k):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    y0 = 0
    for c in range(k):
        v = np.sort(sil[lab == c])
        ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, color=COLORES_CLUSTER[c % 8], alpha=0.85, linewidth=0)
        ax.text(-0.06, y0 + len(v) / 2, str(c), va="center", fontsize=9, color=TINTA_2)
        y0 += len(v) + 40
    ax.axvline(sil.mean(), color=ROJO, linestyle="--", linewidth=1.5, label=f"Promedio = {num(sil.mean(), 3)}")
    ax.set(title="Silhouette por observación y clúster (submuestra de 5.000)", xlabel="Silhouette", yticks=[])
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_tamanos(tamanos):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    colores = [COLORES_CLUSTER[int(c) % 8] for c in tamanos["Clúster"]]
    ax.bar(tamanos["Clúster"].astype(str), tamanos["Clientes"], color=colores, width=0.6)
    for i, (n_, p_) in enumerate(zip(tamanos["Clientes"], tamanos["Porcentaje (%)"])):
        ax.text(i, n_, f"{num(n_, 0)}\n({num(p_, 1)} %)", ha="center", va="bottom", fontsize=8, color=TINTA_2)
    ax.set(title="Tamaño de cada clúster", xlabel="Clúster", ylabel="Clientes", ylim=(0, tamanos["Clientes"].max() * 1.25))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def fig_perfiles(indice, dif_std):
    fig, ejes = plt.subplots(1, 2, figsize=(12, 3.9))
    limite = max(abs(indice.values).max(), 1)
    sns.heatmap(indice.T, annot=True, fmt=".0f", cmap=DIVERGENTE, vmin=-limite, vmax=limite, linewidths=2,
                linecolor="white", cbar_kws={"label": "% frente a la media global"}, ax=ejes[0])
    ejes[0].set_title("Desviación porcentual respecto a la media global", loc="left")
    sns.heatmap(dif_std.T, annot=True, fmt=".2f", cmap=DIVERGENTE, vmin=-1.5, vmax=1.5, linewidths=2,
                linecolor="white", cbar_kws={"label": "Desviaciones estándar"}, ax=ejes[1])
    ejes[1].set_title("Diferencia estandarizada", loc="left")
    for ax in ejes:
        ax.grid(False)
        ax.set_xlabel("Clúster")
    fig.tight_layout()
    return fig


def fig_real_predicho(pred):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(pred["balance_real"], pred["balance_predicho"], s=10, alpha=0.35, color=AZUL, edgecolors="none")
    limites = [min(pred["balance_real"].min(), pred["balance_predicho"].min()),
               max(pred["balance_real"].max(), pred["balance_predicho"].max())]
    ax.plot(limites, limites, color=ROJO, linestyle="--", linewidth=1.5, label="Predicción perfecta (y = x)")
    ax.set(xlabel="Saldo real (euros)", ylabel="Saldo predicho (euros)", title="Valores reales frente a predichos")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def fig_residuos(pred):
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.2))
    ejes[0].scatter(pred["balance_predicho"], pred["residuo"], s=10, alpha=0.35, color=AZUL, edgecolors="none")
    ejes[0].axhline(0, color=ROJO, linestyle="--", linewidth=1.5)
    ejes[0].set(xlabel="Saldo predicho (euros)", ylabel="Residuo (euros)", title="Residuos frente a valores predichos")
    sns.histplot(pred["residuo"], bins=60, color=AZUL, edgecolor="white", linewidth=0.5, ax=ejes[1])
    ejes[1].axvline(0, color=ROJO, linestyle="--", linewidth=1.5)
    ejes[1].set(xlabel="Residuo (euros)", ylabel="Clientes", title="Distribución de los residuos")
    fig.tight_layout()
    return fig


def fig_barras_signo(tabla, columna, titulo, etiqueta_x, referencia=0.0):
    tabla = tabla.iloc[::-1]
    colores = [AZUL if v >= referencia else ROJO for v in tabla[columna]]
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(tabla) + 1.4))
    ax.barh(tabla["Variable"], tabla[columna] - referencia, left=referencia, color=colores, height=0.65)
    ax.axvline(referencia, color=EJE, linewidth=1)
    if referencia == 1.0:  # razones de odds: escala logarítmica, simétrica alrededor de 1
        ax.set_xscale("log")
    ax.set(title=titulo, xlabel=etiqueta_x)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def fig_matriz_confusion(m):
    matriz = np.array([[m["VN"], m["FP"]], [m["FN"], m["VP"]]])
    nombres = np.array([["Verdaderos negativos", "Falsos positivos"], ["Falsos negativos", "Verdaderos positivos"]])
    total = matriz.sum()
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.imshow(matriz, cmap=SECUENCIAL)
    for i in range(2):
        for j in range(2):
            color = "white" if matriz[i, j] > matriz.max() * 0.55 else TINTA
            ax.text(j, i, f"{nombres[i, j]}\n{num(matriz[i, j], 0)}\n{num(100 * matriz[i, j] / total, 1)} %",
                    ha="center", va="center", color=color, fontsize=9)
    ax.set_xticks([0, 1], ["No contrata (0)", "Contrata (1)"])
    ax.set_yticks([0, 1], ["No contrata (0)", "Contrata (1)"])
    ax.set(xlabel="Predicción del modelo", ylabel="Valor real", title=f"Matriz de confusión (umbral {num(m['Umbral'], 3)})")
    ax.grid(False)
    fig.tight_layout()
    return fig


def fig_roc(y, prob, m):
    fpr, tpr, _ = roc_curve(y, prob)
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.plot(fpr, tpr, color=AZUL, label=f"Modelo (AUC = {num(m['ROC-AUC'], 3)})")
    ax.plot([0, 1], [0, 1], color=GRIS, linestyle="--", linewidth=1.2, label="Clasificación al azar (AUC = 0,5)")
    ax.plot(1 - m["Especificidad"], m["Sensibilidad"], "o", markersize=9, color=NARANJA,
            markeredgecolor="white", markeredgewidth=2, label="Punto de operación")
    ax.set(xlabel="Tasa de falsos positivos (1 - especificidad)", ylabel="Sensibilidad",
           title="Curva ROC", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_pr(y, prob, m):
    precision, recall, _ = precision_recall_curve(y, prob)
    prevalencia = float(np.mean(y))
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.plot(recall, precision, color=AZUL, label=f"Modelo (PR-AUC = {num(m['PR-AUC'], 3)})")
    ax.axhline(prevalencia, color=GRIS, linestyle="--", linewidth=1.2, label=f"Prevalencia ({pct(prevalencia)})")
    ax.plot(m["Sensibilidad"], m["Precisión"], "o", markersize=9, color=NARANJA,
            markeredgecolor="white", markeredgewidth=2, label="Punto de operación")
    ax.set(xlabel="Sensibilidad (recall)", ylabel="Precisión", title="Curva precisión-sensibilidad",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_costos(tabla, umbral):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(tabla["Umbral"], tabla["Costo total"], color=AZUL, label="Costo total en entrenamiento")
    ax.axvline(umbral, color=NARANJA, linestyle="--", linewidth=1.5, label=f"Umbral por costo = {num(umbral, 3)}")
    ax.axvline(0.5, color=GRIS, linestyle=":", linewidth=1.5, label="Umbral 0,5")
    ax.set(xlabel="Umbral de decisión", ylabel="Costo total", title="Costo del error según el umbral")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def fig_comparacion(resultados):
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.6))
    for nombre, r in resultados.items():
        color = COLOR_MODELO.get(nombre, AZUL)
        fpr, tpr, _ = roc_curve(r["y_prueba"], r["prob_prueba"])
        prec, rec, _ = precision_recall_curve(r["y_prueba"], r["prob_prueba"])
        m = mdl.metricas_clasificacion(r["y_prueba"], r["prob_prueba"])
        ejes[0].plot(fpr, tpr, color=color, label=f"{nombre} (AUC = {num(m['ROC-AUC'], 3)})")
        ejes[1].plot(rec, prec, color=color, label=f"{nombre} (PR-AUC = {num(m['PR-AUC'], 3)})")
    ejes[0].plot([0, 1], [0, 1], color=GRIS, linestyle="--", linewidth=1.2)
    prevalencia = float(np.mean(next(iter(resultados.values()))["y_prueba"]))
    ejes[1].axhline(prevalencia, color=GRIS, linestyle="--", linewidth=1.2, label=f"Prevalencia ({pct(prevalencia)})")
    ejes[0].set(xlabel="Tasa de falsos positivos", ylabel="Sensibilidad", title="Curvas ROC", xlim=(0, 1), ylim=(0, 1.02))
    ejes[1].set(xlabel="Sensibilidad (recall)", ylabel="Precisión", title="Curvas precisión-sensibilidad",
                xlim=(0, 1), ylim=(0, 1.02))
    ejes[0].legend(loc="lower right", fontsize=8)
    ejes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_ganancias(y, prob, nombre, porcentaje):
    y = np.asarray(y)
    orden = np.argsort(-np.asarray(prob), kind="mergesort")
    capturadas = np.concatenate([[0], np.cumsum(y[orden]) / y.sum() * 100])
    clientes = np.linspace(0, 100, len(capturadas))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(clientes, capturadas, color=COLOR_MODELO.get(nombre, AZUL), label=nombre)
    ax.plot([0, 100], [0, 100], color=GRIS, linestyle="--", linewidth=1.2, label="Selección al azar")
    k = int(round(porcentaje / 100 * len(y)))
    ax.plot(porcentaje, capturadas[k], "o", markersize=9, color=NARANJA, markeredgecolor="white", markeredgewidth=2)
    ax.annotate(f"{num(porcentaje, 0)} % de clientes\n{num(capturadas[k], 1)} % de contrataciones",
                (porcentaje, capturadas[k]), xytext=(12, -34), textcoords="offset points", fontsize=8, color=TINTA_2)
    ax.set(xlabel="Clientes contactados, ordenados por probabilidad (%)", ylabel="Contrataciones capturadas (%)",
           title="Curva de ganancia acumulada", xlim=(0, 100), ylim=(0, 102))
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Barra lateral: navegación, carga del archivo y parámetros
# ---------------------------------------------------------------------------
st.sidebar.title("Bank Marketing")
st.sidebar.caption("Proyecto integrador · Analítica de Datos 2 · UIS")
MODULO = st.sidebar.radio("Módulo", MODULOS, key="modulo")
st.sidebar.divider()
archivo = st.sidebar.file_uploader(
    "Cargar dataset (CSV)", type=["csv"],
    help="Archivo con la estructura de bank-full.csv, separado por ';' o ','. "
         "Si no se carga ninguno, se usa data/bank-full.csv del repositorio.")
st.sidebar.subheader("Validación")
SEMILLA = int(st.sidebar.number_input("Semilla de la partición", min_value=0, max_value=1_000_000, value=42, step=1,
                                      help="42 reproduce las secciones 7 y 10; 1234 reproduce las secciones 8 y 9.2."))
PROP_PRUEBA = round(float(st.sidebar.select_slider("Proporción de prueba", options=[0.15, 0.20, 0.25, 0.30], value=0.20,
                                                   format_func=lambda p: f"{int(round(p * 100))} %")), 2)
st.sidebar.subheader("Costos del error")
COSTO_FN = int(st.sidebar.number_input("Costo de un falso negativo", min_value=0, value=mdl.COSTO_FN, step=1,
                                       help="Cliente interesado al que no se llama."))
COSTO_FP = int(st.sidebar.number_input("Costo de un falso positivo", min_value=0, value=mdl.COSTO_FP, step=1,
                                       help="Llamada a un cliente que no contrata."))
if COSTO_FN == 0 and COSTO_FP == 0:
    st.sidebar.warning("Con ambos costos en cero no hay umbral óptimo; se usan los valores 10 y 1.")
    COSTO_FN, COSTO_FP = mdl.COSTO_FN, mdl.COSTO_FP

if archivo is not None:
    CONTENIDO, ORIGEN = archivo.getvalue(), f"archivo cargado: {archivo.name}"
elif RUTA_DATOS.exists():
    CONTENIDO, ORIGEN = RUTA_DATOS.read_bytes(), "archivo del repositorio: data/bank-full.csv"
else:
    st.error("No se encontró data/bank-full.csv. Cargue el archivo desde la barra lateral.")
    st.stop()

try:
    ORIGINAL, DATOS, REGISTRO = cargar_datos(CONTENIDO)
except ValueError as error:
    st.error(f"El archivo no se pudo usar: {error}")
    st.info("Columnas esperadas: " + ", ".join(mdl.COLUMNAS))
    st.stop()
except Exception as error:  # archivo dañado o con formato inesperado
    st.error(f"Error al leer el archivo: {error}")
    st.stop()

CLAVE = hashlib.md5(CONTENIDO).hexdigest()
PREVALENCIA = DATOS["y"].eq("yes").mean()
if DATOS["y"].eq("yes").sum() < 50 or len(DATOS) < 500:
    st.warning("El archivo tiene pocas observaciones o pocos casos positivos; las métricas serán inestables.")


# ---------------------------------------------------------------------------
# Módulos
# ---------------------------------------------------------------------------
def modulo_inicio():
    st.title("Priorización de llamadas para la venta de depósitos a plazo")
    st.markdown(
        "Esta aplicación integra los resultados del proyecto integrador sobre el conjunto **Bank Marketing** "
        "(Moro et al., 2014), que registra las campañas de telemercadeo de un banco portugués entre mayo de 2008 "
        "y noviembre de 2010. Cada fila es un cliente contactado y la variable objetivo `y` indica si suscribió "
        "un depósito a plazo.")
    st.markdown(
        "**Decisión que apoya:** a qué clientes llamar, y en qué orden, para captar la mayor cantidad de "
        "depósitos sin hacer un número inviable de llamadas innecesarias. Se supone que perder a un cliente "
        "interesado cuesta diez veces más que una llamada sin resultado; ambos costos se pueden cambiar en la "
        "barra lateral.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", num(len(DATOS), 0))
    c2.metric("Variables", len(ORIGINAL.columns))
    c3.metric("Clientes que contratan", pct(PREVALENCIA))
    c4.metric("Partición", f"{int((1 - PROP_PRUEBA) * 100)} / {int(PROP_PRUEBA * 100)}")
    st.caption(f"Fuente de datos en uso: {ORIGEN}.")
    st.subheader("Contenido")
    st.markdown(
        "- **Carga y análisis de datos:** primeras filas, estadísticas descriptivas, tipos de datos, faltantes, "
        "histogramas, gráficos de barras, matriz de correlación y diccionario de datos.\n"
        "- **Análisis de componentes principales:** varianza explicada, criterio de retención y cargas.\n"
        "- **Agrupamiento K-means:** Elbow, silhouette, tamaño y perfiles de los clústeres, sensibilidad y "
        "estabilidad.\n"
        "- **Regresión lineal múltiple:** modelo completo y reducido para el saldo, con diagnóstico de residuos.\n"
        "- **Modelos de clasificación:** regresión logística, árbol de decisión y Random Forest, con matriz de "
        "confusión, reporte de clasificación, curvas ROC y de precisión-sensibilidad y umbral por costo.\n"
        "- **Comparación y priorización:** los tres modelos recomendados en la misma partición de prueba y la "
        "tabla de ganancias para ordenar las llamadas.\n"
        "- **Predicción de nuevos clientes:** puntuación de un archivo de clientes y descarga de la lista de llamadas.\n"
        "- **Reproducibilidad y despliegue:** especificaciones, versiones, verificación frente al informe e "
        "instrucciones de ejecución.")
    st.info("La duración de la llamada (`duration`) se excluye de todos los modelos predictivos, porque solo se "
            "conoce cuando la llamada termina. Las relaciones que muestran los modelos son asociaciones, no causas.")


def modulo_datos():
    st.header("Carga y análisis de datos")
    st.caption(f"Fuente: {ORIGEN}. Filas: {num(len(ORIGINAL), 0)}; columnas: {ORIGINAL.shape[1]}.")
    with st.expander("Resultado de la preparación (reglas de la sección 3)"):
        mostrar_tabla(REGISTRO)
        st.caption(f"Filas después de la preparación: {num(len(DATOS), 0)}. El archivo original no se modifica.")
    pestanas = st.tabs(["Primeras 10 filas", "Estadísticas descriptivas", "Tipos de datos y faltantes",
                        "Visualizaciones", "Diccionario de datos"])
    with pestanas[0]:
        st.dataframe(ORIGINAL.head(10))
    with pestanas[1]:
        st.markdown("**Variables numéricas** (`describe()`)")
        st.dataframe(ORIGINAL.describe().T.style.format(lambda v: num(v, 2)))
        st.markdown("**Variables categóricas**")
        categoricas = [c for c in ORIGINAL.columns if c not in mdl.NUMERICAS]
        st.dataframe(ORIGINAL[categoricas].astype(str).describe().T)
    with pestanas[2]:
        buffer = io.StringIO()
        ORIGINAL.info(buf=buffer)
        st.markdown("**Salida de `info()`**")
        st.code(buffer.getvalue(), language=None)
        faltantes = pd.DataFrame({
            "Variable": ORIGINAL.columns, "Tipo": ORIGINAL.dtypes.astype(str).to_numpy(),
            "Faltantes": ORIGINAL.isna().sum().to_numpy(),
            "Faltantes (%)": (ORIGINAL.isna().mean() * 100).to_numpy(),
            "Valores distintos": ORIGINAL.nunique().to_numpy(),
            "Valores 'unknown'": [int(ORIGINAL[c].astype(str).eq("unknown").sum()) for c in ORIGINAL.columns],
        })
        mostrar_tabla(faltantes, decimales=2)
        st.caption("Las categorías 'unknown' se conservan como una categoría más: no son faltantes al azar.")
    with pestanas[3]:
        st.markdown("**Histogramas de variables numéricas**")
        c1, c2, c3 = st.columns([2, 1, 1])
        variable = c1.selectbox("Variable numérica", mdl.NUMERICAS, index=1)
        recortar = c2.checkbox("Vista P1-P99", value=variable in ("balance", "duration", "campaign", "previous"),
                               help="Solo limita la vista del gráfico; no elimina filas.")
        por_respuesta = c3.checkbox("Separar por contratación", value=False)
        mostrar_figura(fig_histograma(DATOS, variable, recortar, por_respuesta))
        st.markdown("**Gráficos de barras de variables categóricas**")
        categoria = st.selectbox("Variable categórica", mdl.CATEGORICAS, index=8)
        mostrar_figura(fig_categoria(DATOS, categoria))
        st.markdown("**Matriz de correlación**")
        mostrar_figura(fig_correlacion(DATOS))
        st.caption("La correlación de Pearson solo mide relaciones lineales. La duración se correlaciona con la "
                   "contratación, pero no se usa para predecir porque se conoce después de la llamada.")
    with pestanas[4]:
        st.dataframe(mdl.DICCIONARIO, hide_index=True)


def modulo_pca():
    st.header("Análisis de componentes principales (PCA)")
    st.markdown(
        "El PCA resume las siete variables numéricas (`age`, `balance`, `day`, `duration`, `campaign`, `pdays`, "
        "`previous`) en componentes no correlacionadas. Las variables se estandarizan antes del análisis: sin "
        "escalado, `balance`, medida en miles de euros, dominaría la varianza total frente a `campaign` o "
        "`previous`, que se miden en unidades.")
    minimo = st.slider("Varianza acumulada mínima (%)", 70, 95, 80, 5)
    r = pca_calculado(CLAVE, minimo, DATOS)
    k = r["retenidas"]
    acumulada = r["varianza"]["Varianza acumulada (%)"].iloc[k - 1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Componentes retenidas", k)
    c2.metric("Varianza conservada", f"{num(acumulada, 2)} %")
    c3.metric("Varianza que se pierde", f"{num(100 - acumulada, 2)} %")
    col1, col2 = st.columns([3, 2])
    with col1:
        mostrar_figura(fig_varianza_pca(r["varianza"], minimo))
    with col2:
        mostrar_tabla(r["varianza"], decimales=2)
    mostrar_figura(fig_cargas(r["cargas"].iloc[:, :k]))
    st.markdown(
        "Con el criterio del 80 % se retienen cinco componentes (80,48 %). CP1 se asocia con los contactos "
        "previos (`pdays` y `previous`); CP2, con las características de la campaña y la llamada (`campaign`, "
        "`day` y, en sentido opuesto, `duration`); CP3, con la edad y el saldo en el mismo sentido; CP4, con "
        "la duración de la llamada, y CP5, con el saldo y la edad en sentidos opuestos. La varianza se reparte "
        "de forma pareja entre componentes, lo que indica correlaciones bajas entre las variables originales.")
    st.caption("Las cargas describen cómo se combinan las variables; no indican causalidad ni seleccionan "
               "variables para los modelos predictivos.")


def modulo_kmeans():
    st.header("Agrupamiento con K-means")
    with st.expander("Cómo funciona K-means", expanded=False):
        st.markdown(
            "K-means divide a los clientes en k grupos. Empieza con k centros (centroides), asigna cada cliente al "
            "centro más cercano y recalcula cada centro como el promedio de sus clientes, hasta que las "
            "asignaciones no cambian. Busca minimizar la inercia, que es la suma de las distancias al cuadrado de "
            "cada cliente a su centroide:")
        st.latex(r"\text{Inercia} = \sum_{j=1}^{k} \sum_{x_i \in C_j} \lVert x_i - \mu_j \rVert^2")
        st.markdown(
            "El coeficiente silhouette compara, para cada cliente, la distancia media a su propio grupo (a) con la "
            "del grupo vecino más cercano (b): s = (b − a) / máx(a, b). Va de −1 a 1; valores cercanos a 1 indican "
            "grupos bien separados y cercanos a 0, clientes en la frontera entre dos grupos.")
    st.markdown(
        "Se agrupa con `age`, `balance`, `duration` y `campaign`, transformadas con Yeo-Johnson y estandarizadas, "
        "porque son muy asimétricas y, sin transformar, unos pocos clientes con saldos o contactos extremos "
        "definirían los grupos. Las variables categóricas, `pdays` y `previous` no entran al agrupamiento; solo "
        "describen los perfiles. Se usan 10 reinicios (n_init = 10) y semilla 42.")
    k = st.slider("Número de clústeres (k)", mdl.K_MIN, mdl.K_MAX, mdl.K_FINAL, 1, key="k_kmeans",
                  help="El informe usa k = 4.")
    with st.spinner("Calculando la curva del codo y el silhouette para k entre 2 y 8 (la primera vez tarda cerca de un minuto)..."):
        curva = curva_k_calculada(CLAVE, DATOS)
    with st.spinner(f"Ajustando K-means con k = {k}..."):
        r = kmeans_calculado(CLAVE, k, DATOS)
    c1, c2, c3 = st.columns(3)
    c1.metric("Codo aproximado", curva["k_codo"])
    c2.metric("k con mayor silhouette", curva["k_silhouette"])
    c3.metric(f"Silhouette promedio con k = {k}", num(r["silhouette_promedio"], 3))

    pestanas = st.tabs(["Elbow y silhouette", "Silhouette por clúster", "Tamaño", "Perfiles",
                        "Sensibilidad", "Estabilidad", "Asignación de clientes"])
    with pestanas[0]:
        mostrar_figura(fig_curva_k(curva["tabla"], k))
        mostrar_tabla(curva["tabla"], decimales=4)
        st.markdown(
            "La inercia baja de forma gradual, sin un codo marcado (se aproxima en k = 3), y el silhouette se mantiene "
            "entre 0,20 y 0,24 para todos los valores de k, con su máximo en k = 8, el límite del rango. Como los dos "
            "criterios no coinciden, el informe adopta k = 4 como compromiso entre separación e interpretabilidad: "
            "ocho segmentos serían difíciles de gestionar en una campaña comercial.")
    with pestanas[1]:
        col1, col2 = st.columns([3, 2])
        with col1:
            mostrar_figura(fig_silueta_obs(r["sil_obs"], r["lab_obs"], k))
        with col2:
            st.markdown(
                "Cada banda es un clúster; su altura es proporcional al número de clientes y su ancho, al silhouette "
                "de cada cliente. Un silhouette promedio menor que 0,25 indica una estructura débil: los clientes no "
                "forman grupos naturalmente aislados, sino un continuo con muchos casos de frontera. Los segmentos "
                "deben leerse como una descripción orientativa de perfiles.")
    with pestanas[2]:
        col1, col2 = st.columns([3, 2])
        with col1:
            mostrar_figura(fig_tamanos(r["tamanos"]))
        with col2:
            mostrar_tabla(r["tamanos"], decimales=3)
        st.caption("La tasa de contratación de cada clúster no se usó para formarlos; se calcula después, para "
                   "describir el valor comercial de cada segmento.")
    with pestanas[3]:
        mostrar_figura(fig_perfiles(r["indice"], r["dif_std"]))
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Media por clúster (unidades originales)**")
            st.dataframe(r["medias"].style.format(lambda v: num(v, 2)))
        with col2:
            st.markdown("**Mediana por clúster (unidades originales)**")
            st.dataframe(r["medianas"].style.format(lambda v: num(v, 2)))
        categoria = st.selectbox("Perfil categórico (% de clientes dentro de cada clúster)",
                                 list(r["categoricas"]), index=len(r["categoricas"]) - 1, key="cat_kmeans")
        tabla_cat = r["categoricas"][categoria].copy()
        tabla_cat.index.name = "Clúster"
        st.dataframe(tabla_cat.style.format(lambda v: num(v, 1)))
        if k == mdl.K_FINAL:
            st.markdown(
                "**Lectura de los perfiles con k = 4:** el clúster 0 reúne clientes muy contactados (+88 % en "
                "`campaign`) con llamadas breves (−61 % en `duration`) y casi no contrata (1,4 %); el clúster 1, "
                "clientes jóvenes (−22 % en edad) con poco saldo y pocos contactos; el clúster 2, conversaciones "
                "largas (+109 % en duración) y la mayor tasa de contratación (23,3 %); y el clúster 3, clientes de "
                "mayor edad (+27 %) y mayor saldo (+86 %). La diferencia del clúster 2 se explica en buena parte por "
                "la duración, que solo se conoce después de la llamada: el agrupamiento describe, no predice.")
    with pestanas[4]:
        st.markdown("Pone a prueba la solución frente a cambios en la semilla, el escalado y las variables. "
                    "El índice de Rand ajustado (ARI) vale 1 si dos agrupamientos son idénticos y cerca de 0 si "
                    "no se parecen más que por azar.")
        if st.button("Calcular sensibilidad", key="btn_sensibilidad"):
            with st.spinner("Ajustando K-means con 30 semillas, 3 escalados y 4 subconjuntos de variables..."):
                st.session_state[f"sens_{k}"] = sensibilidad_calculada(CLAVE, k, DATOS, r["etiquetas"])
        s = st.session_state.get(f"sens_{k}")
        if s is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("ARI medio entre 30 semillas", num(s["semillas"]["ARI medio"], 3))
            c2.metric("ARI mínimo", num(s["semillas"]["ARI mínimo"], 3))
            c3.metric("Rango de inercia", f"{num(s['semillas']['Inercia mínima'], 0)} – {num(s['semillas']['Inercia máxima'], 0)}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Escalado**")
                mostrar_tabla(s["escalado"])
            with col2:
                st.markdown("**Retirando una variable**")
                mostrar_tabla(s["variables"])
            st.caption("Con una sola inicialización, algunas semillas llevan a soluciones distintas (ARI mínimo bajo); "
                       "por eso se usan 10 reinicios. Cambiar el escalado cambia casi por completo los grupos, y "
                       "retirar duration, campaign o age los altera mucho: son las variables que definen la "
                       "segmentación, mientras que balance pesa menos.")
    with pestanas[5]:
        remuestreos = st.select_slider("Remuestreos del 80 % de los clientes", options=[10, 20, 30, 50], value=20,
                                       key="remuestreos")
        if st.button("Calcular estabilidad", key="btn_estabilidad"):
            with st.spinner(f"Ajustando K-means en {remuestreos} remuestreos..."):
                st.session_state[f"estab_{k}_{remuestreos}"] = estabilidad_calculada(
                    CLAVE, k, remuestreos, DATOS, r["etiquetas"])
        e = st.session_state.get(f"estab_{k}_{remuestreos}")
        if e is not None:
            c1, c2 = st.columns(2)
            c1.metric("ARI medio frente a la solución base", num(e["ari_medio"], 3))
            c2.metric("Intervalo del 95 %", f"{num(e['ic95'][0], 3)} – {num(e['ic95'][1], 3)}")
            mostrar_tabla(e["jaccard"])
            st.caption("Jaccard medio por clúster: mayor que 0,75 es estable, entre 0,60 y 0,75 moderado y menor que "
                       "0,60 inestable. Un clúster inestable cambia de composición con pequeños cambios en los datos.")
    with pestanas[6]:
        st.dataframe(r["asignacion"].head(200), hide_index=True)
        boton_descarga(r["asignacion"], f"asignacion_clusters_k{k}.csv",
                       "Descargar la asignación de clústeres (CSV)", "desc_kmeans")
        st.caption("pdays se muestra con su valor original (-1 = sin contacto previo); en los perfiles se usa 0.")


def modulo_regresion():
    st.header("Regresión lineal múltiple: saldo medio anual")
    with st.expander("Cómo funciona la regresión lineal múltiple", expanded=False):
        st.markdown(
            "La regresión lineal múltiple estima el saldo como una combinación lineal de las variables del "
            "cliente y elige los coeficientes que minimizan la suma de los errores al cuadrado (mínimos "
            "cuadrados ordinarios).")
        st.latex(r"\widehat{balance} = \beta_0 + \beta_1 x_1 + \dots + \beta_k x_k")
        st.markdown(
            "Cada coeficiente es el cambio esperado del saldo, en euros, cuando la variable aumenta, con las "
            "demás fijas; en las numéricas estandarizadas corresponde a una desviación estándar y en las "
            "categóricas compara con la categoría de referencia. R² es la proporción de la variación del "
            "saldo que explica el modelo; MAE es el error absoluto medio y RMSE la raíz del error cuadrático "
            "medio, ambos en euros.")
    especificacion = st.radio("Especificación", ["Reducido", "Completo"], horizontal=True,
                              format_func=lambda e: f"{e} ({len(mdl.VARIABLES_LINEAL[e])} variables)")
    with st.spinner("Ajustando la regresión..."):
        resultados = {e: regresion_entrenada(CLAVE, e, SEMILLA, PROP_PRUEBA, DATOS) for e in mdl.VARIABLES_LINEAL}
    r = resultados[especificacion]
    m = r["metricas"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² en prueba", num(m["R² prueba"], 3))
    c2.metric("MAE en prueba", f"{num(m['MAE prueba (euros)'], 0)} €")
    c3.metric("RMSE en prueba", f"{num(m['RMSE prueba (euros)'], 0)} €")
    c4.metric("MAE prediciendo la mediana", f"{num(m['MAE con la mediana (euros)'], 0)} €")
    st.markdown("**Comparación de especificaciones** (misma partición de prueba)")
    tabla_reg = pd.DataFrame([resultados[e]["metricas"] for e in ["Completo", "Reducido"]])
    for col in [c for c in tabla_reg.columns if "euros" in c]:
        tabla_reg[col] = tabla_reg[col].round(0).astype(int)
    mostrar_tabla(tabla_reg)
    st.caption("El modelo reducido elimina day, campaign, pdays, previous y poutcome (prueba F parcial, "
               "sección 9.1). Ambos explican cerca del 5 % de la variación del saldo: sirven para describir "
               "asociaciones, no para estimar el saldo de un cliente concreto.")
    pestanas = st.tabs(["Reales frente a predichos", "Residuos", "Coeficientes", "Tabla de predicciones"])
    with pestanas[0]:
        mostrar_figura(fig_real_predicho(r["predicciones"]))
    with pestanas[1]:
        mostrar_figura(fig_residuos(r["predicciones"]))
        st.caption("El patrón en abanico indica heterocedasticidad y la cola derecha muestra que el modelo "
                   "subestima los saldos altos; los intervalos de confianza de MCO no son confiables.")
    with pestanas[2]:
        coef = r["coeficientes"].head(15)
        mostrar_figura(fig_barras_signo(coef, "Coeficiente", "15 coeficientes de mayor magnitud",
                                        "Cambio esperado del saldo (euros)"))
        st.caption(f"Intercepto: {num(r['intercepto'], 2)} euros. Azul: asociación positiva; rojo: negativa.")
    with pestanas[3]:
        st.dataframe(r["predicciones"].head(200), hide_index=True)
        boton_descarga(r["predicciones"], f"predicciones_regresion_{especificacion.lower()}_semilla{SEMILLA}.csv",
                       "Descargar predicciones de la regresión (CSV)", "desc_reg")


EXPLICACIONES = {
    "Regresión logística": (
        "La regresión logística estima la probabilidad de que el cliente contrate a partir de una combinación "
        "lineal de sus variables, transformada por la función logística:",
        r"P(y=1 \mid x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_k x_k)}}",
        "Cada coeficiente indica cuánto cambia el logaritmo de las odds de contratar cuando la variable aumenta "
        "y las demás se mantienen; su exponencial es la razón de odds. Las variables numéricas se estandarizan y "
        "las categóricas se convierten en indicadores 0/1 sin la primera categoría, para evitar colinealidad "
        "perfecta. Como solo el 11,7 % de los clientes contrata, `class_weight='balanced'` da más peso a esa "
        "clase durante el ajuste. El modelo entrega una probabilidad y la decisión de llamar se toma al "
        "compararla con un umbral."),
    "Árbol de decisión": (
        "El árbol de decisión divide a los clientes en grupos cada vez más homogéneos mediante preguntas "
        "binarias sobre una variable, por ejemplo, si la campaña anterior fue exitosa. En cada nodo elige la "
        "división que más reduce la impureza de Gini:",
        r"G = 1 - \sum_{i} p_i^{2}",
        "donde $p_i$ es la proporción de cada clase en el nodo. La probabilidad asignada a un cliente es la "
        "proporción ponderada de contrataciones en la hoja a la que llega. Sin control, el árbol crece hasta "
        "memorizar el entrenamiento; la versión recomendada limita la profundidad a 10 niveles y exige al menos "
        "50 clientes por hoja. No necesita escalado, porque sus cortes dependen del orden de los valores y no "
        "de su magnitud."),
    "Random Forest": (
        "El Random Forest combina 300 árboles de decisión. Cada árbol se entrena con una muestra bootstrap de "
        "los clientes y, en cada división, solo considera un subconjunto aleatorio de variables. La "
        "probabilidad final es el promedio de los árboles:",
        r"\hat{p}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{p}_b(x), \qquad B = 300",
        "Al promediar árboles poco correlacionados se reduce la varianza y las predicciones son más estables que "
        "las de un solo árbol (Breiman, 2001). Los clientes que quedan fuera de la muestra de un árbol "
        "(out-of-bag) permiten evaluarlo sin usar la prueba; la aplicación usa esas probabilidades para elegir "
        "el umbral por costo. Su desventaja es que no produce reglas fáciles de leer."),
}


def interpretar_modelo(r):
    modelo = r["modelo"]
    if r["nombre"] in (mdl.LOGISTICA_REDUCIDA, mdl.LOGISTICA_COMPLETA):
        nombres = nombres_limpios(modelo.named_steps["transformer"].get_feature_names_out())
        coef = pd.DataFrame({"Variable": nombres, "Coeficiente": modelo.named_steps["classifier"].coef_[0]})
        coef["Razón de odds"] = np.exp(coef["Coeficiente"])
        coef = coef.reindex(coef["Coeficiente"].abs().sort_values(ascending=False).index).head(15)
        mostrar_figura(fig_barras_signo(coef, "Razón de odds", "15 variables con mayor efecto (razón de odds)",
                                        "Razón de odds (1 = sin efecto)", referencia=1.0))
        mostrar_tabla(coef)
        st.caption("Razón de odds mayor que 1: la característica se asocia con mayor probabilidad de contratar; "
                   "menor que 1, con menor probabilidad. En las numéricas el cambio es de una desviación estándar.")
        return
    arbol = modelo.named_steps["modelo"]
    nombres = nombres_limpios(modelo.named_steps["prep_arb"].get_feature_names_out())
    importancia = pd.DataFrame({"Variable": nombres, "Importancia": arbol.feature_importances_})
    importancia = importancia.sort_values("Importancia", ascending=False).head(15)
    if r["nombre"] != mdl.RANDOM_FOREST:
        c1, c2 = st.columns(2)
        c1.metric("Profundidad", arbol.get_depth())
        c2.metric("Hojas", num(arbol.get_n_leaves(), 0))
        fig, ax = plt.subplots(figsize=(15, 6.5))
        plot_tree(arbol, max_depth=2, feature_names=nombres, class_names=["No contrata", "Contrata"],
                  filled=True, proportion=True, impurity=False, rounded=True, fontsize=8, ax=ax)
        ax.set_title("Primeros niveles del árbol", loc="left")
        mostrar_figura(fig)
        with st.expander("Reglas de los primeros tres niveles"):
            st.code(export_text(arbol, feature_names=nombres, max_depth=3, show_weights=False), language=None)
    fig = fig_barras_signo(importancia, "Importancia", "15 variables con mayor importancia",
                           "Reducción media de la impureza de Gini")
    mostrar_figura(fig)
    st.caption("La importancia mide cuánto contribuye cada variable a reducir la impureza en el entrenamiento; "
               "no es un efecto causal y favorece a las variables con muchos valores posibles.")


def modulo_clasificacion():
    st.header("Construcción y evaluación de modelos de clasificación")
    c1, c2 = st.columns(2)
    algoritmo = c1.selectbox("Algoritmo", list(EXPLICACIONES), key="algoritmo")
    if algoritmo == "Regresión logística":
        nombre = c2.radio("Especificación", [mdl.LOGISTICA_REDUCIDA, mdl.LOGISTICA_COMPLETA], key="esp_logistica",
                          format_func=lambda n: "Reducida (recomendada)" if n == mdl.LOGISTICA_REDUCIDA else "Completa")
    elif algoritmo == "Árbol de decisión":
        nombre = c2.radio("Especificación", [mdl.ARBOL_PODADO, mdl.ARBOL_LIBRE], key="esp_arbol",
                          format_func=lambda n: "Podado (recomendado)" if n == mdl.ARBOL_PODADO else "Sin límite de profundidad")
    else:
        nombre = mdl.RANDOM_FOREST
        c2.markdown("**Especificación**  \n300 árboles (sección 10)")
    texto_1, formula, texto_2 = EXPLICACIONES[algoritmo]
    with st.expander(f"Cómo funciona: {algoritmo.lower()}", expanded=True):
        st.markdown(texto_1)
        st.latex(formula)
        st.markdown(texto_2)
    st.caption(f"{mdl.DESCRIPCION_MODELOS[nombre]} Partición estratificada "
               f"{int((1 - PROP_PRUEBA) * 100)}/{int(PROP_PRUEBA * 100)} con semilla {SEMILLA}.")

    r = obtener_clasificador(nombre)
    umbral_costo = mdl.umbral_por_costo(r["y_entrenamiento"], r["prob_seleccion"], COSTO_FN, COSTO_FP)
    regla = st.radio("Regla de decisión", ["Umbral por costo", "Umbral 0,5", "Umbral manual"], horizontal=True,
                     key="regla")
    if regla == "Umbral por costo":
        umbral = umbral_costo
    elif regla == "Umbral 0,5":
        umbral = 0.5
    else:
        umbral = st.slider("Umbral", 0.01, 0.99, float(round(umbral_costo, 2)), 0.01)
    st.caption(f"Umbral por costo = {num(umbral_costo, 3)}, elegido con {r['metodo_umbral']} "
               f"(costo FN = {COSTO_FN}, costo FP = {COSTO_FP}). El grupo de prueba solo se usa para evaluar.")

    m = mdl.metricas_clasificacion(r["y_prueba"], r["prob_prueba"], umbral, COSTO_FN, COSTO_FP)
    cols = st.columns(6)
    for col, (etiqueta, valor) in zip(cols, [("ROC-AUC", num(m["ROC-AUC"])), ("PR-AUC", num(m["PR-AUC"])),
                                            ("Sensibilidad", num(m["Sensibilidad"])), ("Precisión", num(m["Precisión"])),
                                            ("F1", num(m["F1"])), ("Costo total", num(m["Costo total"], 0))]):
        col.metric(etiqueta, valor)

    pestanas = st.tabs(["Matriz de confusión", "Reporte de clasificación", "Curva ROC",
                        "Curva precisión-sensibilidad", "Umbral y costo", "Interpretación", "Tabla de predicciones"])
    with pestanas[0]:
        col1, col2 = st.columns([3, 2])
        with col1:
            mostrar_figura(fig_matriz_confusion(m))
        with col2:
            st.markdown(
                f"- **Verdaderos positivos ({num(m['VP'], 0)}):** clientes que contratan y el modelo señala.\n"
                f"- **Falsos positivos ({num(m['FP'], 0)}):** llamadas a clientes que no contratan.\n"
                f"- **Falsos negativos ({num(m['FN'], 0)}):** clientes interesados a los que no se llama.\n"
                f"- **Verdaderos negativos ({num(m['VN'], 0)}):** clientes que no contratan y no se llaman.\n\n"
                f"Costo total = {COSTO_FN} × {num(m['FN'], 0)} + {COSTO_FP} × {num(m['FP'], 0)} = "
                f"**{num(m['Costo total'], 0)}**. Exactitud: {num(m['Exactitud'])}; especificidad: "
                f"{num(m['Especificidad'])}.")
    with pestanas[1]:
        pred = (r["prob_prueba"] >= umbral).astype(int)
        reporte = classification_report(r["y_prueba"], pred, labels=[0, 1], output_dict=True, zero_division=0,
                                        target_names=["No contrata (0)", "Contrata (1)"])
        exactitud = reporte.pop("accuracy")
        tabla = pd.DataFrame(reporte).T.reset_index()
        tabla.columns = ["Clase", "Precisión", "Sensibilidad (recall)", "F1-score", "Soporte"]
        tabla["Clase"] = tabla["Clase"].replace({"macro avg": "Promedio macro", "weighted avg": "Promedio ponderado"})
        tabla["Soporte"] = tabla["Soporte"].astype(int)
        mostrar_tabla(tabla)
        st.caption(f"Exactitud (accuracy): {num(exactitud)}. Con una prevalencia de {pct(PREVALENCIA)}, decir "
                   f"que nadie contrata daría una exactitud de {num(1 - PREVALENCIA)} sin identificar a ningún "
                   "cliente; por eso se priorizan sensibilidad, precisión, F1, PR-AUC y costo.")
    with pestanas[2]:
        col1, col2 = st.columns([3, 2])
        with col1:
            mostrar_figura(fig_roc(r["y_prueba"], r["prob_prueba"], m))
        with col2:
            st.markdown(f"El área bajo la curva ROC es **{num(m['ROC-AUC'])}**: es la probabilidad de que un "
                        "cliente que contrata, tomado al azar, reciba una probabilidad mayor que uno que no "
                        "contrata. Un valor de 0,5 equivale a ordenar al azar. El punto naranja es el umbral en uso.")
    with pestanas[3]:
        col1, col2 = st.columns([3, 2])
        with col1:
            mostrar_figura(fig_pr(r["y_prueba"], r["prob_prueba"], m))
        with col2:
            st.markdown(f"Con clases desbalanceadas, la curva de precisión y sensibilidad es más informativa que "
                        f"la ROC. El PR-AUC de **{num(m['PR-AUC'])}** se compara con la prevalencia "
                        f"({num(PREVALENCIA)}), que es lo que se obtendría al llamar al azar.")
    with pestanas[4]:
        costos = mdl.costos_por_umbral(r["y_entrenamiento"], r["prob_seleccion"], COSTO_FN, COSTO_FP)
        mostrar_figura(fig_costos(costos, umbral_costo))
        m05 = mdl.metricas_clasificacion(r["y_prueba"], r["prob_prueba"], 0.5, COSTO_FN, COSTO_FP)
        mc = mdl.metricas_clasificacion(r["y_prueba"], r["prob_prueba"], umbral_costo, COSTO_FN, COSTO_FP)
        comparacion = pd.DataFrame([dict(Regla="Umbral 0,5", **m05), dict(Regla="Umbral por costo", **mc)])
        mostrar_tabla(comparacion[["Regla", "Umbral", "VP", "FP", "FN", "VN", "Sensibilidad",
                                   "Precisión", "F1", "Costo total"]])
        st.caption("La curva se calcula con las probabilidades fuera de muestra del entrenamiento; la tabla "
                   "muestra el efecto de cada umbral en el grupo de prueba.")
    with pestanas[5]:
        interpretar_modelo(r)
    with pestanas[6]:
        tabla = mdl.tabla_predicciones(DATOS, r, umbral)
        st.dataframe(tabla.sort_values("probabilidad", ascending=False).head(200), hide_index=True)
        st.caption(f"{num(len(tabla), 0)} clientes del grupo de prueba. id_registro es la fila del archivo "
                   "original (contando desde 0). resultado: VP, FP, FN o VN según el umbral en uso.")
        boton_descarga(tabla, f"predicciones_{nombre_corto(nombre)}_semilla{SEMILLA}.csv",
                       "Descargar tabla de predicciones (CSV)", "desc_clas")


def modulo_comparacion():
    st.header("Comparación de modelos y priorización de llamadas")
    resultados = {n: obtener_clasificador(n) for n in mdl.RECOMENDADOS}
    st.markdown(
        f"Los tres modelos recomendados se evalúan en **la misma partición de prueba** (semilla {SEMILLA}, "
        f"{num(len(resultados[mdl.RANDOM_FOREST]['y_prueba']), 0)} clientes), de modo que sus métricas son "
        "comparables. Cada umbral por costo se elige con datos de entrenamiento.")
    tabla, _ = mdl.comparar_clasificadores(DATOS, mdl.RECOMENDADOS, SEMILLA, PROP_PRUEBA,
                                           COSTO_FN, COSTO_FP, resultados=dict(resultados))
    incluir_05 = st.checkbox("Incluir los resultados con umbral 0,5", value=False)
    vista = tabla if incluir_05 else tabla[tabla["Regla"] == "Umbral por costo"]
    columnas = ["Modelo", "Regla", "Umbral", "VP", "FP", "FN", "VN", "Sensibilidad", "Precisión", "F1",
                "ROC-AUC", "PR-AUC", "Costo total"]
    mostrar_tabla(vista[columnas])
    y_prueba = next(iter(resultados.values()))["y_prueba"]
    costo_todos = COSTO_FP * int((y_prueba == 0).sum())
    costo_nadie = COSTO_FN * int((y_prueba == 1).sum())
    st.caption(f"Referencias en la misma prueba: llamar a todos cuesta {num(costo_todos, 0)} y no llamar a nadie "
               f"cuesta {num(costo_nadie, 0)}.")
    boton_descarga(tabla, f"comparacion_modelos_semilla{SEMILLA}.csv", "Descargar tabla comparativa (CSV)", "desc_comp")
    mostrar_figura(fig_comparacion(resultados))

    st.subheader("Priorización de llamadas")
    c1, c2 = st.columns(2)
    elegido = c1.selectbox("Modelo para ordenar a los clientes", mdl.RECOMENDADOS, index=2, key="modelo_ganancias")
    porcentaje = c2.slider("Porcentaje de clientes que se contactaría", 5, 100, 20, 5, key="porcentaje")
    r = resultados[elegido]
    ganancias = mdl.tabla_ganancias(r["y_prueba"], r["prob_prueba"])
    orden = np.argsort(-r["prob_prueba"], kind="mergesort")
    k = int(round(porcentaje / 100 * len(orden)))
    capturados = int(r["y_prueba"][orden[:k]].sum())
    total = int(r["y_prueba"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Llamadas", num(k, 0))
    c2.metric("Contrataciones capturadas", f"{num(capturados, 0)} de {num(total, 0)}", pct(capturados / total),
              delta_color="off")
    c3.metric("Tasa de éxito de las llamadas", pct(capturados / max(k, 1)), f"base {pct(total / len(orden))}",
              delta_color="off")
    col1, col2 = st.columns(2)
    with col1:
        mostrar_figura(fig_ganancias(r["y_prueba"], r["prob_prueba"], elegido, porcentaje))
    with col2:
        vista = ganancias[["Decil", "Contrataciones", "Tasa de contratación (%)",
                           "Contrataciones capturadas (%)", "Lift acumulado"]]
        vista.columns = ["Decil", "Contratan", "Tasa (%)", "Capturadas acum. (%)", "Lift acum."]
        mostrar_tabla(vista, decimales=2)
    lista = mdl.tabla_predicciones(DATOS, r, mdl.umbral_por_costo(r["y_entrenamiento"], r["prob_seleccion"],
                                                                  COSTO_FN, COSTO_FP))
    lista = lista.sort_values("probabilidad", ascending=False, kind="mergesort").reset_index(drop=True)
    lista.insert(1, "prioridad", np.arange(1, len(lista) + 1))
    lista.insert(2, "decil", np.ceil(lista["prioridad"] / len(lista) * 10).astype(int))
    c1, c2 = st.columns(2)
    with c1:
        boton_descarga(ganancias, f"ganancias_{nombre_corto(elegido)}_semilla{SEMILLA}.csv",
                       "Descargar tabla de ganancias (CSV)", "desc_gan")
    with c2:
        boton_descarga(lista, f"priorizacion_{nombre_corto(elegido)}_semilla{SEMILLA}.csv",
                       "Descargar lista priorizada del grupo de prueba (CSV)", "desc_lista")
    st.caption("El lift acumulado indica cuántas veces es mayor la tasa de contratación del grupo contactado "
               "frente a llamar al azar. Las probabilidades de modelos entrenados con pesos balanceados sirven "
               "para ordenar a los clientes, no como probabilidades calibradas.")


def modulo_prediccion():
    st.header("Predicción de nuevos clientes")
    st.markdown(
        "Se puntúa un archivo de clientes con la estructura de bank-full.csv (no requiere `y` ni `duration`). "
        "El modelo es el entrenado con la partición de entrenamiento actual y la decisión usa el umbral elegido.")
    plantilla = ORIGINAL.drop(columns=[c for c in ("y", "duration") if c in ORIGINAL.columns]).head(20)
    st.download_button("Descargar plantilla de ejemplo (CSV)", data=plantilla.to_csv(index=False, sep=";").encode("utf-8"),
                       file_name="plantilla_clientes.csv", mime="text/csv", key="desc_plantilla")
    archivo_nuevos = st.file_uploader("Archivo de clientes por puntuar (CSV)", type=["csv"], key="nuevos")
    c1, c2 = st.columns(2)
    nombre = c1.selectbox("Modelo", mdl.RECOMENDADOS, index=2, key="modelo_prediccion")
    regla = c2.radio("Umbral", ["Umbral por costo", "Umbral manual"], horizontal=True, key="regla_prediccion")
    r = obtener_clasificador(nombre)
    umbral = mdl.umbral_por_costo(r["y_entrenamiento"], r["prob_seleccion"], COSTO_FN, COSTO_FP)
    if regla == "Umbral manual":
        umbral = st.slider("Umbral de decisión", 0.01, 0.99, float(round(umbral, 2)), 0.01)

    if archivo_nuevos is None:
        st.info("No se ha cargado un archivo. Como ejemplo se puntúan los clientes del grupo de prueba.")
        nuevos = DATOS.iloc[r["idx_prueba"]][mdl.ENTRADAS_PREDICCION]
        excluidas = 0
    else:
        try:
            nuevos, excluidas = mdl.preparar_nuevos(mdl.leer_csv(archivo_nuevos.getvalue()))
        except ValueError as error:
            st.error(f"El archivo no se pudo puntuar: {error}")
            st.info("Columnas obligatorias: " + ", ".join(mdl.ENTRADAS_PREDICCION))
            return
    if excluidas:
        st.warning(f"{excluidas} filas tienen valores faltantes o no numéricos en columnas obligatorias y no se puntuaron.")
    desconocidas = mdl.categorias_desconocidas(r, nuevos)
    if desconocidas:
        st.warning("Categorías que no aparecen en el entrenamiento (se tratan como ausentes): "
                   + "; ".join(f"{k}: {', '.join(v[:5])}" for k, v in desconocidas.items()))
    salida = mdl.puntuar_clientes(r, nuevos, umbral)
    llamar = int((salida["decision"] == "Llamar").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Clientes puntuados", num(len(salida), 0))
    c2.metric("Clientes para llamar", num(llamar, 0), pct(llamar / len(salida)), delta_color="off")
    c3.metric("Umbral", num(umbral, 3))
    st.dataframe(salida.head(300), hide_index=True)
    boton_descarga(salida, f"lista_llamadas_{nombre_corto(nombre)}.csv",
                   "Descargar predicciones y prioridades (CSV)", "desc_nuevos")
    st.caption("prioridad 1 es el cliente con mayor probabilidad; decil 1 agrupa al 10 % con mayor probabilidad. "
               "Los archivos se descargan con separador ';' y coma decimal para abrirlos en Excel en español.")


def modulo_reproducibilidad():
    st.header("Reproducibilidad y despliegue")
    st.subheader("Especificaciones de los modelos")
    specs = pd.DataFrame([
        ("K-means (k = 4)", "age, balance, duration, campaign (Yeo-Johnson y estandarización)",
         "Todos los registros (no supervisado)", "42", "n_init = 10; k evaluado entre 2 y 8", "No aplica"),
        ("Regresión lineal (reducida)", "age, job, marital, education, default, housing, loan, contact, month",
         "Aleatoria 80/20", "42", "StandardScaler + OneHotEncoder(drop='first') + MCO", "No aplica"),
        (mdl.LOGISTICA_REDUCIDA, ", ".join(mdl.VARIABLES_LOGISTICA["reducida"]), "Estratificada 80/20", "1234 (informe) / 42 (comparación)",
         "class_weight='balanced', lbfgs, max_iter=1000", "Validación cruzada de 5 grupos en entrenamiento"),
        (mdl.ARBOL_PODADO, "16 variables; pdays dividida en contactado_antes y pdays_real", "Estratificada 80/20", "42",
         "Gini, max_depth=10, min_samples_leaf=50, class_weight='balanced'", "Validación cruzada de 5 grupos en entrenamiento"),
        (mdl.RANDOM_FOREST, "Las mismas del árbol", "Estratificada 80/20", "42",
         "300 árboles, min_samples_leaf=5, class_weight='balanced_subsample'", "Probabilidades out-of-bag"),
    ], columns=["Modelo", "Variables", "Partición", "Semilla", "Hiperparámetros", "Selección del umbral"])
    st.table(specs.set_index("Modelo"))
    st.caption("Costos del error: falso negativo = 10, falso positivo = 1. Malla de umbrales: 100 valores entre "
               "0,01 y 0,99. duration se excluye de todos los modelos.")

    st.subheader("Verificación frente al informe técnico")
    st.markdown("Recalcula las cifras publicadas en los puntos 6, 9 y 10 con las semillas originales "
                "(tarda cerca de un minuto).")
    if st.button("Ejecutar verificación", key="verificar"):
        with st.spinner("Recalculando los modelos del informe..."):
            previos = {n: clasificador_entrenado(CLAVE, n, 42, 0.2, DATOS)
                       for n in (mdl.ARBOL_PODADO, mdl.RANDOM_FOREST)}
            st.session_state["verificacion"] = mdl.verificar_reproducibilidad(DATOS, previos)
    if "verificacion" in st.session_state:
        verificacion = st.session_state["verificacion"]
        decimales = [fila[-1] for fila in mdl.VALORES_INFORME]
        vista = verificacion.copy()
        for col in ["Valor del informe", "Valor reproducido"]:
            vista[col] = [num(v, d) for v, d in zip(verificacion[col], decimales)]
        st.dataframe(vista, hide_index=True, height=36 * (len(vista) + 1) + 4)
        st.caption("En los modelos con árboles profundos pueden aparecer diferencias en el tercer decimal entre "
                   "sistemas operativos, por empates en las divisiones. Los demás valores deben coincidir.")
        boton_descarga(verificacion, "verificacion_reproducibilidad.csv", "Descargar verificación (CSV)", "desc_ver")

    st.subheader("Entorno de ejecución")
    versiones = pd.DataFrame({
        "Componente": ["Python", "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "streamlit"],
        "Versión": [platform.python_version(), pd.__version__, np.__version__, sklearn.__version__,
                    matplotlib.__version__, sns.__version__, st.__version__],
    })
    st.dataframe(versiones, hide_index=True)

    st.subheader("Estructura del proyecto")
    st.code(".\n├── app.py              # aplicación Streamlit (módulos de la barra lateral)\n"
            "├── modelos.py          # preparación, modelos y métricas compartidos con Colab\n"
            "├── requirements.txt    # dependencias con versiones fijas\n"
            "├── README.md           # documentación e instrucciones\n"
            "├── .streamlit/config.toml\n"
            "└── data/\n    └── bank-full.csv   # copia sin modificar del archivo original", language=None)

    st.subheader("Ejecución local")
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")

    st.subheader("Despliegue en Streamlit Community Cloud")
    st.markdown(
        "1. Crear un repositorio público en GitHub y subir `app.py`, `modelos.py`, `requirements.txt`, "
        "`README.md`, la carpeta `.streamlit` y la carpeta `data` con `bank-full.csv`.\n"
        "2. Entrar a [share.streamlit.io](https://share.streamlit.io), pulsar **Continue to sign-in** y elegir "
        "**Continue with GitHub**; autorizar el acceso de Streamlit a los repositorios.\n"
        "3. En el espacio de trabajo, pulsar **Create app** y responder **Yup, I have an app**.\n"
        "4. Indicar el repositorio, la rama `main` y el archivo principal `app.py`. Opcionalmente, elegir un "
        "subdominio en **App URL**.\n"
        "5. En **Advanced settings**, seleccionar la misma versión de Python con la que se generó "
        "`requirements.txt` (la que aparece en README.md) y pulsar **Save**.\n"
        "6. Pulsar **Deploy**. La instalación tarda unos minutos; la aplicación queda en una dirección "
        "`https://<subdominio>.streamlit.app` y se actualiza con cada cambio que se suba a la rama.\n\n"
        "Si la aplicación no recibe visitas en 12 horas entra en reposo; basta con abrirla para reactivarla.")


PAGINAS = {
    "Inicio": modulo_inicio,
    "Carga y análisis de datos": modulo_datos,
    "Análisis de componentes principales": modulo_pca,
    "Agrupamiento K-means": modulo_kmeans,
    "Regresión lineal múltiple": modulo_regresion,
    "Modelos de clasificación": modulo_clasificacion,
    "Comparación y priorización": modulo_comparacion,
    "Predicción de nuevos clientes": modulo_prediccion,
    "Reproducibilidad y despliegue": modulo_reproducibilidad,
}

try:
    PAGINAS[MODULO]()
except Exception as error:  # cualquier fallo de un módulo se informa sin detener la aplicación
    st.error(f"Ocurrió un error al ejecutar el módulo: {error}")
    st.exception(error)
