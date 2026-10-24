# Bank Marketing: priorización de llamadas para depósitos a plazo

Aplicación Streamlit del proyecto integrador de Analítica de Datos 2 (Universidad Industrial de Santander).
Integra el análisis exploratorio, el PCA, la regresión lineal múltiple y los modelos de clasificación
(regresión logística, árbol de decisión y Random Forest) del informe técnico, y permite descargar las
tablas de predicciones.

## Datos

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5K306 (licencia CC BY 4.0). Se usa `bank-full.csv` (45.211 registros y 17
variables), guardado sin modificaciones en `data/`.

## Estructura

```
.
├── app.py              # aplicación Streamlit
├── modelos.py          # preparación, modelos y métricas
├── requirements.txt    # dependencias con versiones fijas
├── README.md
├── .streamlit/config.toml
└── data/
    └── bank-full.csv
```

## Ejecución local

```
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Community Cloud

1. Subir el contenido de esta carpeta a un repositorio público de GitHub (rama `main`).
2. Entrar a https://share.streamlit.io con la opción **Continue with GitHub**.
3. Pulsar **Create app**, responder **Yup, I have an app** e indicar el repositorio, la rama `main` y el
   archivo principal `app.py`.
4. En **Advanced settings**, elegir Python 3.13, la versión con la que se generó `requirements.txt`.
5. Pulsar **Deploy**.

## Reproducibilidad

- Preparación: eliminación de duplicados, filas con menos del 80 % de datos e imputación con mediana o moda
  (sin efecto en `bank-full.csv`). `duration` se excluye de los modelos por fuga de información.
- Regresión lineal: partición aleatoria 80/20, semilla 42.
- Regresión logística: partición estratificada 80/20, semilla 1234 en el informe (secciones 8 y 9.2).
- Árbol de decisión y Random Forest: partición estratificada 80/20, semilla 42 (sección 10).
- Comparación de los tres clasificadores: misma partición estratificada, semilla 42.
- Costos del error: falso negativo = 10, falso positivo = 1. El umbral se elige con validación cruzada de
  5 grupos en entrenamiento (probabilidades out-of-bag en el Random Forest).
- Versiones: Python 3.13.1; ver `requirements.txt`.

Los CSV descargados usan separador `;` y coma decimal. Para leerlos con pandas:
`pd.read_csv(archivo, sep=";", decimal=",")`.
