import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def obtener_precios_fred(
    fecha_inicio="2019-01-01",
    fecha_fin="2025-12-31"
):
    series = {
        "precio_cacao_usd_tm": "PCOCOUSDM",
        "tasa_cop_usd": "COLCCUSMA02STM"
    }

    datos = []

    for nombre, serie_id in series.items():

        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={serie_id}"

        df = pd.read_csv(url)

        df["observation_date"] = pd.to_datetime(df["observation_date"])
        df[nombre] = pd.to_numeric(df[serie_id], errors="coerce")

        df = df[
            (df["observation_date"] >= fecha_inicio) &
            (df["observation_date"] <= fecha_fin)
        ]

        df["año"] = df["observation_date"].dt.year

        anual = (
            df.groupby("año")[nombre]
              .mean()
              .reset_index()
        )

        datos.append(anual)

    resultado = datos[0]

    for df in datos[1:]:
        resultado = resultado.merge(df, on="año", how="inner")

    resultado["precio_cacao_cop_tm"] = (
        resultado["precio_cacao_usd_tm"]
        * resultado["tasa_cop_usd"]
    )

    return resultado


def proyectar_concepto(datos, concepto):
    años_historicos = list(range(2019, 2026))
    años_proyeccion = list(range(2026, 2031))

    fila = datos.loc[datos["CONCEPTO"] == concepto]

    if fila.empty:
        raise ValueError(f"No se encontró el concepto: {concepto}")

    # Valores históricos
    y = fila[años_historicos].iloc[0].astype(float).values

    X = pd.DataFrame({"año": años_historicos})

    modelo = LinearRegression()
    modelo.fit(X, y)

    X_futuro = pd.DataFrame({"año": años_proyeccion})
    ingresos_proyectados = modelo.predict(X_futuro)

    proyeccion = pd.DataFrame({
        "año": años_proyeccion,
        "ingresos_proyectados": ingresos_proyectados
    })

    proyeccion.set_index("año")

    return proyeccion

def proyeccion_porcentual(valor, pcte, t):
    return valor * (1+ (pcte/100))**t

def calcular_EBIT(anio, proyeccion_ingresos, proyeccion_costos , gastos_admin, otros_ingresos_ops):
    t = anio - 2025

    EBIT = (
    (0.25 * proyeccion_ingresos.loc[[anio]]) - proyeccion_costos.loc[[anio]]
        + 0
    )


def vpn_estrategia(flujos, tasa_descuento):
    flujos = np.asarray(flujos, dtype=float)
    periodos = np.arange(len(flujos))

    return np.sum(
        flujos / (1 + tasa_descuento) ** periodos
    )

## TEMP

def precio_futuro(S0, r, T, **ajustes):
    carry = r

    carry += ajustes.get("costo_almacenamiento", 0)
    
    carry -= ajustes.get("convenience_yield", 0)

    return S0 * np.exp(carry * T)  

def call(S, K, r, sigma, T):
    if T <= 0:
        return max(S - K, 0)

    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    precio = (
        S * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )

    return precio

def call_trm(
    S,
    K,
    r_domestic,
    r_foreign,
    sigma,
    T
):
    if T <= 0:
        return max(S - K, 0)

    d1 = (
        np.log(S / K)
        + (r_domestic - r_foreign + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    precio = (
        S * np.exp(-r_foreign * T) * norm.cdf(d1)
        - K * np.exp(-r_domestic * T) * norm.cdf(d2)
    )

    return precio