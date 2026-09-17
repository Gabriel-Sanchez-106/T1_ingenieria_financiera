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

def proyectar_costo_ventas(df_historico, df_precios_fred, df_ingresos_proyectados, df_escenario):
    historico = pd.merge(df_historico, df_precios_fred, on='año')

    historico['costo_ventas_ajustado'] = historico['Costo_ventas'] + historico['Gastos_produccion']
    historico['compras_cacao_millones'] = 0.05 * historico['costo_ventas_ajustado']

    historico['toneladas'] = (historico['compras_cacao_millones'] * 1_000_000) / historico['precio_cacao_cop_tm']
    historico['intensidad'] = historico['toneladas'] / historico['Ingresos_operacionales']
    intensidad_promedio = historico.loc[historico['año'].between(2019, 2022), 'intensidad'].mean()

    proyeccion = pd.merge(df_ingresos_proyectados, df_escenario, on='año')
    proyeccion['toneladas_proyectadas'] = proyeccion['Ingresos_operacionales'] * intensidad_promedio
    proyeccion['valor_compra_cacao'] = (
        proyeccion['toneladas_proyectadas'] * proyeccion['precio_cacao_usd'] * proyeccion['tasa_cambio']
    ) / 1_000_000

    proyeccion['otros_costos_ventas'] = 0.50 * proyeccion['Ingresos_operacionales']
    proyeccion['costo_ventas_total'] = proyeccion['valor_compra_cacao'] + proyeccion['otros_costos_ventas']

    costo_ventas_final = proyeccion[['año', 'costo_ventas_total']].set_index('año')['costo_ventas_total']

    return costo_ventas_final, proyeccion['toneladas_proyectadas']

def evaluar_modelo_base(datos_historicos, df_escenarios_consolidados):
    
    df_precios_fred = obtener_precios_fred()
    
    ingresos_proy = proyectar_concepto(datos_historicos, "Ingresos_operacionales")
    ppe_proy = proyectar_concepto(datos_historicos, "Propiedades_planta_equipo")
    
    gastos_admin_2025 = float(datos_historicos.loc[datos_historicos["CONCEPTO"] == "Gastos_administracion", 2025].iloc[0])
    otros_ingresos_2025 = float(datos_historicos.loc[datos_historicos["CONCEPTO"] == "Otros_ingresos_netos", 2025].iloc[0])
    
    tasa_impositiva = 30
    
    resultados = {}
    
    escenarios_unicos = df_escenarios_consolidados['escenario'].unique()
    
    for esc in escenarios_unicos:
        print(f"Evaluando Escenario {esc}...")
        
        df_esc_filtrado = df_escenarios_consolidados[df_escenarios_consolidados['escenario'] == esc].copy()
        
        costos_esc, toneladas_esc = proyectar_costo_ventas(
            datos_historicos, 
            df_precios_fred, 
            ingresos_proy, 
            df_esc_filtrado
        )
        
        flujos_esc, vpn_esc = calcular_flujos_y_vpn(
            tasa_impositiva, 
            ppe_proy, 
            ingresos_proy, 
            costos_esc, 
            gastos_admin_2025, 
            otros_ingresos_2025
        )
        
        resultados[f"Escenario_{esc}"] = {
            "VPN": vpn_esc,
            "Flujos": flujos_esc,
            "Toneladas": toneladas_esc,
            "Costos": costos_esc
        }
        
        print(f"VPN Escenario {esc}: {vpn_esc:,.2f} Millones COP\n")
        
    return resultados

def evaluar_estrategias_cobertura(diccionario_resultados, df_escenarios, df_derivados):

    tasa_descuento = 0.07
    resultados_cobertura = {}
    
    suma_vpn_est1 = 0
    suma_vpn_est2 = 0
    num_escenarios = len(diccionario_resultados)
    
    for esc_nombre, datos_base in diccionario_resultados.items():
        esc = int(esc_nombre.split("_")[1])
        df_esc = df_escenarios[df_escenarios['escenario'] == esc].set_index('año')
        
        vpn_est1 = 0
        vpn_est2 = 0
        
        for anio in range(2026, 2031):
            t = anio - 2025
            
            toneladas_totales = datos_base["Toneladas"].loc[anio]
            toneladas_cubiertas = toneladas_totales * 0.70
            spot_cacao = df_esc.loc[anio, 'precio_cacao_usd']
            spot_trm = df_esc.loc[anio, 'tasa_cambio']
            fcl_base = datos_base["Flujos"].loc[anio]
            
            # --- EXTRACCIÓN DE COTIZACIONES ---
            fw_cacao = df_derivados.loc[(df_derivados['año']==anio) & (df_derivados['especie']=='cacao') & (df_derivados['instrumento']=='futuro'), 'valor'].values[0]
            prima_call_cacao = df_derivados.loc[(df_derivados['año']==anio) & (df_derivados['especie']=='cacao') & (df_derivados['instrumento']=='prima_call'), 'valor'].values[0]
            strike_cacao = fw_cacao # ATMF
            
            fw_trm = df_derivados.loc[(df_derivados['año']==anio) & (df_derivados['especie']=='TRM') & (df_derivados['instrumento']=='futuro'), 'valor'].values[0]
            prima_call_trm = df_derivados.loc[(df_derivados['año']==anio) & (df_derivados['especie']=='TRM') & (df_derivados['instrumento']=='prima_call'), 'valor'].values[0]
            strike_trm = fw_trm # ATMF
            
            # --- ESTRATEGIA 1 (Cacao: Futuros | TRM: Opciones Call) ---
            pago_futuro_cacao_usd = (spot_cacao - fw_cacao) * toneladas_cubiertas
            dolares_a_cubrir_e1 = toneladas_cubiertas * fw_cacao
            pago_call_trm_cop = (max(spot_trm - strike_trm, 0) - prima_call_trm) * dolares_a_cubrir_e1
            
            flujo_cobertura_1 = ((pago_futuro_cacao_usd * spot_trm) + pago_call_trm_cop) / 1_000_000
            fcl_est1 = fcl_base + flujo_cobertura_1
            vpn_est1 += fcl_est1 / ((1 + tasa_descuento)**t)
            
            # --- ESTRATEGIA 2 (Cacao: Opciones Call | TRM: Futuros) ---
            pago_call_cacao_usd = (max(spot_cacao - strike_cacao, 0) - prima_call_cacao) * toneladas_cubiertas
            # Usamos el Strike para determinar el tamaño del contrato de TRM
            dolares_a_cubrir_e2 = toneladas_cubiertas * strike_cacao 
            pago_futuro_trm_cop = (spot_trm - fw_trm) * dolares_a_cubrir_e2
            
            flujo_cobertura_2 = ((pago_call_cacao_usd * spot_trm) + pago_futuro_trm_cop) / 1_000_000
            fcl_est2 = fcl_base + flujo_cobertura_2
            vpn_est2 += fcl_est2 / ((1 + tasa_descuento)**t)
            
        resultados_cobertura[esc_nombre] = {
            "VPN_Estrategia_1": vpn_est1,
            "VPN_Estrategia_2": vpn_est2
        }
        suma_vpn_est1 += vpn_est1
        suma_vpn_est2 += vpn_est2
        
    # --- PROMEDIOS Y CONCLUSIÓN ---
    promedio_e1 = suma_vpn_est1 / num_escenarios
    promedio_e2 = suma_vpn_est2 / num_escenarios
    
    resultados_cobertura["Promedios"] = {
        "Promedio_VPN_E1": promedio_e1,
        "Promedio_VPN_E2": promedio_e2
    }
    
    mejor_estrategia = "Estrategia 1" if promedio_e1 > promedio_e2 else "Estrategia 2"
    
    print("--- RESULTADOS DE COBERTURA ---")
    print(f"Promedio VPN Estrategia 1: {promedio_e1:,.2f} Millones COP")
    print(f"Promedio VPN Estrategia 2: {promedio_e2:,.2f} Millones COP")
    print(f"La estrategia más factible para la compañía es: {mejor_estrategia}")
    
    return resultados_cobertura


## Calculo FCL

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
        concepto: ingresos_proyectados
    })

    proyeccion.set_index("año", inplace = True)

    return proyeccion

def proyeccion_porcentual(valor, pcte, t):
    return valor * (1+ (pcte/100))**t

def calcular_EBIT(anio, proyeccion_ingresos, proyeccion_costos , gastos_admin, otros_ingresos_ops):
    t = anio - 2025
    pcte_admin = 14
    pcte_otros_ingresos_ops = 10

    EBIT = (
    (0.25 * proyeccion_ingresos.loc[anio])  #Ingresos operacionales, otros costos de venta , gastos de venta
    - proyeccion_costos.loc[anio] # Costo de ventas
    - proyeccion_porcentual(gastos_admin, pcte_admin, t) # Gastos de administracion
    + proyeccion_porcentual(otros_ingresos_ops, pcte_otros_ingresos_ops, t) # Otros ingresos netos operacionales
    )

    return EBIT

def calcular_CAPEX(anio, df_PPE):
    CAPEX = (
    ((1 + (8.5/100)) * df_PPE.loc[anio]) - df_PPE.loc[anio-1] 
    )
    return CAPEX

def calcular_FCL(tasa_impositiva, anio, df_PPE, proyeccion_ingresos, proyeccion_costos, gastos_admin, otros_ingresos_ops):
    FCL = (
    ((1-(tasa_impositiva/100))* calcular_EBIT(anio, proyeccion_ingresos, proyeccion_costos, gastos_admin, otros_ingresos_ops))
    + ((8.5/100) * df_PPE.loc[anio])
    - calcular_CAPEX(anio, df_PPE)     
    )

def calcular_flujos_y_vpn(tasa_impositiva, df_PPE, proyeccion_ingresos, proyeccion_costos, gastos_admin, otros_ingresos_ops):
    años_proyeccion = list(range(2026, 2031))
    flujos_caja = {}
    vpn = 0
    tasa_descuento = 0.07
    
    for anio in años_proyeccion:
        fcl = calcular_FCL(
            tasa_impositiva, 
            anio, 
            df_PPE, 
            proyeccion_ingresos, 
            proyeccion_costos, 
            gastos_admin, 
            otros_ingresos_ops
        )
        

        flujos_caja[anio] = fcl
        
        t = anio - 2025
        vpn += fcl / ((1 + tasa_descuento)**t)
    
    flujos_serie = pd.Series(flujos_caja, name="FCL")
        
    return flujos_serie, vpn

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