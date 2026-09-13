import numpy as np


def precio_futuro(S0, r, T, costo_almacenamiento=0):
    return S0 * np.exp((r + costo_almacenamiento) * T)

def precio_forward_fx(S0, r_domestica, r_extranjera, T):
    return S0 * np.exp((r_domestica - r_extranjera) * T)

def vpn_cobertura_nickel(
    spot_nickel,
    spot_fx,
    tasas_usd,
    tasas_krw,
    almacenamiento,
    toneladas
):
    costos = []

    for i, T in enumerate([1, 2, 3]):

        F_nickel = precio_futuro(
            S0=spot_nickel,
            r=tasas_usd[i],
            T=T,
            costo_almacenamiento=almacenamiento
        )

        F_fx = precio_forward_fx(
            S0=spot_fx,
            r_domestica=tasas_krw[i],
            r_extranjera=tasas_usd[i],
            T=T
        )

        costo = toneladas[i] * F_nickel * F_fx

        costos.append(costo)

    vpn = sum(
        costos[i] / (1 + tasas_krw[i]) ** (i + 1)
        for i in range(3)
    )

    return vpn