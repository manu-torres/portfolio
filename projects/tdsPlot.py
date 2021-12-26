#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.style.use('ggplot')

import math

#Esta funcion representa la matriz de confusion de un modelo mediante la TDS

def plot(Pa, Pfa, returnP = False, plot = True):
    Pe = 1 - Pa ; Prc = 1 - Pfa
    
    Za = norm.ppf(Pa, loc = 0, scale = 1)
    Ze = norm.ppf(Pe, loc = 0, scale = 1)
    Zfa = norm.ppf(Pfa, loc = 0, scale = 1)
    Zrc = norm.ppf(Prc, loc = 0, scale = 1)
    
    dPrima = Za - Zfa
    C = (-1 / 2) * (Za + Zfa) #Observador ideal
    beta = math.exp(dPrima * C)
    
    def GeneraDatos(meanR, meanS,
                            zFrom = -5, zTo = 5, zBy = 1001):
      Z = np.linspace(start = zFrom,
                      stop = zTo,
                      num = zBy)
      
      df = pd.DataFrame({
        "Zscore" : Z,
        "Zacum" : norm.cdf(Z),
        "Zden0" : norm.pdf(Z, loc = meanR, scale = 1),
        "Zden1" : norm.pdf(Z, loc = meanS, scale = 1)
      })
      
      df.columns = ["Z", "F(X)", "f(x/r)", "f(x/s)"]
      
      return(df)
    
    
    df = GeneraDatos(meanR = 0, 
                      meanS = 0 + dPrima)
    
    #Encontramos el valor de Z asociado a los valores 
    #f(x/s) y f(x/r) que hacen que la operacion 
    #f(x/s) / f(x/r) se aproxime mas a beta
    Temp = (df["f(x/s)"] / df["f(x/r)"]) - beta
    Xc = df["Z"][Temp.abs() == Temp.abs().min()].values[0]
    
    #Para obtener el punto en el que el observador ideal da respuesta tomamos beta = 1
    Temp = (df["f(x/s)"] / df["f(x/r)"]) - 1
    Xideal = df["Z"][Temp.abs() == Temp.abs().min()].values[0]
    
    DatosGrafico = df[["Z", "f(x/s)", "f(x/r)"]]
    DatosGrafico = DatosGrafico.set_index("Z", drop = True)
    
    if plot == True:
        fig, ax = plt.subplots()
        DatosGrafico["f(x/r)"].plot()
        DatosGrafico["f(x/s)"].plot()
        ax.axvline(x = Xideal, color = "green")
        ax.axvline(x = Xc, color = "black")
        
        handles, labels = ax.get_legend_handles_labels()
        handles.append(mpatches.Patch(color = "green", label = "Observador ideal"))
        handles.append(mpatches.Patch(color ="black", label = "Observador"))
        
        ax.legend(handles = handles)
    
    if returnP == True:
        Resultados = {
                    "ObsIdeal" : Xideal,
                    "Obs" : Xc,
                    "C" : C,
                    "dPrima" : dPrima,
                    "beta" : beta,
                    "Pa" : Pa,
                    "Pe" : Pe,
                    "Pfa" : Pfa,
                    "Prc" : Prc                    
                }
        
        for dict_value in Resultados:
            Resultados[dict_value] = round(Resultados[dict_value], 3)
        
        return(Resultados)
    
if __name__ == "__main__":
    #Probamos la funcion
    try:
        plt.close("all")
    except:
        pass

    P = plot(Pa = 0.7, Pfa = 0.1, 
             returnP = True)
    
    import json
    print(json.dumps(P, indent = 4))
    


