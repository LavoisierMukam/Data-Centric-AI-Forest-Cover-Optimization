import numpy as np
import pandas as pd


def transform_data(X_raw):
    """
    Entrée : X_raw (DataFrame Pandas) contenant les features brutes.
    Sortie : DataFrame Pandas contenant les features transformées.
    """

    X_out = X_raw.copy()

    #Sécurisation des valeurs numériques
    num_cols = X_out.select_dtypes(include=[np.number]).columns
    X_out[num_cols] = X_out[num_cols].fillna(0)
    X_out[num_cols] = X_out[num_cols].replace([np.inf, -np.inf], 0)

    for col in ("Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"):
        if col in X_out.columns:
            X_out[col] = X_out[col].clip(0, 255)

    #Aspect (trigonométrie)
    if "Aspect" in X_out.columns:
        aspect_rad = np.deg2rad(X_out["Aspect"] % 360)
        X_out["Aspect_sin"] = np.sin(aspect_rad)
        X_out["Aspect_cos"] = np.cos(aspect_rad)

    #Hydrologie ( distances euclidiennes et log )
    if {"Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology"}.issubset(X_out.columns):
        h = X_out["Horizontal_Distance_To_Hydrology"]
        v = X_out["Vertical_Distance_To_Hydrology"]
        d = np.sqrt(h**2 + v**2)

        X_out["Hydro_Distance_Euclid"] = d
        X_out["Hydro_Distance_log"] = np.log1p(d)
        X_out["Hydro_Vertical_Abs"] = np.abs(v)

    #Distances longues (log)
    for col in ("Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points"):
        if col in X_out.columns:
            X_out[f"{col}_log"] = np.log1p(X_out[col])

    #Interactions terrain élévation/pente
    if {"Elevation", "Slope"}.issubset(X_out.columns):
        X_out["Elev_x_Slope"] = X_out["Elevation"] * X_out["Slope"]
        X_out["Elev_bin_100m"] = (X_out["Elevation"] // 100).astype(int)

    #Hillshade synthèse ( moyenne et amplitude )
    hs = ("Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm")
    if set(hs).issubset(X_out.columns):
        X_out["Hillshade_mean"] = X_out[list(hs)].mean(axis=1)
        X_out["Hillshade_range"] = X_out["Hillshade_Noon"] - (
            X_out["Hillshade_9am"] + X_out["Hillshade_3pm"]
        ) / 2

    #Wilderness Area
    w_cols = [c for c in X_out.columns if c.startswith("Wilderness_Area_")]
    if w_cols:
        X_out["Wilderness_Code"] = X_out[w_cols].values.argmax(axis=1)
        X_out.drop(columns=w_cols, inplace=True)

    #Soil Type
    s_cols = [c for c in X_out.columns if c.startswith("Soil_Type_")]
    if s_cols:
        X_out["Soil_Code"] = X_out[s_cols].values.argmax(axis=1)
        X_out.drop(columns=s_cols, inplace=True)

    return X_out