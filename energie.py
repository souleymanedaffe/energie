import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.offsets import MonthEnd
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

st.title("Prévision de la Consommation d'énergie par région française")

# Chargement automatique d'un fichier local avec le bon séparateur
if 'dataset' not in st.session_state:
    try:
        data_path = "energie.csv"
        data_csv = pd.read_csv(data_path, sep=";")
        data_csv['Date'] = pd.to_datetime(data_csv['Mois']) + MonthEnd()
        data_csv = data_csv[['Territoire', 'Date', 'Consommation totale']]
        st.session_state.dataset = data_csv.copy()
        st.success("✅ Données chargées automatiquement depuis 'energie.csv'")
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de 'energie.csv' : {e}")

# Interface de prévision
if not st.session_state.get("dataset", pd.DataFrame()).empty:
    territoires = st.session_state.dataset['Territoire'].unique()
    selected_territoire = st.selectbox("Choisissez un territoire pour la prévision", territoires)

    data = st.session_state.dataset.query("Territoire == @selected_territoire").set_index("Date").sort_index()

    # Sélection des dates valides (ayant au moins 24 mois d'historique avant)
    available_dates = [date for i, date in enumerate(data.index.unique()) if i >= 23]
    selected_start_date = st.selectbox("Sélectionnez la date de départ pour la prévision", options=available_dates, format_func=lambda x: x.strftime('%Y-%m'))

    forecast_steps = st.slider("Nombre de mois à prévoir", 1, 24, 12)
    forecast_trigger = st.button("📈 Appeler la prédiction des mois prochains")

    if forecast_trigger:
        data_filtered = data[data.index <= selected_start_date]

        if len(data_filtered) >= 24:
            model = SARIMAX(data_filtered['Consommation totale'],
                            order=(1,1,1),
                            seasonal_order=(1,1,1,12),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            results = model.fit()

            forecast = results.get_forecast(steps=forecast_steps)
            forecast_index = pd.date_range(data_filtered.index[-1] + MonthEnd(), periods=forecast_steps, freq='M')

            forecast_df = pd.DataFrame({
                'Prévision': forecast.predicted_mean,
                'Borne inférieure': forecast.conf_int().iloc[:, 0],
                'Borne supérieure': forecast.conf_int().iloc[:, 1]
            }, index=forecast_index)

            st.subheader(f"Prévision pour {selected_territoire} à partir de {forecast_index[0].strftime('%Y-%m')}")
            fig, ax = plt.subplots(figsize=(10, 5))
            data_filtered['Consommation totale'].plot(ax=ax, label='Historique')
            forecast_df['Prévision'].plot(ax=ax, label='Prévision')
            ax.fill_between(forecast_df.index, forecast_df['Borne inférieure'], forecast_df['Borne supérieure'], color='gray', alpha=0.3)
            plt.legend()
            st.pyplot(fig)

            st.subheader("Prévision sous forme de graphique en barres")
            st.bar_chart(forecast_df['Prévision'])

            st.write("Tableau des prévisions")
            st.dataframe(forecast_df.round(2))
        else:
            st.warning("Le territoire sélectionné doit contenir au moins 24 mois de données avant la date choisie.")
else:
    st.info("Aucune donnée disponible pour le moment.")
