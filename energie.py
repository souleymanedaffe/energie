import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.offsets import MonthEnd
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")

# ✅ Titre principal bien visible
st.markdown("<h1 style='font-size: 36px;'> Prévision de la Consommation d'énergie par région française</h1>", unsafe_allow_html=True)

# Chargement des données
if 'dataset' not in st.session_state:
    try:
        data_path = "energie.csv"
        data_csv = pd.read_csv(data_path, sep=";")
        data_csv['Date'] = pd.to_datetime(data_csv['Mois']) + MonthEnd()
        data_csv = data_csv[['Territoire', 'Date', 'Consommation totale']]
        st.session_state.dataset = data_csv.copy()
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de 'energie.csv' : {e}")

# Interface utilisateur
if not st.session_state.get("dataset", pd.DataFrame()).empty:
    territoires = st.session_state.dataset['Territoire'].unique()
    selected_territoire = st.selectbox("📍 Sélectionnez un territoire", territoires)

    data = st.session_state.dataset.query("Territoire == @selected_territoire").set_index("Date").sort_index()


    available_dates = [date for i, date in enumerate(data.index.unique()) if i >= 23]
    selected_start_date = st.selectbox("📅 Sélectionnez la date de départ", options=available_dates, format_func=lambda x: x.strftime('%Y-%m'))

    forecast_steps = st.slider("Le nombre de mois à prévoir", 1, 24, 12)
    forecast_trigger = st.button(" Lancer la prévision")



    if forecast_trigger:
        data_filtered = data[data.index <= selected_start_date]

        if len(data_filtered) >= 24:
            model = SARIMAX(data_filtered['Consommation totale'],
                            order=(1, 1, 1),
                            seasonal_order=(1, 1, 1, 12),
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

            # 📈 Graphe principal interactif
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=data_filtered.index,
                y=data_filtered['Consommation totale'],
                mode='lines',
                name='Historique',
                line=dict(color='blue')
            ))

            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['Prévision'],
                mode='lines+markers',
                name='Prévision',
                line=dict(color='orange')
            ))

            fig.add_trace(go.Scatter(
                x=forecast_df.index.tolist() + forecast_df.index[::-1].tolist(),
                y=forecast_df['Borne supérieure'].tolist() + forecast_df['Borne inférieure'][::-1].tolist(),
                fill='toself',
                fillcolor='rgba(255,165,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name='Intervalle de confiance'
            ))

            fig.update_layout(
                title=f"Prévision de la consommation d'énergie – {selected_territoire}",
                xaxis_title="Date",
                yaxis_title="Consommation (MWh)",
                legend_title="Légende",
                hovermode="x unified",
                font=dict(size=16),
                title_font=dict(size=20)
            )

            st.plotly_chart(fig, use_container_width=True)

            # 📊 Graphe barres : Historique + Prévision
            st.subheader("Historique récent et Prévisions (barres groupées)")
            recent_hist = data_filtered.tail(6)['Consommation totale'].rename("Historique")
            combined_df = pd.concat([recent_hist, forecast_df['Prévision']])

            fig_bar = go.Figure()

            fig_bar.add_trace(go.Bar(
                x=recent_hist.index,
                y=recent_hist.values,
                name='Historique',
                marker_color='steelblue'
            ))

            fig_bar.add_trace(go.Bar(
                x=forecast_df.index,
                y=forecast_df['Prévision'],
                name='Prévision',
                marker_color='orange'
            ))

            # Ligne verticale avec annotation
            transition_date = forecast_df.index[0]
            fig_bar.add_shape(
                type="line",
                x0=transition_date,
                y0=0,
                x1=transition_date,
                y1=max(combined_df.values) * 1.1,
                line=dict(color="gray", width=2, dash="dash")
            )

            fig_bar.add_annotation(
                x=transition_date,
                y=max(combined_df.values) * 1.05,
                text="Début des prévisions",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )

            fig_bar.update_layout(
                title="Consommation récente vs prévision",
                xaxis_title="Date",
                yaxis_title="Consommation (MWh)",
                barmode='group',
                xaxis_tickformat='%Y-%m',
                font=dict(size=16),
                title_font=dict(size=20)
            )

            st.plotly_chart(fig_bar)

            # 📋 Tableau
            st.subheader("Détail des prévisions")
            st.dataframe(forecast_df.round(2))

        else:
            st.warning("⚠️ Il faut au moins 24 mois de données historiques avant la date sélectionnée.")
else:
    st.info("ℹ️ Aucune donnée disponible. Vérifiez le fichier `energie.csv`.")

