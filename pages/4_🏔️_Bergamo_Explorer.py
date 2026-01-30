import streamlit as st
import folium
from streamlit_folium import st_folium

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Bergamo Mountain Explorer", layout="wide")

st.title("🏔️ Giro delle Montagne di Bergamo")
st.markdown("""
Questa mappa interattiva mostra **3 itinerari iconici** delle Alpi Orobie vicino a Bergamo.
Seleziona un percorso dalla barra laterale per vederne i dettagli sulla mappa.
""")

# --- DATI ITINERARI (Coordinate Approssimative per Demo) ---
# Nota: In un'app reale, useresti file .GPX precisi. Qui usiamo punti chiave per disegnare linee rette indicative.
itineraries = {
    "Canto Alto (Facile/Medio)": {
        "desc": "La montagna di casa. Vista spettacolare su Bergamo Alta e sulla pianura.",
        "start_point": [45.7369, 9.6622], # Sorisole
        "end_point": [45.7592, 9.6844],   # Croce Canto Alto
        "color": "green",
        "zoom": 13,
        "type": "Escursionismo"
    },
    "Rifugio Laghi Gemelli (Medio/Difficile)": {
        "desc": "Il cuore delle Orobie. Un ambiente alpino severo e bellissimo con laghi in quota.",
        "start_point": [46.0235, 9.7925], # Carona
        "end_point": [45.9936, 9.8052],   # Rifugio Laghi Gemelli
        "color": "blue",
        "zoom": 13,
        "type": "Trekking Alpino"
    },
    "Monte Resegone (Medio)": {
        "desc": "La montagna descritta da Manzoni. Salita dal versante bergamasco (Brumano).",
        "start_point": [45.8544, 9.5033], # Brumano
        "end_point": [45.8669, 9.4897],   # Punta Cermenati (Vetta)
        "color": "red",
        "zoom": 14,
        "type": "Escursionismo Esperto"
    }
}

# --- SIDEBAR ---
with st.sidebar:
    st.header("🥾 Scegli il tuo Tragitto")
    selected_route_name = st.radio("Itinerari disponibili:", list(itineraries.keys()))
    
    route_data = itineraries[selected_route_name]
    
    st.info(f"**Descrizione:** {route_data['desc']}")
    st.warning(f"**Tipo:** {route_data['type']}")
    st.markdown("---")
    st.caption("Coordinate Approssimative per visualizzazione demo.")

# --- MAPPA ---
# Coordinate Centro Bergamo
bergamo_center = [45.6983, 9.6773]

# Creazione Mappa Base
m = folium.Map(location=bergamo_center, zoom_start=10, tiles="OpenStreetMap")

# Aggiungi Marker su Bergamo Città
folium.Marker(
    bergamo_center, 
    popup="Bergamo Città", 
    icon=folium.Icon(color="black", icon="home")
).add_to(m)

# Disegna TUTTI i percorsi o SOLO quello selezionato? 
# Qui disegniamo quello selezionato con focus, e gli altri in grigio.

for name, data in itineraries.items():
    is_active = (name == selected_route_name)
    color = data["color"] if is_active else "gray"
    opacity = 1.0 if is_active else 0.5
    weight = 5 if is_active else 3
    
    # Linea del tragitto (Semplificata: Start -> End)
    folium.PolyLine(
        locations=[data["start_point"], data["end_point"]],
        color=color,
        weight=weight,
        opacity=opacity,
        tooltip=name
    ).add_to(m)
    
    # Marker Partenza
    folium.Marker(
        data["start_point"],
        popup=f"Partenza: {name}",
        icon=folium.Icon(color="green" if is_active else "lightgray", icon="play")
    ).add_to(m)

    # Marker Arrivo
    folium.Marker(
        data["end_point"],
        popup=f"Arrivo: {name}",
        icon=folium.Icon(color="red" if is_active else "lightgray", icon="flag")
    ).add_to(m)

# Seleziona lo zoom sull'itinerario attivo
if route_data:
    # Calcola il punto medio per centrare la mappa
    mid_lat = (route_data["start_point"][0] + route_data["end_point"][0]) / 2
    mid_lon = (route_data["start_point"][1] + route_data["end_point"][1]) / 2
    m.location = [mid_lat, mid_lon]
    m.zoom_start = route_data["zoom"]

# Visualizza Mappa in Streamlit
st_folium(m, width=900, height=600)
