import streamlit as st
import pandas as pd
import numpy as np
import time
from main_simulation import KilnDigitalTwin

st.set_page_config(page_title="Cement Kiln Digital Twin", layout="wide")

st.title("🏭 Çimento Fabrikası Kontrol Odası (V2.0)")
st.markdown("---")

if 'kiln' not in st.session_state:
    st.session_state.kiln = KilnDigitalTwin()
    st.session_state.history = []

# --- SIDEBAR: KONTROLLER ---
st.sidebar.header("🕹️ Operatör Paneli")
feed_rate = st.sidebar.slider("Besleme Hızı (tph)", 50.0, 250.0, 120.0)
calciner_fuel = st.sidebar.slider("Kalsinatör Yakıtı (tph)", 0.0, 10.0, 4.5)
main_fuel = st.sidebar.slider("Ana Brülör Yakıtı (tph)", 0.0, 8.0, 3.2)
kiln_rpm = st.sidebar.slider("Fırın Devri (RPM)", 0.5, 5.0, 3.5)
fan_speed = st.sidebar.slider("ID Fan Hızı (%)", 0.0, 100.0, 75.0)
cooler_fans = st.sidebar.slider("Soğutucu Fanları (Hz)", 0.0, 50.0, 35.0)

current_controls = {
    'feed_rate': feed_rate, 'calciner_fuel': calciner_fuel,
    'main_fuel': main_fuel, 'kiln_rpm': kiln_rpm,
    'fan_speed': fan_speed * 2, 'cooler_fans': cooler_fans * 5
}

# --- HESAPLAMA ---
results = st.session_state.kiln.run_step(current_controls)
st.session_state.history.append(results)
df = pd.DataFrame(st.session_state.history).tail(100)

# --- METRİKLER ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Klinker (Burning Zone)", f"{results['clinker_temp']:.1f} °C")
m2.metric("Kalsinatör Çıkış", f"{results['T_calciner']:.1f} °C")
m3.metric("Tersiyer Hava", f"{results['T_tertiary']:.1f} °C")
m4.metric("Preheater (Baca)", f"{results['T_preheater']:.1f} °C")

# --- GRAFİKLER ---
st.markdown("### 📈 Canlı Analiz Trendleri")
col_high, col_low = st.columns(2)

with col_high:
    st.subheader("🔥 Yüksek Sıcaklık Bölgesi")
    st.line_chart(df[['clinker_temp', 'T_calciner']])
    st.caption("Klinker (1450-1550) ve Kalsinatör (850-950) Takibi")

with col_low:
    st.subheader("❄️ Isı Geri Kazanım & Baca")
    st.line_chart(df[['T_tertiary', 'T_preheater']])
    st.caption("Tersiyer (700-850) ve Baca Gazı (250-350) Takibi")

st.markdown("---")
b1, b2 = st.columns(2)
with b1:
    st.subheader("⚙️ Tork & Reaksiyon Verimi")
    st.line_chart(df[['kiln_torque', 'calcination']])
with b2:
    st.subheader("💨 Hava/Yakıt Oranı (Lambda)")
    st.line_chart(df[['lambda']])

time.sleep(0.05)
st.rerun()