"""
Galilei — Dashboard de Mercado Eléctrico Chileno
Streamlit app con Prophet, agente LLM con tools, y despliegue en EC2 con boto3.
"""

import streamlit as st
import pandas as pd
import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
import anthropic

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Galilei — Mercado Eléctrico",
    page_icon="⚡",
    layout="wide",
)

# ─────────────────────────────────────────────
# CSS PERSONALIZADO
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace;
}

.stApp {
    background: #0a0f1e;
    color: #e0e8ff;
}

.metric-box {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}

.anomaly-alert {
    background: #2d1515;
    border-left: 3px solid #ef4444;
    padding: 0.5rem 1rem;
    margin: 0.25rem 0;
    border-radius: 0 4px 4px 0;
    font-size: 0.85rem;
}

.warning-alert {
    background: #1a1a00;
    border-left: 3px solid #f59e0b;
    padding: 0.5rem 1rem;
    margin: 0.25rem 0;
    border-radius: 0 4px 4px 0;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATOS Y PIPELINE (igual que el notebook)
# ─────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    precios_raw = pd.DataFrame({
        'fecha': ['2024-01-01','2024-01-01','2024-01-02','2024-01-02','2024-01-03','2024-01-03'],
        'hora': [8, 20, 8, 20, 8, 20],
        'barra': ['Quillota','Alto Jahuel','Quillota','Alto Jahuel','Quillota','Alto Jahuel'],
        'precio_spot': [45.2, 51.8, None, 49.3, 60.1, None],
        'moneda': ['USD','USD','USD','USD','USD','USD']
    })

    generacion_raw = pd.DataFrame({
        'timestamp': ['2024-01-01 08:00','2024-01-01 20:00','2024-01-02 08:00',
                      '2024-01-02 20:00','2024-01-03 08:00','2024-01-03 20:00'],
        'central': ['Los Cóndores','Los Cóndores','Volcán','Volcán','Los Cóndores','Volcán'],
        'tipo': ['hidro','hidro','eólica','eólica','hidro','eólica'],
        'mwh_generados': [320, 280, 150, 170, 410, 'ERROR'],
        'barra_conexion': ['QUILLOTA-123','QUILLOTA-123','ALTO_JAHUEL-456',
                           'ALTO_JAHUEL-456','QUILLOTA-123','ALTO_JAHUEL-456']
    })

    tipo_cambio_json = '[{"fecha":"2024-01-01","usd_clp":895.0},{"fecha":"2024-01-02","usd_clp":901.5},{"fecha":"2024-01-03","usd_clp":899.0}]'

    # Limpieza
    precios = precios_raw.copy()
    precios['fecha'] = pd.to_datetime(precios['fecha'])

    gen = generacion_raw.copy()
    gen['mwh_generados'] = pd.to_numeric(gen['mwh_generados'], errors='coerce')
    gen['timestamp'] = pd.to_datetime(gen['timestamp'])
    gen['fecha'] = gen['timestamp'].dt.normalize()
    gen['hora'] = gen['timestamp'].dt.hour
    gen['barra'] = (
        gen['barra_conexion']
        .str.replace(r'-\d+$', '', regex=True)
        .str.replace('_', ' ')
        .str.title()
    )

    tc = pd.DataFrame(json.loads(tipo_cambio_json))
    tc['fecha'] = pd.to_datetime(tc['fecha'])

    # Integración
    mercado = gen.merge(
        precios[['fecha','hora','barra','precio_spot','moneda']],
        on=['fecha','hora','barra'], how='left'
    ).merge(tc, on='fecha', how='left')

    mercado['ingreso_clp'] = mercado['mwh_generados'] * mercado['precio_spot'] * mercado['usd_clp']
    mercado['periodo'] = mercado['hora'].map({8: 'peak', 20: 'off-peak'})

    return mercado, precios, gen, tc

@st.cache_resource
def get_db(mercado):
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    m = mercado.copy()
    m['fecha'] = m['fecha'].astype(str)
    m.to_sql('mercado', conn, index=False, if_exists='replace')
    return conn

# ─────────────────────────────────────────────
# INTERPOLACIÓN Y PROPHET
# ─────────────────────────────────────────────
def generar_serie_expandida(mercado, n_dias=60):
    """
    Con solo 6 observaciones, Prophet no tiene sentido directo.
    Estrategia: interpolar la serie original a escala horaria y luego
    simular variación realista para los 60 días usando los patrones
    observados (diferencial peak/off-peak, variabilidad por barra).
    
    LIMITACIÓN EXPLÍCITA: esto es una SIMULACIÓN basada en pocos datos,
    no un forecast real. Se usa para demostrar la arquitectura del pipeline.
    En producción necesitaríamos meses de datos del CEN/Coordinador.
    """
    import warnings
    try:
        from prophet import Prophet
        PROPHET_AVAILABLE = True
    except ImportError:
        PROPHET_AVAILABLE = False

    resultados = {}
    
    for barra in mercado['barra'].unique():
        df_barra = mercado[mercado['barra'] == barra].sort_values('timestamp')
        precios_validos = df_barra['precio_spot'].dropna()
        
        if len(precios_validos) < 2:
            resultados[barra] = None
            continue
        
        # Paso 1: interpolar entre los puntos conocidos para tener más densidad
        fechas_conocidas = df_barra['timestamp'].tolist()
        precios_conocidos = df_barra['precio_spot'].tolist()
        
        # Crear serie horaria expandida con interpolación + ruido controlado
        fecha_inicio = df_barra['timestamp'].min()
        fecha_fin = fecha_inicio + timedelta(days=n_dias)
        serie_fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='12h')
        
        # Base: interpolación lineal de los puntos reales
        media = precios_validos.mean()
        std = precios_validos.std() if len(precios_validos) > 1 else media * 0.1
        tendencia = (precios_validos.iloc[-1] - precios_validos.iloc[0]) / len(precios_validos)
        
        np.random.seed(42)
        precios_sim = []
        for i, fecha in enumerate(serie_fechas):
            # Tendencia lineal + componente diario (peak vs off-peak) + ruido
            base = media + tendencia * i
            componente_hora = 3.0 if fecha.hour < 12 else -2.0  # peak diurno
            ruido = np.random.normal(0, std * 0.3)
            precio = max(10, base + componente_hora + ruido)
            precios_sim.append(precio)
        
        df_prophet = pd.DataFrame({'ds': serie_fechas, 'y': precios_sim})
        
        if PROPHET_AVAILABLE and len(df_prophet) >= 10:
            # Entrenar Prophet sobre la serie expandida
            modelo = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=True,
                changepoint_prior_scale=0.1
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                modelo.fit(df_prophet)
            
            futuro = modelo.make_future_dataframe(periods=30, freq='12h')
            forecast = modelo.predict(futuro)
            resultados[barra] = {
                'historico': df_prophet,
                'forecast': forecast[['ds','yhat','yhat_lower','yhat_upper']],
                'modelo': 'Prophet (datos interpolados)',
                'advertencia': '⚠️ Serie generada por interpolación — no es forecast real'
            }
        else:
            # Fallback: regresión lineal simple si no hay Prophet
            from numpy.polynomial import polynomial as P
            x = np.arange(len(df_prophet))
            coefs = np.polyfit(x, df_prophet['y'], 1)
            x_futuro = np.arange(len(df_prophet), len(df_prophet) + 30)
            y_futuro = np.polyval(coefs, x_futuro)
            fechas_futuro = pd.date_range(start=serie_fechas[-1], periods=30, freq='12h')
            
            resultados[barra] = {
                'historico': df_prophet,
                'forecast': pd.DataFrame({'ds': fechas_futuro, 'yhat': y_futuro,
                                          'yhat_lower': y_futuro * 0.9, 'yhat_upper': y_futuro * 1.1}),
                'modelo': 'Regresión lineal (Prophet no instalado)',
                'advertencia': '⚠️ Serie generada por interpolación — no es forecast real'
            }
    
    return resultados

# ─────────────────────────────────────────────
# TOOLS DEL AGENTE
# ─────────────────────────────────────────────
TOOLS_DEF = [
    {
        "name": "consulta_sql",
        "description": "Ejecuta una consulta SQL sobre la tabla 'mercado' en SQLite. Úsala cuando el usuario pregunte sobre precios, generación, centrales o barras. La tabla tiene columnas: fecha, hora, periodo, central, tipo, barra, mwh_generados, precio_spot, moneda, usd_clp, ingreso_clp, timestamp, barra_conexion.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query SQL válida para SQLite"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "diagnostico_datos",
        "description": "Corre validaciones de calidad sobre el dataset. Retorna un resumen de nulos, anomalías y cobertura. Úsala cuando el usuario pregunte sobre la calidad de los datos o problemas encontrados.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "resumen_mercado",
        "description": "Genera un resumen ejecutivo del mercado: precio promedio por barra, generación por tipo, alertas de calidad. Úsala cuando el usuario pida un resumen general o reporte del mercado.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "explicar_modelo",
        "description": "Explica el modelo de forecasting: qué datos usó, qué técnica aplicó, cuáles son sus limitaciones y métricas de confianza. Úsala cuando el usuario pregunte sobre el modelo predictivo o su confiabilidad.",
        "input_schema": {
            "type": "object",
            "properties": {
                "barra": {"type": "string", "description": "Nombre de la barra: 'Quillota' o 'Alto Jahuel'"}
            },
            "required": ["barra"]
        }
    },
    {
        "name": "buscar_precio_commodity",
        "description": "Busca en la web el precio actual de commodities relacionados al sector eléctrico: gas natural, carbón, petróleo, o el precio spot del SEN. Úsala para contextualizar los precios locales con el mercado global.",
        "input_schema": {
            "type": "object",
            "properties": {
                "commodity": {"type": "string", "description": "Commodity a buscar: 'gas natural', 'carbon', 'petroleo', 'sen spot'"}
            },
            "required": ["commodity"]
        }
    }
]

def ejecutar_tool(tool_name, tool_input, conn, mercado):
    """Ejecuta la tool solicitada por el agente y retorna el resultado."""
    
    if tool_name == "consulta_sql":
        try:
            resultado = pd.read_sql(tool_input["query"], conn)
            if resultado.empty:
                return "La consulta no retornó resultados."
            return resultado.to_string(index=False)
        except Exception as e:
            return f"Error en SQL: {str(e)}"
    
    elif tool_name == "diagnostico_datos":
        lines = []
        nulos_precio = mercado['precio_spot'].isna().sum()
        nulos_mwh = mercado['mwh_generados'].isna().sum()
        lines.append(f"precio_spot: {nulos_precio} nulos de {len(mercado)} filas")
        lines.append(f"mwh_generados: {nulos_mwh} nulos (incluye 1 'ERROR' convertido a NaN)")
        lines.append(f"Barras cubiertas: {mercado['barra'].nunique()}")
        lines.append(f"Centrales: {', '.join(mercado['central'].unique())}")
        lines.append(f"Rango fechas: {mercado['fecha'].min().date()} a {mercado['fecha'].max().date()}")
        ingreso_calculable = mercado['ingreso_clp'].notna().sum()
        lines.append(f"Filas con ingreso calculable: {ingreso_calculable}/{len(mercado)}")
        return "\n".join(lines)
    
    elif tool_name == "resumen_mercado":
        try:
            precio_barra = pd.read_sql(
                "SELECT barra, ROUND(AVG(precio_spot),2) AS precio_avg_usd FROM mercado GROUP BY barra",
                conn
            )
            gen_tipo = pd.read_sql(
                "SELECT tipo, ROUND(SUM(mwh_generados),1) AS mwh_total FROM mercado WHERE mwh_generados IS NOT NULL GROUP BY tipo",
                conn
            )
            ingreso = pd.read_sql(
                "SELECT central, ROUND(SUM(ingreso_clp)/1e6, 2) AS ingreso_mm_clp FROM mercado WHERE ingreso_clp IS NOT NULL GROUP BY central",
                conn
            )
            resumen = f"""RESUMEN DEL MERCADO:
Precio promedio por barra (USD/MWh):
{precio_barra.to_string(index=False)}

Generación por tipo (MWh):
{gen_tipo.to_string(index=False)}

Ingreso estimado por central (MM CLP):
{ingreso.to_string(index=False)}

Alertas: 2 precios nulos, 1 dato de generación inválido ('ERROR')."""
            return resumen
        except Exception as e:
            return f"Error generando resumen: {e}"
    
    elif tool_name == "explicar_modelo":
        barra = tool_input.get("barra", "Quillota")
        return f"""Modelo para barra '{barra}':
- Técnica: Prophet (Facebook) con datos interpolados
- Datos originales: 2-3 observaciones reales de precio
- Expansión: serie de 60 días generada por interpolación lineal + componente diario + ruido controlado
- ADVERTENCIA CRÍTICA: Con solo 2-3 puntos reales, este modelo NO tiene valor predictivo real. Es una demostración arquitectural.
- Para un forecast confiable se necesitan mínimo 2-3 meses de datos históricos horarios del CEN/Coordinador.
- Métricas de ajuste no aplican: la serie de entrenamiento fue generada, no observada.
- Uso válido: demostrar el pipeline técnico y la integración del modelo en la plataforma."""
    
    elif tool_name == "buscar_precio_commodity":
        commodity = tool_input.get("commodity", "gas natural")
        return f"""[Simulación — en producción usaría web_search tool de la API de Anthropic]
Commodity buscado: {commodity}
En producción, este tool haría una búsqueda web para traer el precio actual de {commodity}
y lo contextualizaría con los precios spot del mercado chileno. 
Por ejemplo: "El gas natural está a X USD/MMBtu, lo que históricamente correlaciona con 
precios spot en el SEN de Y-Z USD/MWh"."""
    
    return f"Tool '{tool_name}' no reconocida."

# ─────────────────────────────────────────────
# INTERFAZ PRINCIPAL
# ─────────────────────────────────────────────
mercado, precios, gen, tc = cargar_datos()
conn = get_db(mercado)

st.markdown("# ⚡ Galilei — Mercado Eléctrico Chile")
st.markdown("*Dashboard de análisis · Datos: SEN (dataset demo)*")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Exploración", "🔮 Forecasting", "💬 Agente SQL", "🔍 Calidad de Datos"])

# ─────────────────────────────────────────────
# TAB 1: EXPLORACIÓN
# ─────────────────────────────────────────────
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        precio_prom = mercado['precio_spot'].mean()
        st.metric("Precio Promedio", f"{precio_prom:.1f} USD/MWh")
    with col2:
        mwh_total = mercado['mwh_generados'].sum()
        st.metric("Generación Total", f"{mwh_total:.0f} MWh")
    with col3:
        ingreso_total = mercado['ingreso_clp'].sum() / 1e6
        st.metric("Ingreso Estimado", f"{ingreso_total:.1f} MM CLP")
    with col4:
        n_nulos = mercado['precio_spot'].isna().sum() + mercado['mwh_generados'].isna().sum()
        st.metric("Datos Faltantes", f"{n_nulos} registros", delta="2 precios + 1 MWh", delta_color="inverse")

    st.markdown("### Precios Spot por Barra")
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        
        df_plot = mercado.dropna(subset=['precio_spot']).copy()
        df_plot['datetime'] = df_plot['timestamp']
        
        fig = go.Figure()
        for barra in df_plot['barra'].unique():
            d = df_plot[df_plot['barra'] == barra]
            fig.add_trace(go.Scatter(
                x=d['datetime'], y=d['precio_spot'],
                mode='lines+markers', name=barra,
                line=dict(width=2), marker=dict(size=8)
            ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0a0f1e',
            plot_bgcolor='#111827',
            xaxis_title='Fecha/Hora',
            yaxis_title='Precio (USD/MWh)',
            legend=dict(bgcolor='#111827'),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Generación por Central")
            gen_central = mercado.groupby('central')['mwh_generados'].sum().reset_index()
            fig2 = px.bar(gen_central, x='central', y='mwh_generados',
                          color='central', template='plotly_dark',
                          labels={'mwh_generados': 'MWh', 'central': 'Central'})
            fig2.update_layout(paper_bgcolor='#0a0f1e', plot_bgcolor='#111827', showlegend=False, height=300)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col_b:
            st.markdown("### Peak vs Off-Peak")
            peak_df = mercado.groupby(['barra','periodo'])['precio_spot'].mean().reset_index()
            fig3 = px.bar(peak_df, x='barra', y='precio_spot', color='periodo',
                          barmode='group', template='plotly_dark',
                          labels={'precio_spot': 'Precio USD/MWh', 'barra': 'Barra'})
            fig3.update_layout(paper_bgcolor='#0a0f1e', plot_bgcolor='#111827', height=300)
            st.plotly_chart(fig3, use_container_width=True)
    except ImportError:
        st.info("Instala plotly para ver los gráficos: `pip install plotly`")
        st.dataframe(mercado[['fecha','hora','barra','precio_spot','mwh_generados','ingreso_clp']])

# ─────────────────────────────────────────────
# TAB 2: FORECASTING
# ─────────────────────────────────────────────
with tab2:
    st.warning("""
    ⚠️ **Aviso importante:** Este dataset tiene solo 6 observaciones reales.
    El modelo Prophet se entrena sobre una **serie interpolada y simulada** de 60 días
    construida a partir de los patrones observados (media, tendencia, diferencial peak/off-peak).
    
    **Esto es una demostración arquitectural**, no un forecast válido.  
    En producción se usarían meses de datos horarios del CEN/Coordinador.
    """)
    
    st.markdown("### Configuración del Modelo")
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        barra_sel = st.selectbox("Barra", mercado['barra'].unique())
    with col_cfg2:
        n_dias = st.slider("Días a extrapolar para entrenamiento", 30, 90, 60)
    
    if st.button("🔮 Entrenar modelo y proyectar", type="primary"):
        with st.spinner("Interpolando serie y entrenando Prophet..."):
            resultados = generar_serie_expandida(mercado, n_dias=n_dias)
            
            if barra_sel in resultados and resultados[barra_sel] is not None:
                res = resultados[barra_sel]
                
                try:
                    import plotly.graph_objects as go
                    hist = res['historico']
                    fc = res['forecast']
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist['ds'], y=hist['y'],
                        name='Serie interpolada (entrenamiento)',
                        line=dict(color='#60a5fa', width=1.5),
                        opacity=0.7
                    ))
                    fig.add_trace(go.Scatter(
                        x=fc['ds'], y=fc['yhat'],
                        name='Proyección Prophet',
                        line=dict(color='#f59e0b', width=2, dash='dot')
                    ))
                    fig.add_trace(go.Scatter(
                        x=list(fc['ds']) + list(fc['ds'][::-1]),
                        y=list(fc['yhat_upper']) + list(fc['yhat_lower'][::-1]),
                        fill='toself',
                        fillcolor='rgba(245,158,11,0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Intervalo de confianza'
                    ))
                    
                    # Puntos reales originales
                    datos_reales = mercado[mercado['barra'] == barra_sel].dropna(subset=['precio_spot'])
                    fig.add_trace(go.Scatter(
                        x=datos_reales['timestamp'],
                        y=datos_reales['precio_spot'],
                        mode='markers',
                        name='Datos reales (3 obs)',
                        marker=dict(color='#ef4444', size=12, symbol='diamond')
                    ))
                    
                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='#0a0f1e',
                        plot_bgcolor='#111827',
                        title=f"Barra {barra_sel} — {res['modelo']}",
                        xaxis_title='Fecha',
                        yaxis_title='Precio (USD/MWh)',
                        height=450
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.info(res['advertencia'])
                    
                    st.markdown("**Estadísticas de la proyección:**")
                    stats_fc = res['forecast']
                    cols = st.columns(3)
                    cols[0].metric("Precio Proyectado Promedio", f"{stats_fc['yhat'].mean():.1f} USD/MWh")
                    cols[1].metric("Mínimo Proyectado", f"{stats_fc['yhat'].min():.1f} USD/MWh")
                    cols[2].metric("Máximo Proyectado", f"{stats_fc['yhat'].max():.1f} USD/MWh")
                    
                except ImportError:
                    st.dataframe(res['forecast'].head(10))
            else:
                st.error(f"No hay suficientes datos para la barra '{barra_sel}'.")

# ─────────────────────────────────────────────
# TAB 3: AGENTE SQL
# ─────────────────────────────────────────────
with tab3:
    st.markdown("### 🤖 Agente de Mercado Eléctrico")
    st.markdown("Pregúntame sobre precios, generación, centrales o pídeme un resumen del mercado.")
    
    # Ejemplos de preguntas
    with st.expander("💡 Preguntas de ejemplo"):
        ejemplos = [
            "¿Cuál fue el precio promedio por barra?",
            "¿Cuánto generó la central Los Cóndores en total?",
            "Dame un resumen del mercado",
            "¿Cómo está la calidad de los datos?",
            "¿Qué tan confiable es el modelo para Quillota?",
            "¿Cómo está el precio del gas natural hoy?",
            "¿Cuál barra tuvo mayor ingreso estimado?",
        ]
        for e in ejemplos:
            if st.button(e, key=f"btn_{e[:20]}"):
                st.session_state['pregunta_agente'] = e

    api_key = st.text_input("API Key de Anthropic", type="password", 
                             help="Necesaria para activar el agente. Obtén la tuya en console.anthropic.com")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Mostrar historial
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    pregunta = st.chat_input("Pregunta sobre el mercado eléctrico...")
    if 'pregunta_agente' in st.session_state:
        pregunta = st.session_state.pop('pregunta_agente')
    
    if pregunta:
        st.session_state.chat_history.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)
        
        if not api_key:
            respuesta = "⚠️ Ingresa tu API Key de Anthropic para activar el agente."
            st.session_state.chat_history.append({"role": "assistant", "content": respuesta})
            with st.chat_message("assistant"):
                st.markdown(respuesta)
        else:
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    try:
                        client = anthropic.Anthropic(api_key=api_key)
                        
                        system_prompt = """Eres un analista experto del mercado eléctrico chileno trabajando en Galilei.
Tienes acceso a datos del Sistema Eléctrico Nacional (SEN): precios spot por barra, generación por central y tipo de cambio USD/CLP.

Cuando el usuario haga una pregunta:
1. Analiza qué tools necesitas usar
2. Úsalas (una o varias en secuencia si es necesario)
3. Interpreta los resultados con contexto del mercado eléctrico chileno
4. Responde de forma clara y directa

Si la pregunta involucra SQL, genera queries limpias y bien estructuradas.
Si el usuario pide un resumen, usa la tool resumen_mercado.
Si pregunta sobre calidad de datos, usa diagnostico_datos.
Si pregunta sobre el modelo o forecasting, usa explicar_modelo.
Si pregunta sobre precios de commodities actuales, usa buscar_precio_commodity.

Responde siempre en español. Sé conciso pero informativo."""

                        messages = [{"role": "user", "content": pregunta}]
                        
                        # Loop agentico: el modelo puede usar múltiples tools
                        respuesta_final = ""
                        max_iterations = 5
                        iteration = 0
                        
                        while iteration < max_iterations:
                            iteration += 1
                            response = client.messages.create(
                                model="claude-sonnet-4-20250514",
                                max_tokens=1000,
                                system=system_prompt,
                                tools=TOOLS_DEF,
                                messages=messages
                            )
                            
                            if response.stop_reason == "end_turn":
                                for block in response.content:
                                    if hasattr(block, 'text'):
                                        respuesta_final = block.text
                                break
                            
                            elif response.stop_reason == "tool_use":
                                # Agregar respuesta del asistente al historial
                                messages.append({"role": "assistant", "content": response.content})
                                
                                # Ejecutar todas las tools solicitadas
                                tool_results = []
                                for block in response.content:
                                    if block.type == "tool_use":
                                        tool_result = ejecutar_tool(
                                            block.name, block.input, conn, mercado
                                        )
                                        tool_results.append({
                                            "type": "tool_result",
                                            "tool_use_id": block.id,
                                            "content": str(tool_result)
                                        })
                                
                                messages.append({"role": "user", "content": tool_results})
                            else:
                                break
                        
                        if not respuesta_final:
                            respuesta_final = "No pude generar una respuesta. Intenta de nuevo."
                        
                        st.markdown(respuesta_final)
                        st.session_state.chat_history.append({
                            "role": "assistant", "content": respuesta_final
                        })
                        
                    except Exception as e:
                        error_msg = f"Error al conectar con el agente: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant", "content": error_msg
                        })

# ─────────────────────────────────────────────
# TAB 4: CALIDAD DE DATOS
# ─────────────────────────────────────────────
with tab4:
    st.markdown("### 🔍 Reporte de Calidad de Datos")
    
    col_q1, col_q2 = st.columns(2)
    
    with col_q1:
        st.markdown("**precios_raw**")
        nulos = mercado['precio_spot'].isna().sum()
        total = len(mercado)
        st.progress(1 - nulos/total, text=f"Completitud precio_spot: {total-nulos}/{total}")
        
        anomalias_precio = []
        for _, row in mercado.iterrows():
            if pd.isna(row['precio_spot']):
                anomalias_precio.append(f"🟡 Nulo: {row['fecha'].date()} h{row['hora']} {row['barra']}")
            elif not (10 <= row['precio_spot'] <= 250):
                anomalias_precio.append(f"🔴 Fuera de rango: {row['precio_spot']} USD/MWh")
        
        for a in anomalias_precio:
            st.markdown(f"<div class='warning-alert'>{a}</div>", unsafe_allow_html=True)
        
        if not anomalias_precio:
            st.success("Sin anomalías en precios")
    
    with col_q2:
        st.markdown("**generacion_raw**")
        nulos_mwh = mercado['mwh_generados'].isna().sum()
        st.progress(1 - nulos_mwh/total, text=f"Completitud mwh_generados: {total-nulos_mwh}/{total}")
        
        st.markdown("<div class='anomaly-alert'>🔴 BLOQUEANTE: 1 valor 'ERROR' (string) convertido a NaN en Volcán 2024-01-03 20:00</div>", unsafe_allow_html=True)
        st.markdown("<div class='anomaly-alert'>🔴 BLOQUEANTE: barra_conexion en formato diferente a precios — normalizado</div>", unsafe_allow_html=True)
    
    st.markdown("**tipo_cambio_json**")
    st.success("✅ JSON válido · 3 registros · Cobertura completa para todas las fechas")
    
    st.markdown("---")
    st.markdown("**DataFrame `mercado` integrado:**")
    st.dataframe(
        mercado[['fecha','hora','periodo','central','tipo','barra',
                 'mwh_generados','precio_spot','usd_clp','ingreso_clp']],
        use_container_width=True
    )
