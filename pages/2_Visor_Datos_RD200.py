import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import io
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np  # Importa numpy


st.set_page_config(
    page_title="Radón-RD200",
    page_icon="📈",
    layout="wide",
)


# CSS personalizado para cambiar ancho del sidebar
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            width: 275px !important;
        }
        section[data-testid="stSidebar"] > div:first-child {
            width: 275px !important;
        }
    </style>
""", unsafe_allow_html=True)

def cargar_datos_meteorologicos(archivo_json):
    """
    Carga datos meteorológicos desde un archivo JSON y crea un DataFrame pivotante.
    """
    try:
        data = json.load(archivo_json)
        df = pd.DataFrame(data)

        # Convertir la columna 'Fecha' a datetime
        df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y %H:%M')
        #print("DataFrame meteorológico inicial:")
        #print(df.head())

        # Obtener las variables únicas del DataFrame
        variables = df['Variable'].unique()

        # Crear DataFrame pivotante usando Fecha como índice y Variable como columnas
        df_pivot = df.pivot(index='Fecha', columns='Variable', values='Valor')

        # Usar todas las variables disponibles
        df_pivot = df_pivot[variables]

        #print("DataFrame meteorológico pivotado:")
        #print(df_pivot.head())

        return df_pivot

    except Exception as e:
        st.error(f"Error al procesar el archivo JSON: {e}")
        return None


def obtener_datos_meteorologicos(df_meteorologico, timestamp):
    """
    Busca el valor más cercano por timestamp en el DataFrame de datos meteorológicos.
    Esta versión es compatible con versiones antiguas de Pandas.
    """
    if df_meteorologico is None or df_meteorologico.empty:
        print("DataFrame meteorológico es None o está vacío en obtener_datos_meteorologicos")
        return None

    try:
        # Encontrar el timestamp más cercano (compatible con Pandas antiguas)
        time_diffs = df_meteorologico.index - timestamp
        abs_time_diffs = np.abs(time_diffs.values.astype('timedelta64[ns]'))  # Calcula los valores absolutos usando numpy
        idx = df_meteorologico.index[np.argmin(abs_time_diffs)]
        closest_row = df_meteorologico.loc[idx]

        # Crear un diccionario con todas las columnas disponibles
        resultado = {}
        for columna in df_meteorologico.columns:
            resultado[columna] = float(closest_row[columna])

        #print(f"Datos meteorológicos para {timestamp}: {resultado}")  # Imprime el resultado
        return resultado

    except Exception as e:
        print(f"Error al obtener datos meteorológicos: {e}")
        return None


def cargar_y_procesar_datos(archivo, fecha_inicial, df_meteorologico):
    """
    Carga datos de radón desde un archivo de texto, crea un DataFrame con timestamps.
    Añade todas las columnas del dataframe meteorológico al dataframe de radón, usando el valor mas cercano por tiempo.
    """

    try:
        lines = io.TextIOWrapper(archivo).readlines()
        datos_radon = []
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit() and ")" in line:
                try:
                    valor = int(line.split(")")[1].strip())
                    datos_radon.append(valor)
                except ValueError:
                    print(f"Error al convertir la línea a entero: {line}")
                    continue

        # Crear timestamps a partir de la fecha inicial
        timestamps = [fecha_inicial + timedelta(hours=i) for i in range(len(datos_radon))]

        # Crear el DataFrame inicial con radón
        df = pd.DataFrame({'Radon (Bq/m3)': datos_radon}, index=timestamps)
        #print("DataFrame de radón inicial:")
        #print(df.head())

        # Añadir datos meteorológicos reales
        # Iterar a través de las columnas del DataFrame meteorológico
        for columna in df_meteorologico.columns:
            # Crear una nueva columna en el DataFrame de radón con los datos meteorológicos
            df[columna] = [obtener_datos_meteorologicos(df_meteorologico, ts).get(columna) if obtener_datos_meteorologicos(df_meteorologico, ts) else None for ts in timestamps]

        #print("DataFrame de radón con datos meteorológicos:")
        #print(df.head())

        df.index.name = 'Timestamp'  # Nombre del índice
        return df

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        return None


# Inicializar session_state si es necesario
if 'df_radon' not in st.session_state:
    st.session_state.df_radon = None
if 'df_meteorologico' not in st.session_state:
    st.session_state.df_meteorologico = None
if 'hora_inicial' not in st.session_state:
    st.session_state.hora_inicial = datetime.strptime('09:30', '%H:%M').time()
if 'hora_linea1' not in st.session_state:
    st.session_state.hora_linea1 = datetime.strptime('07:00', '%H:%M').time()
if 'hora_linea2' not in st.session_state:
    st.session_state.hora_linea2 = datetime.strptime('17:00', '%H:%M').time()
if 'incluir_fines_semana' not in st.session_state:
    st.session_state.incluir_fines_semana = False
# Nuevas variables para el modelo de predicción
if 'modelo_prediccion' not in st.session_state:
    st.session_state.modelo_prediccion = None
if 'vars_modelo' not in st.session_state:
    st.session_state.vars_modelo = None
if 'df_modelo' not in st.session_state:
    st.session_state.df_modelo = None
if 'media_radon' not in st.session_state:
    st.session_state.media_radon = None


st.title("Visualizador Radón-RD200 y Meteorología 📈")


# Main area

# Crear pestañas para diferentes visualizaciones
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Configuración", "Datos", "Gráfica", "Estadísticas", "Histograma Radón", "Correlaciones", "Modelo Predicción"])

with tab0:
    st.subheader("Carga de datos")
    
    # Crear 3 columnas para los controles
    col1, col2, col3 = st.columns(3)

    with col1:
        nuevo_archivo = st.file_uploader("📇Cargar archivo de datos de radón", type=["txt"], key="nuevo_radon")
        
    with col2:
        nueva_fecha = st.date_input("📆 Fecha inicial", datetime(2025, 4, 29).date(), key="nueva_fecha")
        nueva_hora = st.time_input("🕒 Hora inicial", st.session_state.hora_inicial, key="nueva_hora")
        
    with col3:
        st.page_link("https://www.meteogalicia.gal/web/observacion/rede-meteoroloxica/historico", label="Meteogalicia", icon="🌎")
        # Widget para cargar el archivo JSON meteorológico
        nuevo_archivo_meteo = st.file_uploader("Cargar archivo JSON meteorológico", type=["json"], key="nuevo_meteo")

    
    # Botón centrado
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        st.divider()
        if st.button("Actualizar datos", use_container_width=True, key="actualizar_btn"):
            if nuevo_archivo is None:
                st.error("Por favor, carga un archivo de datos de radón.")
            elif nuevo_archivo_meteo is None:
                st.error("Por favor, carga un archivo de datos meteorológicos (JSON).")
            else:
                # Procesar archivos meteorológicos
                df_meteo_nuevo = cargar_datos_meteorologicos(nuevo_archivo_meteo)
                if df_meteo_nuevo is not None:
                    st.session_state.df_meteorologico = df_meteo_nuevo
                    
                    # Procesar archivo de radón
                    fecha_hora_inicial = datetime.combine(nueva_fecha, nueva_hora)
                    df_radon_nuevo = cargar_y_procesar_datos(nuevo_archivo, fecha_hora_inicial, st.session_state.df_meteorologico)
                    
                    if df_radon_nuevo is not None:
                        st.session_state.df_radon = df_radon_nuevo
                        st.success("Datos cargados correctamente!")

with tab1:
    # Añadir información de resumen sobre los datos
    if st.session_state.df_radon is not None:
        with st.container(border=True):
            # Calcular métricas importantes
            num_registros = len(st.session_state.df_radon)
            fecha_hora_inicial = st.session_state.df_radon.index.min().strftime("%d/%m/%Y %H:%M")
            fecha_hora_final = st.session_state.df_radon.index.max().strftime("%d/%m/%Y %H:%M")
            
            # Calcular el período de muestreo en horas
            if num_registros > 1:
                primer_timestamp = st.session_state.df_radon.index[0]
                segundo_timestamp = st.session_state.df_radon.index[1]
                periodo_muestreo = (segundo_timestamp - primer_timestamp).total_seconds() / 3600
                periodo_muestreo_str = f"{periodo_muestreo:.1f} horas"
            else:
                periodo_muestreo_str = "N/A"
            
            # Crear columnas para mostrar la información
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Número de registros", f"{num_registros}")
                st.metric("Período de muestreo", periodo_muestreo_str)
            with col2:
                st.metric("Fecha y hora inicial", fecha_hora_inicial)
                st.metric("Fecha y hora final", fecha_hora_final)
            
        # Mostrar el dataframe
        st.dataframe(st.session_state.df_radon)
        
        # Botón de descarga CSV
        csv = st.session_state.df_radon.to_csv().encode('utf-8')
        st.download_button(
            label="Descargar datos como CSV",
            data=csv,
            file_name='datos_radon.csv',
            mime='text/csv',
            key="descarga_datos_tab1"
        )
    else:
        st.info("No hay datos disponibles. Por favor, carga los archivos en la pestaña 'Configuración'.")

with tab2:
    st.subheader("Visualización de datos")

    if st.session_state.df_radon is not None:
        # Variables disponibles son todas las columnas del DataFrame de radón
        variables_disponibles = list(st.session_state.df_radon.columns)
        
        with st.expander("Configuración de áreas sombreadas diarias", expanded=False):
            # Crear una fila para los controles de las áreas sombreadas
            st.subheader("Configuración de áreas sombreadas diarias")
            col_linea1, col_linea2 = st.columns(2)
            
            with col_linea1:
                hora_linea1 = st.time_input("Hora de inicio", st.session_state.hora_linea1)
                st.session_state.hora_linea1 = hora_linea1
                color_linea1 = st.color_picker("Color del área sombreada", "#808080")
                # Opción para incluir o excluir fines de semana
                incluir_fines_semana = st.checkbox("Incluir sábados y domingos", value=st.session_state.incluir_fines_semana)
                st.session_state.incluir_fines_semana = incluir_fines_semana
            
            with col_linea2:
                hora_linea2 = st.time_input("Hora de fin", st.session_state.hora_linea2)
                st.session_state.hora_linea2 = hora_linea2
                opacity = st.slider("Opacidad", 0.0, 1.0, 0.2, 0.05)
        
        # Crear gráfica con Plotly para todas las variables
        fig = go.Figure()

        # Definir los colores para cada variable
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        color_index = 0

        # Añadir cada variable a la gráfica
        for i, variable in enumerate(variables_disponibles):
            if variable == 'Radon (Bq/m3)':
                visible = True
            else:
                visible = 'legendonly'  # Solo el radón visible por defecto

            fig.add_trace(
                go.Scatter(
                    x=st.session_state.df_radon.index,
                    y=st.session_state.df_radon[variable],
                    name=variable,
                    line=dict(color=default_colors[color_index % len(default_colors)]),
                    visible=visible,
                    yaxis=f"y{i+1}" if i > 0 else "y",
                    hovertemplate=f"{variable}: %{{y:.2f}}<extra></extra>"
                )
            )

            # Configurar los ejes Y adicionales (excepto el primero que es el default)
            if i > 0:
                fig.update_layout(**{
                    f'yaxis{i+1}': dict(
                        overlaying='y',
                        side='right',
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,  # Ocultar etiquetas de los ejes secundarios
                        showline=False,  # Ocultar línea del eje
                        showspikes=False,  # Ocultar spikes en hover
                        visible=False  # Hacer el eje completamente invisible
                    )
                })
            else:
                # Configurar el primer eje Y (Radón)
                fig.update_layout(
                    yaxis=dict(
                        title=dict(
                            text=variable,
                            font=dict(color=default_colors[color_index % len(default_colors)])
                        ),
                        tickfont=dict(color=default_colors[color_index % len(default_colors)])
                    )
                )
            color_index += 1

        # Si el radón está en los datos, añadir la línea de límite
        if 'Radon (Bq/m3)' in variables_disponibles:
            fig.add_shape(
                type="line",
                x0=st.session_state.df_radon.index[0],
                y0=300,
                x1=st.session_state.df_radon.index[-1],
                y1=300,
                line=dict(color="red", width=2, dash="dash")
            )

            # Añadir anotación para el límite
            fig.add_annotation(
                x=st.session_state.df_radon.index[-1],
                y=300,
                text="Límite: 300 Bq/m³",
                showarrow=False,
                xshift=10,
                font=dict(color="red")
            )
            
        # Añadir líneas verticales para las horas seleccionadas cada día
        if not st.session_state.df_radon.empty:
            # Obtener el primer y último día en los datos
            primer_dia = st.session_state.df_radon.index[0].date()
            ultimo_dia = st.session_state.df_radon.index[-1].date()
            
            # Generar fechas para cada día en las horas seleccionadas
            dias = pd.date_range(start=primer_dia, end=ultimo_dia, freq='D')
            
            # Primera línea vertical (hora seleccionada 1)
            horas_linea1 = [datetime.combine(dia.date(), st.session_state.hora_linea1) for dia in dias]
            horas_linea1_filtradas = [hora for hora in horas_linea1 if 
                                        st.session_state.df_radon.index[0] <= hora <= st.session_state.df_radon.index[-1]]
            
            # Segunda línea vertical (hora seleccionada 2)
            horas_linea2 = [datetime.combine(dia.date(), st.session_state.hora_linea2) for dia in dias]
            horas_linea2_filtradas = [hora for hora in horas_linea2 if 
                                        st.session_state.df_radon.index[0] <= hora <= st.session_state.df_radon.index[-1]]
            
            # Añadir áreas sombreadas entre las horas seleccionadas para cada día
            # Asumiendo que hora_linea1 es anterior a hora_linea2 en el mismo día
            for i in range(len(dias) - 1):
                # Obtener el día actual y el siguiente
                dia_actual = dias[i].date()
                dia_siguiente = dias[i+1].date()
                
                # Verificar si es fin de semana (5=sábado, 6=domingo)
                es_fin_semana = dia_actual.weekday() >= 5
                
                # Si es fin de semana y no se incluyen fines de semana, saltamos este día
                if es_fin_semana and not st.session_state.incluir_fines_semana:
                    continue
                
                # Crear timestamps para las horas en el día actual
                inicio = datetime.combine(dia_actual, st.session_state.hora_linea1)
                fin = datetime.combine(dia_actual, st.session_state.hora_linea2)
                
                # Si hora_linea1 es después de hora_linea2, ajustamos para sombrear desde hora_linea1 hasta hora_linea2 del día siguiente
                if st.session_state.hora_linea1 > st.session_state.hora_linea2:
                    fin = datetime.combine(dia_siguiente, st.session_state.hora_linea2)
                
                # Verificar si el rango está dentro de los datos
                if (inicio >= st.session_state.df_radon.index[0] and 
                    fin <= st.session_state.df_radon.index[-1]):
                    # Añadir área sombreada
                    fig.add_shape(
                        type="rect",
                        x0=inicio,
                        y0=0,
                        x1=fin,
                        y1=1,
                        yref="paper",
                        fillcolor=color_linea1,
                        opacity=opacity,
                        layer="below",
                        line_width=0,
                    )

        # Mejorar diseño
        fig.update_layout(
            title="Evolución temporal de variables",
            xaxis_title="Fecha y Hora",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified"
        )

        # Mostrar gráfica
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay datos disponibles. Por favor, carga los archivos en la pestaña 'Configuración'.")

with tab3:
    st.subheader("Estadísticas")

    if st.session_state.df_radon is not None:
        
        # Variables disponibles para estadísticas
        variables_disponibles = list(st.session_state.df_radon.columns)
        
        # Añadir selector para filtrar por rango de tiempo
        with st.expander("🔍 Filtrar por rango de tiempo", expanded=False):
            st.write("Selecciona un rango de tiempo para filtrar los datos:")
            tiempo_min = st.session_state.df_radon.index.min().to_pydatetime()
            tiempo_max = st.session_state.df_radon.index.max().to_pydatetime()
            col_fecha1, col_fecha2 = st.columns(2)
            with col_fecha1:
                fecha_inicio = st.date_input("Fecha inicio", tiempo_min.date(), min_value=tiempo_min.date(), max_value=tiempo_max.date())
                hora_inicio = st.time_input("Hora inicio", tiempo_min.time())
            with col_fecha2:
                fecha_fin = st.date_input("Fecha fin", tiempo_max.date(), min_value=tiempo_min.date(), max_value=tiempo_max.date())
                hora_fin = st.time_input("Hora fin", tiempo_max.time())
        
        # Crear timestamp combinando fecha y hora
        timestamp_inicio = datetime.combine(fecha_inicio, hora_inicio)
        timestamp_fin = datetime.combine(fecha_fin, hora_fin)
        
        # Filtrar DataFrame por el rango de tiempo seleccionado
        df_filtrado = st.session_state.df_radon.loc[timestamp_inicio:timestamp_fin]
        
        with st.expander("📏 Métricas básicas", expanded=False):
            col1, col2 = st.columns(2)
            for i, variable in enumerate(variables_disponibles):
                # Diseño mejorado para las métricas
                if i % 2 == 0:
                    col = col1
                else:
                    col = col2
                
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{variable}**")
                        
                        media = df_filtrado[variable].mean()
                        maximo = df_filtrado[variable].max()
                        minimo = df_filtrado[variable].min()
                        std = df_filtrado[variable].std()
                        
                        # Mejorar presentación de estadísticas con iconos
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("Media", f"{media:.2f}")
                            st.metric("Desv. Est.", f"{std:.2f}")
                        with c2:
                            st.metric("Máximo", f"{maximo:.2f}")
                            st.metric("Mínimo", f"{minimo:.2f}")

            
        

        # Pestañas para diferentes tipos de gráficos
        with st.expander("📊 Gráficos", expanded=False):
            tab_bar, tab_box, tab_hist = st.tabs(["Barras", "Boxplot", "Histograma"])
        
            # Crear un gráfico de barras para comparar medias, máximos y mínimos
            with tab_bar:
                st.write("#### Comparativa de variables")
                
                # Selector de variables (multiselección)
                vars_seleccionadas_bar = st.multiselect(
                    "Selecciona variables para comparar",
                    variables_disponibles,
                    default=variables_disponibles[:min(3, len(variables_disponibles))]  # Selección por defecto: primeras 3 variables o menos
                )
                
                if vars_seleccionadas_bar:
                    # Preparar datos para el gráfico de barras
                    stats_data = {
                        'Variable': [],
                        'Valor': [],
                        'Estadística': []
                    }
                    
                    for var in vars_seleccionadas_bar:
                        stats_data['Variable'].extend([var, var, var])
                        stats_data['Valor'].extend([
                            df_filtrado[var].mean(),
                            df_filtrado[var].max(),
                            df_filtrado[var].min()
                        ])
                        stats_data['Estadística'].extend(['Media', 'Máximo', 'Mínimo'])
                    
                    df_stats = pd.DataFrame(stats_data)
                    
                    # Crear gráfico de barras agrupadas
                    fig_bar = px.bar(
                        df_stats, 
                        x='Variable', 
                        y='Valor', 
                        color='Estadística',
                        barmode='group',
                        title='Comparativa estadística por variable',
                        labels={'Valor': 'Valor', 'Variable': 'Variable'},
                        color_discrete_sequence=px.colors.qualitative.G10
                    )
                    
                    # Mejorar diseño
                    fig_bar.update_layout(
                        legend_title="Estadística",
                        xaxis_title="",
                        height=400,
                    )
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("Selecciona al menos una variable para visualizar")
            
            # Visualización con boxplots
            with tab_box:
                st.write("#### Distribución de valores (Boxplot)")
                
                # Selector de variables (multiselección)
                vars_seleccionadas_box = st.multiselect(
                    "Selecciona variables para el boxplot",
                    variables_disponibles,
                    default=variables_disponibles[:min(3, len(variables_disponibles))]  # Selección por defecto: primeras 3 variables o menos
                )
                
                if vars_seleccionadas_box:
                    # Crear un DataFrame en formato largo para visualización
                    df_long = pd.melt(
                        df_filtrado.reset_index(), 
                        id_vars=['Timestamp'],
                        value_vars=vars_seleccionadas_box,
                        var_name='Variable', 
                        value_name='Valor'
                    )
                    
                    # Crear el boxplot
                    fig_box = px.box(
                        df_long, 
                        x='Variable', 
                        y='Valor', 
                        color='Variable',
                        title='Distribución de valores por variable',
                        labels={'Valor': 'Valor'},
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    
                    # Mejorar diseño
                    fig_box.update_layout(
                        showlegend=False,
                        xaxis_title="",
                        height=400,
                    )
                    
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.info("Selecciona al menos una variable para visualizar")
            
            # Histogramas
            with tab_hist:
                st.write("#### Distribución de frecuencia (Histograma)")
                
                # Selector de variable
                var_seleccionada = st.selectbox("Selecciona una variable", variables_disponibles)
                num_bins = st.slider("Número de intervalos", min_value=5, max_value=50, value=20)
                
                # Crear histograma
                fig_hist = px.histogram(
                    df_filtrado, 
                    x=var_seleccionada,
                    nbins=num_bins,
                    title=f'Histograma de {var_seleccionada}',
                    labels={var_seleccionada: 'Valor'},
                    color_discrete_sequence=['#636EFA']
                )
                
                # Añadir línea vertical con la media
                fig_hist.add_vline(
                    x=df_filtrado[var_seleccionada].mean(), 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Media",
                    annotation_position="top right"
                )
                
                # Mejorar diseño
                fig_hist.update_layout(
                    xaxis_title="Valor",
                    yaxis_title="Frecuencia",
                    height=400,
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
            
        # Separador para la nueva sección
        with st.expander("📊 Análisis de Niveles de Radón por Rango de Horas", expanded=False):
            st.markdown("---")
            st.subheader("Análisis de Niveles de Radón por Rango de Horas")
            
            # Verificar que la columna de radón existe
            if 'Radon (Bq/m3)' in st.session_state.df_radon.columns:
                # Crear un DataFrame para el análisis
                if 'df_modelo' not in st.session_state or st.session_state.df_modelo is None:
                    # Si no existe, crearlo ahora
                    df_analisis = st.session_state.df_radon.copy()
                    
                    # Añadir características temporales
                    df_analisis['Hora'] = df_analisis.index.hour
                    df_analisis['DiaSemana'] = df_analisis.index.dayofweek
                    df_analisis['Mes'] = df_analisis.index.month
                    df_analisis['EsFinDeSemana'] = df_analisis['DiaSemana'].apply(lambda x: 1 if x >= 5 else 0)
                    
                    # Verificar si el timestamp está dentro del rango de horas sombreadas
                    def esta_en_rango_horas(timestamp, hora_inicio, hora_fin):
                        hora = timestamp.hour + timestamp.minute / 60
                        hora_inicio_decimal = hora_inicio.hour + hora_inicio.minute / 60
                        hora_fin_decimal = hora_fin.hour + hora_fin.minute / 60
                        
                        if hora_inicio_decimal <= hora_fin_decimal:
                            return 1 if hora_inicio_decimal <= hora <= hora_fin_decimal else 0
                        else:
                            # Caso especial: el rango cruza la medianoche
                            return 1 if hora >= hora_inicio_decimal or hora <= hora_fin_decimal else 0
                    
                    # Aplicar la función para marcar si está dentro del rango de horas sombreadas
                    df_analisis['EnRangoHoras'] = df_analisis.index.map(
                        lambda x: esta_en_rango_horas(x, st.session_state.hora_linea1, st.session_state.hora_linea2)
                    )
                else:
                    # Usar el DataFrame que ya existe
                    df_analisis = st.session_state.df_modelo.copy()
                
                # Permitir al usuario modificar las horas de análisis
                col_config1, col_config2 = st.columns(2)
                
                with col_config1:
                    hora_inicio_analisis = st.time_input(
                        "Hora de inicio del rango", 
                        st.session_state.hora_linea1,
                        key="hora_inicio_analisis"
                    )
                
                with col_config2:
                    hora_fin_analisis = st.time_input(
                        "Hora de fin del rango", 
                        st.session_state.hora_linea2,
                        key="hora_fin_analisis"
                    )
                
                # Actualizar las horas en el DataFrame
                def esta_en_rango_horas_actualizado(timestamp, hora_inicio, hora_fin):
                    hora = timestamp.hour + timestamp.minute / 60
                    hora_inicio_decimal = hora_inicio.hour + hora_inicio.minute / 60
                    hora_fin_decimal = hora_fin.hour + hora_fin.minute / 60
                    
                    if hora_inicio_decimal <= hora_fin_decimal:
                        return 1 if hora_inicio_decimal <= hora <= hora_fin_decimal else 0
                    else:
                        # Caso especial: el rango cruza la medianoche
                        return 1 if hora >= hora_inicio_decimal or hora <= hora_fin_decimal else 0
                
                df_analisis['EnRangoHoras'] = df_analisis.index.map(
                    lambda x: esta_en_rango_horas_actualizado(x, hora_inicio_analisis, hora_fin_analisis)
                )
                
                # Dividir por rango de horas
                df_en_rango = df_analisis[df_analisis['EnRangoHoras'] == 1]
                df_fuera_rango = df_analisis[df_analisis['EnRangoHoras'] == 0]
                
                # Calcular estadísticas
                stats_en_rango = {
                    'Media': df_en_rango['Radon (Bq/m3)'].mean(),
                    'Mediana': df_en_rango['Radon (Bq/m3)'].median(),
                    'Máximo': df_en_rango['Radon (Bq/m3)'].max(),
                    'Mínimo': df_en_rango['Radon (Bq/m3)'].min(),
                    'Desv. Est.': df_en_rango['Radon (Bq/m3)'].std(),
                    'Registros': len(df_en_rango)
                }
                
                stats_fuera_rango = {
                    'Media': df_fuera_rango['Radon (Bq/m3)'].mean(),
                    'Mediana': df_fuera_rango['Radon (Bq/m3)'].median(),
                    'Máximo': df_fuera_rango['Radon (Bq/m3)'].max(),
                    'Mínimo': df_fuera_rango['Radon (Bq/m3)'].min(),
                    'Desv. Est.': df_fuera_rango['Radon (Bq/m3)'].std(),
                    'Registros': len(df_fuera_rango)
                }
                
                # Mostrar estadísticas en columnas
                col_rango1, col_rango2 = st.columns(2)
                
                with col_rango1:
                    st.write(f"#### En horas seleccionadas ({hora_inicio_analisis.strftime('%H:%M')} - {hora_fin_analisis.strftime('%H:%M')})")
                    for key, value in stats_en_rango.items():
                        if key != 'Registros':
                            st.metric(key, f"{value:.2f}" + (" Bq/m³" if key != 'Registros' else ""))
                        else:
                            st.metric(key, f"{value}")
                
                with col_rango2:
                    st.write("#### Fuera de horas seleccionadas")
                    for key, value in stats_fuera_rango.items():
                        if key != 'Registros':
                            st.metric(key, f"{value:.2f}" + (" Bq/m³" if key != 'Registros' else ""))
                        else:
                            st.metric(key, f"{value}")
                
                # Crear gráfico comparativo
                data_comp = {
                    'Categoría': ['En horas seleccionadas', 'Fuera de horas seleccionadas'],
                    'Media': [stats_en_rango['Media'], stats_fuera_rango['Media']],
                    'Mediana': [stats_en_rango['Mediana'], stats_fuera_rango['Mediana']],
                    'Máximo': [stats_en_rango['Máximo'], stats_fuera_rango['Máximo']],
                    'Mínimo': [stats_en_rango['Mínimo'], stats_fuera_rango['Mínimo']]
                }
                
                df_comp = pd.DataFrame(data_comp)
                df_comp_melt = pd.melt(df_comp, id_vars=['Categoría'], value_vars=['Media', 'Mediana', 'Máximo', 'Mínimo'])
                
                fig_comp = px.bar(
                    df_comp_melt,
                    x='Categoría',
                    y='value',
                    color='variable',
                    barmode='group',
                    title='Comparación de estadísticas por rango horario',
                    labels={'value': 'Valor (Bq/m³)', 'variable': 'Estadística', 'Categoría': ''},
                    color_discrete_sequence=px.colors.qualitative.G10
                )
                
                # Añadir línea de límite
                fig_comp.add_shape(
                    type="line",
                    x0=-0.5,
                    y0=300,
                    x1=1.5,
                    y1=300,
                    line=dict(color="red", width=2, dash="dash"),
                    name="Límite"
                )
                
                fig_comp.add_annotation(
                    x=1.5,
                    y=300,
                    text="Límite: 300 Bq/m³",
                    showarrow=False,
                    xshift=10,
                    font=dict(color="red")
                )
                
                # Mejorar diseño
                fig_comp.update_layout(
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Añadir interpretación
                diferencia = stats_en_rango['Media'] - stats_fuera_rango['Media']
                porcentaje = (diferencia / stats_fuera_rango['Media']) * 100 if stats_fuera_rango['Media'] > 0 else 0
                
                if abs(porcentaje) > 10:
                    if diferencia > 0:
                        st.info(f"Los niveles de radón son un {porcentaje:.1f}% más altos durante las horas seleccionadas.")
                    else:
                        st.info(f"Los niveles de radón son un {abs(porcentaje):.1f}% más bajos durante las horas seleccionadas.")
                else:
                    st.info("No hay diferencias significativas entre los niveles de radón dentro y fuera del rango de horas seleccionado.")
                
                # Añadir análisis por hora del día
                st.subheader("Análisis por Hora del Día")
                
                # Agrupar datos por hora y calcular estadísticas
                df_por_hora = df_analisis.groupby('Hora')['Radon (Bq/m3)'].agg(['mean', 'median', 'std', 'count']).reset_index()
                df_por_hora.columns = ['Hora', 'Media', 'Mediana', 'Desv_Est', 'Registros']
                
                # Crear gráfico de líneas para mostrar la evolución por hora
                fig_horas = px.line(
                    df_por_hora, 
                    x='Hora', 
                    y=['Media', 'Mediana'],
                    title='Niveles de radón por hora del día',
                    labels={'value': 'Concentración de Radón (Bq/m³)', 'Hora': 'Hora del día', 'variable': 'Estadística'},
                    markers=True
                )
                
                # Añadir banda de desviación estándar
                fig_horas.add_traces(
                    go.Scatter(
                        x=df_por_hora['Hora'].tolist() + df_por_hora['Hora'].tolist()[::-1],
                        y=(df_por_hora['Media'] + df_por_hora['Desv_Est']).tolist() + 
                        (df_por_hora['Media'] - df_por_hora['Desv_Est']).tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(0,100,80,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Desviación estándar'
                    )
                )
                
                # Añadir línea de límite
                fig_horas.add_shape(
                    type="line",
                    x0=-0.5,
                    y0=300,
                    x1=23.5,
                    y1=300,
                    line=dict(color="red", width=2, dash="dash"),
                    name="Límite"
                )
                
                # Mejorar diseño
                fig_horas.update_layout(
                    height=500,
                    xaxis=dict(
                        tickmode='linear',
                        tick0=0,
                        dtick=1
                    )
                )
                
                st.plotly_chart(fig_horas, use_container_width=True)
                
                # Hora con mayor nivel de radón
                hora_max = df_por_hora.loc[df_por_hora['Media'].idxmax()]
                hora_min = df_por_hora.loc[df_por_hora['Media'].idxmin()]
                
                st.markdown(f"""
                **Observaciones:**
                - La hora con mayor nivel medio de radón es las **{int(hora_max['Hora']):02d}:00** con **{hora_max['Media']:.2f} Bq/m³**
                - La hora con menor nivel medio de radón es las **{int(hora_min['Hora']):02d}:00** con **{hora_min['Media']:.2f} Bq/m³**
                - La diferencia entre el máximo y mínimo horario es de **{hora_max['Media'] - hora_min['Media']:.2f} Bq/m³**
                """)
                
                # Añadir análisis por día de la semana
                st.subheader("Análisis por Día de la Semana")
                
                # Mapear números de día de la semana a nombres
                dia_mapping = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 
                            4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
                
                # Crear columna con nombre del día
                df_analisis['NombreDia'] = df_analisis['DiaSemana'].map(dia_mapping)
                
                # Agrupar datos por día y calcular estadísticas
                df_por_dia = df_analisis.groupby('DiaSemana')['Radon (Bq/m3)'].agg(['mean', 'median', 'std', 'count']).reset_index()
                df_por_dia['NombreDia'] = df_por_dia['DiaSemana'].map(dia_mapping)
                df_por_dia = df_por_dia.sort_values('DiaSemana')  # Ordenar por día de la semana
                df_por_dia.columns = ['DiaSemana', 'Media', 'Mediana', 'Desv_Est', 'Registros', 'NombreDia']
                
                # Crear gráfico de barras para mostrar la evolución por día
                fig_dias = px.bar(
                    df_por_dia, 
                    x='NombreDia', 
                    y=['Media', 'Mediana'],
                    barmode='group',
                    title='Niveles de radón por día de la semana',
                    labels={'value': 'Concentración de Radón (Bq/m³)', 'NombreDia': 'Día de la semana', 'variable': 'Estadística'},
                    color_discrete_sequence=['#636EFA', '#EF553B']
                )
                
                # Añadir línea de límite
                fig_dias.add_shape(
                    type="line",
                    x0=-0.5,
                    y0=300,
                    x1=6.5,
                    y1=300,
                    line=dict(color="red", width=2, dash="dash"),
                    name="Límite"
                )
                
                # Mejorar diseño
                fig_dias.update_layout(
                    height=500,
                    xaxis={'categoryorder': 'array', 'categoryarray': [dia_mapping[i] for i in range(7)]}
                )
                
                st.plotly_chart(fig_dias, use_container_width=True)
                
                # Día con mayor nivel de radón
                dia_max = df_por_dia.loc[df_por_dia['Media'].idxmax()]
                dia_min = df_por_dia.loc[df_por_dia['Media'].idxmin()]
                
                st.markdown(f"""
                **Observaciones:**
                - El día con mayor nivel medio de radón es el **{dia_max['NombreDia']}** con **{dia_max['Media']:.2f} Bq/m³**
                - El día con menor nivel medio de radón es el **{dia_min['NombreDia']}** con **{dia_min['Media']:.2f} Bq/m³**
                - La diferencia entre el máximo y mínimo diario es de **{dia_max['Media'] - dia_min['Media']:.2f} Bq/m³**
                """)
            else:
                st.warning("No se encuentra la columna 'Radon (Bq/m3)' en los datos cargados.")
    else:
        st.info("No hay datos disponibles. Por favor, carga los archivos en la pestaña 'Configuración'.")

with tab4:
    st.subheader("Histograma Radón")
    
    if st.session_state.df_radon is not None:
        # Verificar que la columna de radón existe
        if 'Radon (Bq/m3)' in st.session_state.df_radon.columns:
            # Añadir selector para filtrar por rango de tiempo
            with st.expander("🔍 Filtrar por rango de tiempo", expanded=False):
                st.write("Selecciona un rango de tiempo para filtrar los datos:")
                tiempo_min = st.session_state.df_radon.index.min().to_pydatetime()
                tiempo_max = st.session_state.df_radon.index.max().to_pydatetime()
                col_fecha1, col_fecha2 = st.columns(2)
                with col_fecha1:
                    fecha_inicio = st.date_input("Fecha inicio", tiempo_min.date(), min_value=tiempo_min.date(), max_value=tiempo_max.date(), key="hist_fecha_inicio")
                    hora_inicio = st.time_input("Hora inicio", tiempo_min.time(), key="hist_hora_inicio")
                with col_fecha2:
                    fecha_fin = st.date_input("Fecha fin", tiempo_max.date(), min_value=tiempo_min.date(), max_value=tiempo_max.date(), key="hist_fecha_fin")
                    hora_fin = st.time_input("Hora fin", tiempo_max.time(), key="hist_hora_fin")
            
            # Crear timestamp combinando fecha y hora
            timestamp_inicio = datetime.combine(fecha_inicio, hora_inicio)
            timestamp_fin = datetime.combine(fecha_fin, hora_fin)
            
            # Filtrar DataFrame por el rango de tiempo seleccionado
            df_filtrado = st.session_state.df_radon.loc[timestamp_inicio:timestamp_fin]
            
            # Configuración del histograma
            with st.expander("Configuración del histograma", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    num_bins = st.slider("Número de intervalos (bins)", min_value=5, max_value=100, value=30, key="num_bins_radon")
                    hist_color = st.color_picker("Color del histograma", "#1f77b4", key="hist_color")
                
                with col2:
                    mostrar_media = st.checkbox("Mostrar línea de media", value=True, key="mostrar_media")
                    mostrar_mediana = st.checkbox("Mostrar línea de mediana", value=True, key="mostrar_mediana")
                    mostrar_limite_300 = st.checkbox("Mostrar línea de límite (300 Bq/m³)", value=True, key="mostrar_limite")
                    normalizado = st.checkbox("Histograma normalizado", value=False, key="hist_normalizado")
            
            # Crear contenedor para estadísticas básicas del radón
            with st.expander("Estadísticas básicas del radón", expanded=False):
                with st.container(border=True):
                    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                    
                    # Calcular estadísticas del radón
                    media = df_filtrado['Radon (Bq/m3)'].mean()
                    mediana = df_filtrado['Radon (Bq/m3)'].median()
                    maximo = df_filtrado['Radon (Bq/m3)'].max()
                    minimo = df_filtrado['Radon (Bq/m3)'].min()
                    std = df_filtrado['Radon (Bq/m3)'].std()
                    cv = (std / media) * 100 if media > 0 else 0  # Coeficiente de variación
                    
                    percentil_25 = df_filtrado['Radon (Bq/m3)'].quantile(0.25)
                    percentil_75 = df_filtrado['Radon (Bq/m3)'].quantile(0.75)
                    percentil_95 = df_filtrado['Radon (Bq/m3)'].quantile(0.95)
                    
                    # Calcular porcentaje de valores por encima de 300 Bq/m³
                    pct_sobre_limite = (df_filtrado['Radon (Bq/m3)'] > 300).mean() * 100
                    
                    with col_stats1:
                        st.metric("Media", f"{media:.2f} Bq/m³")
                        st.metric("Mínimo", f"{minimo:.2f} Bq/m³")
                    
                    with col_stats2:
                        st.metric("Mediana", f"{mediana:.2f} Bq/m³")
                        st.metric("Máximo", f"{maximo:.2f} Bq/m³")
                    
                    with col_stats3:
                        st.metric("Desv. Estándar", f"{std:.2f} Bq/m³")
                        st.metric("Coef. Variación", f"{cv:.2f}%")
                    
                    with col_stats4:
                        st.metric("Percentil 75", f"{percentil_75:.2f} Bq/m³")
                        st.metric("% sobre límite", f"{pct_sobre_limite:.2f}%")
            
            # Crear histograma
            histnorm = 'probability density' if normalizado else None
            fig_hist = px.histogram(
                df_filtrado, 
                x='Radon (Bq/m3)',
                nbins=num_bins,
                histnorm=histnorm,
                title=f'Histograma de Radón ({len(df_filtrado)} mediciones)',
                labels={'Radon (Bq/m3)': 'Concentración de Radón (Bq/m³)'},
                color_discrete_sequence=[hist_color]
            )
            
            # Añadir línea vertical con la media si se selecciona
            if mostrar_media:
                fig_hist.add_vline(
                    x=media, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Media: {media:.1f} Bq/m³",
                    annotation_position="top right"
                )
            
            # Añadir línea vertical con la mediana si se selecciona
            if mostrar_mediana:
                fig_hist.add_vline(
                    x=mediana, 
                    line_dash="dot", 
                    line_color="green",
                    annotation_text=f"Mediana: {mediana:.1f} Bq/m³",
                    annotation_position="top left"
                )
            
            # Añadir línea vertical con el límite de 300 Bq/m³ si se selecciona
            if mostrar_limite_300:
                fig_hist.add_vline(
                    x=300, 
                    line_dash="solid", 
                    line_color="black",
                    annotation_text="Límite: 300 Bq/m³",
                    annotation_position="bottom right"
                )
            
            # Mejorar diseño
            fig_hist.update_layout(
                xaxis_title="Concentración de Radón (Bq/m³)",
                yaxis_title="Frecuencia" if not normalizado else "Densidad",
                height=600,
            )
            
            # Mostrar histograma
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Añadir opción para descargar los datos filtrados
            csv = df_filtrado[['Radon (Bq/m3)']].to_csv().encode('utf-8')
            st.download_button(
                label="Descargar datos de radón como CSV",
                data=csv,
                file_name='datos_radon_filtrado.csv',
                mime='text/csv',
                key="descarga_radon_filtrado"
            )
        else:
            st.warning("No se encuentra la columna 'Radon (Bq/m3)' en los datos cargados.")
    else:
        st.info("No hay datos disponibles. Por favor, carga los archivos en la pestaña 'Configuración'.")

with tab5:
    st.subheader("Mapa de Correlaciones")
    
    if st.session_state.df_radon is not None:
        # Calcular la matriz de correlación
        corr_matrix = st.session_state.df_radon.corr()
        
        # Crear mapa de calor con Plotly
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',  # Mostrar valores con 2 decimales
            color_continuous_scale='RdBu_r',  # Escala de colores
            aspect='auto',
            title='Correlación entre variables'
        )
        
        # Mejorar diseño
        fig_corr.update_layout(
            height=600,
            width=800,
            xaxis=dict(title=''),
            yaxis=dict(title='')
        )
        
        # Mostrar el mapa de correlaciones
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Explicación
        st.markdown("""
        **Interpretación del mapa de correlaciones:**
        - Los valores cercanos a 1 (azul intenso) indican una fuerte correlación positiva
        - Los valores cercanos a -1 (rojo intenso) indican una fuerte correlación negativa
        - Los valores cercanos a 0 indican poca o ninguna correlación
        """)
    else:
        st.info("No hay datos disponibles. Por favor, carga los archivos en la pestaña 'Configuración'.")

with tab6:
    st.subheader("Modelo de Predicción de Radón")
    
    if st.session_state.df_radon is not None:
        # Importamos las bibliotecas necesarias al principio
        import sklearn
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import pandas as pd
        import numpy as np
        from datetime import time, datetime, timedelta

        # Verificamos que la columna de radón existe
        if 'Radon (Bq/m3)' in st.session_state.df_radon.columns:
            # Crear un DataFrame para el modelo
            df_modelo = st.session_state.df_radon.copy()
            
            # Añadir características temporales
            df_modelo['Hora'] = df_modelo.index.hour
            df_modelo['DiaSemana'] = df_modelo.index.dayofweek
            df_modelo['Mes'] = df_modelo.index.month
            df_modelo['EsFinDeSemana'] = df_modelo['DiaSemana'].apply(lambda x: 1 if x >= 5 else 0)
            
            # Verificar si el timestamp está dentro del rango de horas sombreadas
            def esta_en_rango_horas(timestamp, hora_inicio, hora_fin):
                hora = timestamp.hour + timestamp.minute / 60
                hora_inicio_decimal = hora_inicio.hour + hora_inicio.minute / 60
                hora_fin_decimal = hora_fin.hour + hora_fin.minute / 60
                
                if hora_inicio_decimal <= hora_fin_decimal:
                    return 1 if hora_inicio_decimal <= hora <= hora_fin_decimal else 0
                else:
                    # Caso especial: el rango cruza la medianoche
                    return 1 if hora >= hora_inicio_decimal or hora <= hora_fin_decimal else 0
            
            # Aplicar la función para marcar si está dentro del rango de horas sombreadas
            df_modelo['EnRangoHoras'] = df_modelo.index.map(
                lambda x: esta_en_rango_horas(x, st.session_state.hora_linea1, st.session_state.hora_linea2)
            )
            
            # Guardar el DataFrame procesado en session_state
            st.session_state.df_modelo = df_modelo
            
            # Interfaz para configurar el modelo
            st.write("### Configuración del Modelo")
            
            # Dividir la configuración en columnas
            col1, col2 = st.columns(2)
            
            with col1:
                # Selector de variables predictoras disponibles
                variables_disponibles = [col for col in df_modelo.columns if col != 'Radon (Bq/m3)']
                vars_seleccionadas = st.multiselect(
                    "Selecciona variables predictoras:",
                    variables_disponibles,
                    default=['Hora', 'DiaSemana', 'EnRangoHoras', 'EsFinDeSemana', 'Temperatura', 'Humedad']
                    if all(var in variables_disponibles for var in ['Temperatura', 'Humedad'])
                    else variables_disponibles[:min(4, len(variables_disponibles))]
                )
                
                # Porcentaje de datos para entrenamiento
                test_size = st.slider("Porcentaje de datos para prueba (%)", 10, 100, 70)
                test_size = test_size / 100  # Convertir a proporción
            
            with col2:
                # Parámetros del modelo RandomForest
                n_estimators = st.slider("Número de árboles", 10, 500, 100)
                max_depth = st.slider("Profundidad máxima de los árboles", 2, 30, 10)
                random_state = 42  # Valor fijo para reproducibilidad
            
            # Botón para entrenar el modelo
            if st.button("Entrenar Modelo", use_container_width=True, key="btn_entrenar"):
                if len(vars_seleccionadas) < 1:
                    st.error("Por favor, selecciona al menos una variable predictora.")
                else:
                    # Crear X e y para el modelo
                    X = df_modelo[vars_seleccionadas]
                    y = df_modelo['Radon (Bq/m3)']
                    
                    # Guardar variables seleccionadas en session_state
                    st.session_state.vars_modelo = vars_seleccionadas
                    st.session_state.media_radon = float(y.mean())
                    
                    # Verificar y manejar valores faltantes
                    if X.isna().any().any():
                        st.warning("Se han detectado valores faltantes. Realizando imputación...")
                        X = X.fillna(X.mean())
                    
                    # Dividir los datos en entrenamiento y prueba
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # Mostrar información sobre el conjunto de datos
                    st.write(f"Datos de entrenamiento: {len(X_train)} registros")
                    st.write(f"Datos de prueba: {len(X_test)} registros")
                    
                    # Crear y entrenar el modelo
                    with st.spinner('Entrenando modelo...'):
                        modelo = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=random_state
                        )
                        modelo.fit(X_train, y_train)
                        
                        # Guardar el modelo en session_state
                        st.session_state.modelo_prediccion = modelo
                        
                        # Hacer predicciones
                        y_pred_train = modelo.predict(X_train)
                        y_pred_test = modelo.predict(X_test)
                        
                        # Calcular métricas
                        mae_train = mean_absolute_error(y_train, y_pred_train)
                        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
                        r2_train = r2_score(y_train, y_pred_train)
                        
                        mae_test = mean_absolute_error(y_test, y_pred_test)
                        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
                        r2_test = r2_score(y_test, y_pred_test)
                    
                    # Mostrar resultados
                    st.write("### Resultados del Modelo")
                    
                    # Mostrar métricas en dos columnas
                    col_met1, col_met2 = st.columns(2)
                    
                    with col_met1:
                        st.write("#### Métricas en Entrenamiento")
                        st.metric("MAE", f"{mae_train:.2f} Bq/m³")
                        st.metric("RMSE", f"{rmse_train:.2f} Bq/m³")
                        st.metric("R²", f"{r2_train:.3f}")
                    
                    with col_met2:
                        st.write("#### Métricas en Prueba")
                        st.metric("MAE", f"{mae_test:.2f} Bq/m³")
                        st.metric("RMSE", f"{rmse_test:.2f} Bq/m³")
                        st.metric("R²", f"{r2_test:.3f}")
                    
                    # Visualizar importancia de características
                    importancia = pd.DataFrame({
                        'Variable': vars_seleccionadas,
                        'Importancia': modelo.feature_importances_
                    }).sort_values('Importancia', ascending=False)
                    
                    st.write("### Importancia de Variables")
                    fig_imp = px.bar(
                        importancia, 
                        x='Importancia', 
                        y='Variable',
                        orientation='h',
                        title='Importancia de variables en el modelo',
                        color='Importancia',
                        color_continuous_scale='Viridis'
                    )
                    
                    # Mejorar diseño
                    fig_imp.update_layout(
                        height=400,
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    
                    st.plotly_chart(fig_imp, use_container_width=True)
                    
                    # Graficar predicciones vs valores reales
                    df_resultados = pd.DataFrame({
                        'Real': y_test,
                        'Predicción': y_pred_test,
                        'Error': y_test - y_pred_test
                    })
                    
                    # Gráfica de dispersión
                    fig_scatter = px.scatter(
                        df_resultados,
                        x='Real',
                        y='Predicción',
                        title='Valores reales vs. predicciones',
                        labels={'Real': 'Valor Real (Bq/m³)', 'Predicción': 'Valor Predicho (Bq/m³)'}
                    )
                    
                    # Añadir línea de referencia
                    fig_scatter.add_trace(
                        go.Scatter(
                            x=[df_resultados['Real'].min(), df_resultados['Real'].max()],
                            y=[df_resultados['Real'].min(), df_resultados['Real'].max()],
                            mode='lines',
                            name='Referencia',
                            line=dict(color='red', dash='dash')
                        )
                    )
                    
                    # Mejorar diseño
                    fig_scatter.update_layout(height=500)
                    
                    st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Sección de predicción interactiva - Solo mostrar si ya hay un modelo entrenado
            if st.session_state.modelo_prediccion is not None and st.session_state.vars_modelo is not None:
                st.write("### Realizar Predicción Personalizada")
                st.write("Configura los valores para realizar una predicción:")
                
                # Crear controles para cada variable seleccionada
                input_values = {}
                
                # Dividir en varias columnas para optimizar espacio
                num_cols = 3
                cols = st.columns(num_cols)
                
                for i, var in enumerate(st.session_state.vars_modelo):
                    col_idx = i % num_cols
                    with cols[col_idx]:
                        # Personalizar el widget según el tipo de variable
                        if var == 'Hora':
                            input_values[var] = st.slider(f"{var}", 0, 23, 12, key=f"pred_{var}")
                        elif var == 'DiaSemana':
                            dias = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 
                                    4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
                            dia_seleccionado = st.selectbox(f"{var}", list(dias.keys()), 
                                                            format_func=lambda x: dias[x], key=f"pred_{var}")
                            input_values[var] = dia_seleccionado
                        elif var == 'Mes':
                            meses = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 
                                     5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
                                     9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
                            mes_seleccionado = st.selectbox(f"{var}", list(meses.keys()), 
                                                            format_func=lambda x: meses[x], key=f"pred_{var}")
                            input_values[var] = mes_seleccionado
                        elif var == 'EsFinDeSemana':
                            input_values[var] = st.checkbox(f"Es fin de semana", value=False, key=f"pred_{var}")
                        elif var == 'EnRangoHoras':
                            input_values[var] = st.checkbox(f"En rango de horas sombreadas", value=True, key=f"pred_{var}")
                        else:
                            # Para variables numéricas, crear slider con rango adaptado
                            min_val = float(st.session_state.df_modelo[var].min())
                            max_val = float(st.session_state.df_modelo[var].max())
                            mean_val = float(st.session_state.df_modelo[var].mean())
                            
                            # Redondear a 1 decimal para mejor visualización
                            step = (max_val - min_val) / 100
                            step = max(0.1, round(step, 1))
                            
                            input_values[var] = st.slider(
                                f"{var}", 
                                min_value=float(min_val),
                                max_value=float(max_val),
                                value=float(mean_val),
                                step=step,
                                key=f"pred_{var}"
                            )
                
                # Botón para realizar predicción con valores personalizados
                if st.button("Realizar Predicción", key="btn_prediccion"):
                    # Crear un DataFrame con los valores de entrada
                    X_pred = pd.DataFrame([input_values])
                    
                    # Realizar predicción
                    try:
                        prediccion = st.session_state.modelo_prediccion.predict(X_pred)[0]
                        
                        # Mostrar resultado
                        st.write("#### Resultado de la Predicción")
                        
                        # Estilo visual según el nivel de radón
                        color = "normal"
                        mensaje = ""
                        if prediccion > 300:
                            color = "danger"
                            mensaje = "⚠️ **Nivel por encima del límite recomendado.**"
                        elif prediccion > 200:
                            color = "warning"
                            mensaje = "⚠️ **Nivel elevado, pero por debajo del límite.**"
                        else:
                            color = "success"
                            mensaje = "✅ **Nivel dentro de los valores normales.**"
                        
                        st.metric(
                            "Concentración de Radón Estimada",
                            f"{prediccion:.2f} Bq/m³",
                            delta=f"{prediccion - st.session_state.media_radon:.2f} Bq/m³ respecto a la media"
                        )
                        
                        st.markdown(f"<div style='padding: 10px; border-radius: 5px; background-color: {'red' if color=='danger' else 'orange' if color=='warning' else 'green'}; color: white;'>{mensaje}</div>", unsafe_allow_html=True)
                        
                        # Añadir recomendaciones según el nivel
                        if color == "danger":
                            st.markdown("""
                                **Recomendaciones:**
                                - Asegurar una buena ventilación del espacio.
                                - Considerar el uso de sistemas de extracción.
                                - Evitar permanecer en la zona durante períodos prolongados si es posible.
                            """)
                        elif color == "warning":
                            st.markdown("""
                                **Recomendaciones:**
                                - Incrementar la ventilación del espacio.
                                - Monitorizar los niveles regularmente.
                            """)
                    except Exception as e:
                        st.error(f"Error al realizar la predicción: {e}")
                        st.info("Asegúrate de que has entrenado el modelo y que las variables de entrada son correctas.")
            elif st.session_state.modelo_prediccion is None:
                st.info("Primero debes entrenar un modelo usando el botón 'Entrenar Modelo'.")
        else:
            st.warning("No se encuentra la columna 'Radon (Bq/m3)' en los datos cargados.")
    else:
        st.info("No hay datos disponibles. Por favor, carga los archivos en la pestaña 'Configuración'.")
