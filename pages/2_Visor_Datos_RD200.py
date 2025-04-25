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
    st.session_state.hora_inicial = datetime.strptime('00:00', '%H:%M').time()
if 'hora_linea1' not in st.session_state:
    st.session_state.hora_linea1 = datetime.strptime('07:00', '%H:%M').time()
if 'hora_linea2' not in st.session_state:
    st.session_state.hora_linea2 = datetime.strptime('17:00', '%H:%M').time()
if 'incluir_fines_semana' not in st.session_state:
    st.session_state.incluir_fines_semana = False


st.title("Visualizador Radón-RD200 y Meteorología 📈")


# Main area

# Crear pestañas para diferentes visualizaciones
tab0, tab1, tab2, tab3, tab4 = st.tabs(["Configuración", "Datos", "Gráfica", "Estadísticas", "Correlaciones"])

with tab0:
    st.subheader("Carga de datos")
    
    # Crear 3 columnas para los controles
    col1, col2, col3 = st.columns(3)

    with col1:
        nuevo_archivo = st.file_uploader("Cargar archivo de datos de radón", type=["txt"], key="nuevo_radon")
        
    with col2:
        nueva_fecha = st.date_input("Fecha inicial", datetime(2025, 4, 21).date(), key="nueva_fecha")
        nueva_hora = st.time_input("Hora inicial", st.session_state.hora_inicial, key="nueva_hora")
        
    with col3:
        st.page_link("https://www.meteogalicia.gal/web/observacion/rede-meteoroloxica/historico", label="Meteogalicia", icon="🌎")
        # Widget para cargar el archivo JSON meteorológico
        nuevo_archivo_meteo = st.file_uploader("Cargar archivo JSON meteorológico", type=["json"], key="nuevo_meteo")

    # Botón centrado
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
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
                        st.success("Datos actualizados correctamente!")

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
        
        # Dividir en dos columnas para la visualización
        col_izq, col_der = st.columns([3, 2])
        
        # Panel de métricas básicas
        with col_der:
            st.write("#### Métricas básicas")
            for i, variable in enumerate(variables_disponibles):
                # Diseño mejorado para las métricas
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
        
        # Contenedor izquierdo para gráficos
        with col_izq:
            # Pestañas para diferentes tipos de gráficos
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
    else:
        st.info("No hay datos disponibles. Por favor, carga los archivos en la pestaña 'Configuración'.")

with tab4:
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
