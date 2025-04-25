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
if 'hora_linea1' not in st.session_state:
    st.session_state.hora_linea1 = datetime.strptime('07:00', '%H:%M').time()
if 'hora_linea2' not in st.session_state:
    st.session_state.hora_linea2 = datetime.strptime('17:00', '%H:%M').time()


st.title("Visualizador Radón-RD200 y Meteorología 📈")

# Panel de configuración en un expander (desplegado por defecto)
with st.expander("Configuración 📋", expanded=True):
    # Crear 3 columnas para los controles
    col1, col2, col3 = st.columns(3)

    with col1:
        archivo_cargado = st.file_uploader("Cargar archivo de datos de radón", type=["txt"])
        
    with col2:
        fecha_inicial = st.date_input("Fecha inicial", datetime(2025, 4, 21).date())
        if 'hora_inicial' not in st.session_state:
            st.session_state.hora_inicial = datetime.now().time().replace(hour=10, minute=15, second=0, microsecond=0)
        hora_inicial = st.time_input("Hora inicial", st.session_state.hora_inicial)
        st.session_state.hora_inicial = hora_inicial
        
    with col3:
        st.page_link("https://www.meteogalicia.gal/web/observacion/rede-meteoroloxica/historico", label="Meteogalicia", icon="🌎")
        # Widget para cargar el archivo JSON meteorológico
        archivo_meteorologico = st.file_uploader("Cargar archivo JSON meteorológico", type=["json"])
        
        if archivo_meteorologico is not None:
            st.session_state.df_meteorologico = cargar_datos_meteorologicos(archivo_meteorologico)
            if st.session_state.df_meteorologico is not None:
                st.success("Archivo meteorológico cargado correctamente!")

    # Botón centrado
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        if st.button("Cargar y Mostrar Datos", use_container_width=True):
            if archivo_cargado is None:
                st.error("Por favor, carga un archivo de datos de radón.")
            elif st.session_state.df_meteorologico is None:
                st.error("Por favor, carga un archivo de datos meteorológicos (JSON).")
            else:
                fecha_hora_inicial = datetime.combine(fecha_inicial, hora_inicial)
                df_radon = cargar_y_procesar_datos(archivo_cargado, fecha_hora_inicial, st.session_state.df_meteorologico)

                if df_radon is not None:
                    st.session_state.df_radon = df_radon
                    st.success("Datos cargados y procesados correctamente!")

# Separador visual
st.markdown("---")

# Main area
if st.session_state.df_radon is not None:
    # Crear pestañas para diferentes visualizaciones
    tab1, tab2, tab3 = st.tabs(["Datos", "Gráfica", "Correlaciones"])

    with tab1:
        st.dataframe(st.session_state.df_radon)

    with tab2:
        st.subheader("Visualización de datos")

        # Variables disponibles son todas las columnas del DataFrame de radón
        variables_disponibles = list(st.session_state.df_radon.columns)
        with st.expander("Configuración de líneas verticales diarias", expanded=False):
            # Crear una fila para los controles de las líneas verticales
            st.subheader("Configuración de líneas verticales diarias")
            col_linea1, col_linea2 = st.columns(2)
            
            with col_linea1:
                hora_linea1 = st.time_input("Primera línea vertical (hora)", st.session_state.hora_linea1)
                st.session_state.hora_linea1 = hora_linea1
                color_linea1 = st.color_picker("Color primera línea", "#0000FF")
            
            with col_linea2:
                hora_linea2 = st.time_input("Segunda línea vertical (hora)", st.session_state.hora_linea2)
                st.session_state.hora_linea2 = hora_linea2
                color_linea2 = st.color_picker("Color segunda línea", "#00FF00")

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
            
            # Añadir líneas verticales para cada día en la primera hora seleccionada
            for hora in horas_linea1_filtradas:
                fig.add_shape(
                    type="line",
                    x0=hora,
                    y0=0,
                    x1=hora,
                    y1=1,
                    yref="paper",
                    line=dict(color=color_linea1, width=1, dash="dot")
                )
                # Añadir etiqueta solo para algunas líneas (para evitar sobrecarga)
                if horas_linea1_filtradas.index(hora) % 2 == 0:  # Mostrar etiqueta en días alternos
                    fig.add_annotation(
                        x=hora,
                        y=1,
                        yref="paper",
                        text=f"{st.session_state.hora_linea1.strftime('%H:%M')}",
                        showarrow=False,
                        textangle=-90,
                        yshift=10,
                        font=dict(size=10, color=color_linea1)
                    )
            
            # Añadir líneas verticales para cada día en la segunda hora seleccionada
            for hora in horas_linea2_filtradas:
                fig.add_shape(
                    type="line",
                    x0=hora,
                    y0=0,
                    x1=hora,
                    y1=1,
                    yref="paper",
                    line=dict(color=color_linea2, width=1, dash="dot")
                )
                # Añadir etiqueta solo para algunas líneas (para evitar sobrecarga)
                if horas_linea2_filtradas.index(hora) % 2 == 0:  # Mostrar etiqueta en días alternos
                    fig.add_annotation(
                        x=hora,
                        y=1,
                        yref="paper",
                        text=f"{st.session_state.hora_linea2.strftime('%H:%M')}",
                        showarrow=False,
                        textangle=-90,
                        yshift=10,
                        font=dict(size=10, color=color_linea2)
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

        # Información estadística básica
        st.subheader("Estadísticas")

        col_stats = st.columns(len(variables_disponibles) if variables_disponibles else 1)

        for i, variable in enumerate(variables_disponibles):
            with col_stats[i]:
                st.metric(f"{variable}", f"Promedio: {st.session_state.df_radon[variable].mean():.2f}")
                st.metric("", f"Máximo: {st.session_state.df_radon[variable].max():.2f}")
                st.metric("", f"Mínimo: {st.session_state.df_radon[variable].min():.2f}")

        # Botón de descarga CSV
        csv = st.session_state.df_radon.to_csv().encode('utf-8')
        st.download_button(
            label="Descargar datos como CSV",
            data=csv,
            file_name='datos_radon.csv',
            mime='text/csv',
        )

    with tab3:
        st.subheader("Mapa de Correlaciones")
        
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
