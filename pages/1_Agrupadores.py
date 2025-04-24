""" Agrupadores Prevención"""

import streamlit as st
import pandas as pd
import io

st.set_page_config(
    page_title="Agrupadores Prevención",
    page_icon="🪪",
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

def main():
    st.title("Agrupadores Prevención 🪪")


    col1, col2 = st.columns(2)
    # Subir archivos
    with col1:
        st.subheader("PERSONAL 👥")
        personal_file = st.file_uploader("Subir archivo Excel de PERSONAL", type=["xlsx", "xls"])
    with col2:
        st.subheader("AGRUPADORES 👷")
        agrupadores_file = st.file_uploader("Subir archivo Excel de AGRUPADORES", type=["xlsx", "xls"])

    if personal_file and agrupadores_file:
        try:
            with st.spinner("Generando archivo...", show_time=True):
                # Leer archivos
                personal_df = pd.read_excel(personal_file, dtype={"Número de personal": str})
                agrupadores_df = pd.read_excel(agrupadores_file, dtype={"ID empleado": str})

                # Filtrar Agrupadores por "Fecha fin" (vacío)
                agrupadores_filtrado_df = agrupadores_df[agrupadores_df["Fecha fin"].isnull()]
                agrupadores_filtrado_df['ID empleado'] = agrupadores_filtrado_df['ID empleado'].str.replace(r'^000', '', regex=True)

                # Crear diccionario para mapear "ID empleado" a "Puesto Principal"
                puestos_principales = agrupadores_filtrado_df.set_index("ID empleado")["Puesto principal"].to_dict()

                # Mapear "Puesto Principal" al DataFrame de Personal
                personal_df["Puesto principal"] = personal_df["Número de personal"].map(puestos_principales)

                # Insertar "Puesto Principal" como tercera columna (índice 2)
                personal_cols = personal_df.columns.tolist()
                if "Puesto principal" in personal_cols:
                    personal_cols.remove("Puesto principal") #remove antes para no duplicar
                personal_cols = personal_cols[:2] + ["Puesto principal"] + personal_cols[2:]
                personal_df = personal_df[personal_cols]

                # Descargar el DataFrame resultante
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    personal_df.to_excel(writer, index=False, sheet_name='Personal_Unido')
                    writer.close()
                    processed_data = output.getvalue()

            st.dataframe(personal_df)
            st.download_button(
                label="Descargar Excel AGRUPADORES",
                data=processed_data,
                file_name=f"Agrupadores_personal_{pd.Timestamp('now').strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.ms-excel"
            )


        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    main()