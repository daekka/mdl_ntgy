import streamlit as st
from streamlit_calendar import calendar

# Diccionario de colores por recurso
resource_colors = {
    "mec": "#FF5733",     # Naranja
    "elec": "#337DFF",    # Azul
    "ic": "#33D1FF",      # Celeste
    "ge": "#28A745",      # Verde
    "qym": "#FFC300",     # Amarillo
    "ope": "#C70039",     # Rojo oscuro
}

# Selector de vista inicial
initial_view = st.selectbox(
    "Selecciona la vista del calendario",
    ["resourceTimelineDay", "resourceTimelineWeek", "resourceTimelineMonth"],
    format_func=lambda x: {
        "resourceTimelineDay": "Día",
        "resourceTimelineWeek": "Semana",
        "resourceTimelineMonth": "Mes"
    }[x]
)

# Configuración del calendario
calendar_options = {
    "editable": False,
    "selectable": True,
    "locale": 'es',
    "firstDay": 1,
    "hour": '2-digit',
    "minute": '2-digit',
    "second": '0-digit',
    "hour12": False,
    "initialView": initial_view,
    "slotMinTime": "06:00:00",
    "slotMaxTime": "20:00:00",
    "resourceGroupField": "area",
    "headerToolbar": {
        "left": "today prev,next",
        "center": "title",
        "right": "timeGridWeek,timeGridDay,resourceTimelineDay,resourceTimelineWeek,resourceTimelineMonth",
    },
    "resources": [
        {"id": "mec", "area": "Mecánico", "title": "Mecánico"},
        {"id": "elec", "area": "Eléctrico", "title": "Eléctrico"},
        {"id": "ic", "area": "I&C", "title": "I&C"},
        {"id": "ge", "area": "GE", "title": "GE"},
        {"id": "qym", "area": "QyM", "title": "QyM"},
        {"id": "ope", "area": "Operación", "title": "Operación"},
    ],
}

# Lista de eventos con tooltips nativos (campo title) y color por recurso
calendar_events = [
    {
        "title": "OT Mecánica 101",
        "start": "2025-05-03T08:00:00",
        "end": "2025-05-03T10:00:00",
        "resourceId": "mec",
        "color": resource_colors["mec"],
        "title": "Revisión de válvulas y ajuste de componentes mecánicos",
    },
    {
        "title": "OT Eléctrica 102",
        "start": "2025-05-03T10:00:00",
        "end": "2025-05-03T12:00:00",
        "resourceId": "elec",
        "color": resource_colors["elec"],
        "title": "Inspección de tableros y cableado",
    },
    {
        "title": "OT I&C 103",
        "start": "2025-05-03T13:00:00",
        "end": "2025-05-03T15:00:00",
        "resourceId": "ic",
        "color": resource_colors["ic"],
        "title": "Calibración de instrumentos de control",
    },
    {
        "title": "OT GE 104",
        "start": "2025-05-04T08:00:00",
        "end": "2025-05-04T10:00:00",
        "resourceId": "ge",
        "color": resource_colors["ge"],
        "title": "Mantenimiento de generadores eléctricos",
    },
    {
        "title": "OT QyM 105",
        "start": "2025-05-04T10:30:00",
        "end": "2025-05-04T12:30:00",
        "resourceId": "qym",
        "color": resource_colors["qym"],
        "title": "Control de calidad de materiales",
    },
    {
        "title": "OT Operación 106",
        "start": "2025-05-04T13:00:00",
        "end": "2025-05-04T15:00:00",
        "resourceId": "ope",
        "color": resource_colors["ope"],
        "title": "Supervisión de arranque de línea de producción",
    },
]

# Estilos personalizados
custom_css = """
    .fc-event-past {
        opacity: 0.8;
    }
    .fc-event-time {
        font-style: italic;
    }
    .fc-event-title {
        font-weight: 700;
    }
    .fc-toolbar-title {
        font-size: 2rem;
    }
"""

# Mostrar calendario en Streamlit
calendar_component = calendar(
    events=calendar_events,
    options=calendar_options,
    custom_css=custom_css,
    key='calendar',
)

st.write(calendar_component)
