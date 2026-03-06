# Configuración de Google Sheets para la aplicación de matrículas

Sigue estos pasos para que el botón **Prueba** (y más adelante el registro automático de matrículas) escriba en una hoja de Google.

---

## 1. Instalar dependencias

En el entorno donde ejecutas la aplicación:

```bash
pip install gspread google-auth
```

---

## 2. Crear un proyecto en Google Cloud

1. Entra en [Google Cloud Console](https://console.cloud.google.com/).
2. Crea un proyecto nuevo (o elige uno existente): **Crear proyecto** → nombre p. ej. `MatriculasApp` → Crear.
3. En el menú lateral, ve a **APIs y servicios** → **Biblioteca**.
4. Busca **Google Sheets API** y haz clic → **Habilitar**.
5. Opcional (para subir imágenes más adelante): busca **Google Drive API** y **Habilitar**.

---

## 3. Crear una cuenta de servicio y descargar el JSON

1. En el menú lateral: **APIs y servicios** → **Credenciales**.
2. **+ Crear credenciales** → **Cuenta de servicio**.
3. Nombre (p. ej. `sheet-matriculas`) → **Crear y continuar** → rol opcional (puedes saltar) → **Listo**.
4. En la tabla de cuentas de servicio, haz clic en la que acabas de crear.
5. Pestaña **Claves** → **Añadir clave** → **Crear clave nueva** → **JSON** → **Crear**. Se descargará un archivo JSON.
6. **Guarda ese archivo** en la carpeta del proyecto con el nombre `google_credentials.json` (o el que prefieras y luego indica la ruta con la variable de entorno `GOOGLE_CREDENTIALS_JSON`).

**Importante:** no subas este JSON a un repositorio público. El `.gitignore` del proyecto ya incluye `google_credentials.json` para evitarlo.

---

## 4. Crear la hoja de Google y compartirla con la cuenta de servicio

1. Ve a [Google Sheets](https://sheets.google.com) y crea una hoja nueva (o usa una existente).
2. Ponle un nombre, por ejemplo **Matrículas detectadas** (o el que quieras).
3. Abre el archivo JSON que descargaste y busca el campo `"client_email"`. Será algo como:
   `matriculasapp-xxxxx@tu-proyecto.iam.gserviceaccount.com`
4. En la hoja de Google: **Compartir** → pega ese email → permiso **Editor** → **Enviar** (puedes desmarcar “Notificar” si no quieres).

---

## 5. Decirle a la aplicación qué hoja usar

Tienes dos opciones:

### Opción A: Usar la URL de la hoja (recomendado)

1. Abre tu hoja en el navegador y copia la URL, por ejemplo:
   `https://docs.google.com/spreadsheets/d/1ABC...xyz/edit`
2. En la carpeta del proyecto, crea o edita el archivo que use la aplicación para config (si no existe, se puede usar variable de entorno):
   - **Variable de entorno:**  
     `export GOOGLE_SHEET_URL="https://docs.google.com/spreadsheets/d/TU_ID_AQUI/edit"`  
     (en Windows: `set GOOGLE_SHEET_URL=...`)
   - **O en código:** en `google_sheet.py` puedes poner la URL en `GOOGLE_SHEET_URL` al inicio del archivo (solo si no vas a subir ese cambio a un repo público).

### Opción B: Usar el nombre del documento

1. El título de la hoja en Google (p. ej. **Matrículas detectadas**) debe coincidir con lo que use la app.
2. Variable de entorno:  
   `export GOOGLE_SHEET_NAME="Matrículas detectadas"`  
   Por defecto el código usa `GOOGLE_SHEET_NAME = "Matrículas detectadas"` si no hay URL.

---

## 6. Probar la conexión

1. Coloca `google_credentials.json` en la raíz del proyecto (o configura `GOOGLE_CREDENTIALS_JSON` con la ruta correcta).
2. Si usas URL, configura `GOOGLE_SHEET_URL`; si usas nombre, deja o configura `GOOGLE_SHEET_NAME`.
3. Ejecuta la aplicación y pulsa el botón **Prueba**.
4. Deberías ver un mensaje de éxito y una **nueva fila** en la hoja con algo como:
   - Fecha y hora, **PRUEBA-123**, 85, N/A, N/A, 1, 1.

Si aparece un error, el mensaje en pantalla y en `google_sheet.py` te indicarán si falta el JSON, la URL/nombre de la hoja o el permiso de la cuenta de servicio (compartir la hoja con el `client_email`).

---

## Resumen rápido

| Qué necesitas | Dónde |
|---------------|--------|
| Dependencias | `pip install gspread google-auth` |
| Proyecto Google Cloud | APIs habilitadas: Sheets (y opcional Drive) |
| Archivo JSON | `google_credentials.json` en el proyecto (o ruta en `GOOGLE_CREDENTIALS_JSON`) |
| Hoja compartida | Con el `client_email` del JSON, permiso Editor |
| Qué hoja usar | `GOOGLE_SHEET_URL` o `GOOGLE_SHEET_NAME` |

Cuando la prueba funcione, la integración real (escribir una fila por cada matrícula detectada) usará la misma configuración y la función `append_deteccion` del módulo `google_sheet.py`.
