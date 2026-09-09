Quiero que trabajes sobre el proyecto actual **proyecto-neumonia** y continúes su desarrollo de forma profesional, siguiendo las buenas prácticas de un proyecto de Ciencia de Datos / Machine Learning / Deep Learning.

## 1. REGLA PRINCIPAL

Antes de modificar o crear cualquier archivo:

1. Inspecciona la estructura REAL actual del proyecto.
2. Revisa los archivos que ya existen.
3. Revisa el `README.md` existente.
4. Revisa `requirements.txt`.
5. Revisa `.gitignore`.
6. Revisa la configuración existente de Git y DVC.
7. Revisa los notebooks, código fuente y tests que ya existan.
8. No dupliques archivos ni funcionalidades que ya existan.
9. No reemplaces configuraciones existentes sin una razón técnica.
10. Mantén coherencia con todo el trabajo realizado anteriormente.

El objetivo es **continuar el proyecto existente**, no crear un proyecto nuevo desde cero.

---

# 2. NO CREAR ARCHIVOS DUPLICADOS

Es MUY IMPORTANTE que respetes lo siguiente:

### README

Ya existe:

```text
README.md
```

NO crees:

```text
README2.md
README_FINAL.md
README_PROJECT.md
README_NEW.md
```

ni ningún otro README.

Utiliza y modifica únicamente el `README.md` existente cuando sea necesario.

---

### Requirements

Ya existe:

```text
requirements.txt
```

Utiliza SIEMPRE este archivo.

NO crees:

```text
requirements-dev.txt
requirements-test.txt
requirements-ml.txt
requirements-new.txt
```

salvo que exista una necesidad técnica real y previamente justificada.

Si una dependencia nueva es necesaria para el proyecto, agrégala al `requirements.txt` existente.

No dupliques dependencias.

---

### Entorno virtual

Ya existe:

```text
.venv/
```

NO crees otro entorno virtual.

NO crees:

```text
venv/
.env/
env2/
.venv2/
```

El entorno existente debe mantenerse.

---

### Git

El proyecto ya utiliza Git.

NO inicialices nuevamente Git.

NO crees otro repositorio.

NO modifiques el historial.

NO hagas commits automáticamente.

NO hagas push automáticamente.

NO hagas merge automáticamente.

El control de ramas y commits será realizado manualmente.

---

### DVC

El proyecto ya está configurado con DVC.

NO ejecutes nuevamente:

```bash
dvc init
```

NO crees otro repositorio DVC.

Utiliza la configuración existente.

El dataset original se encuentra administrado mediante DVC.

No agregues las imágenes directamente a Git.

---

# 3. ESTRUCTURA DEL PROYECTO

Respeta la estructura existente basada en Cookiecutter Data Science.

Utiliza las carpetas existentes:

```text
data/
docs/
models/
notebooks/
references/
reports/
src/
tests/
```

No crees carpetas nuevas si alguna de las existentes puede cumplir correctamente la función requerida.

Si realmente necesitas una nueva carpeta, justifica su necesidad y mantén una estructura coherente.

---

# 4. DOCUMENTACIÓN DEL DOCENTE

El proyecto debe desarrollarse siguiendo las instrucciones de la guía proporcionada para la monografía.

Si existe un documento de guía dentro de:

```text
references/
```

o se proporciona como archivo de referencia, revísalo antes de implementar las etapas correspondientes.

La guía del docente tiene prioridad para determinar:

* estructura;
* metodología;
* objetivos;
* etapas;
* entregables;
* requisitos técnicos;
* documentación;
* resultados;
* pruebas;
* buenas prácticas.

No inventes requisitos que no estén en la guía.

Si algún requisito no está claro, utiliza la interpretación técnicamente más razonable y conserva coherencia con el proyecto.

---

# 5. METODOLOGÍA

El proyecto debe seguir **CRISP-DM**:

1. Comprensión del negocio.
2. Comprensión de los datos.
3. Preparación de los datos.
4. Modelado.
5. Evaluación.
6. Despliegue.

Actualmente debemos avanzar de forma ordenada según la etapa correspondiente.

No marques como completada una etapa que todavía no haya sido ejecutada realmente.

---

# 6. EDA / COMPRENSIÓN DE LOS DATOS

Para la etapa de Comprensión de Datos realiza un EDA completo sobre el dataset real.

Primero inspecciona los datos existentes.

El dataset está administrado mediante DVC y debe encontrarse en:

```text
data/raw/chest_xray/
```

La estructura es:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

La estructura anterior es la del dataset crudo (`data/raw/chest_xray/`, 5,856 imágenes en total).

El reparto experimental actual para el modelado es 70/15/15 (semilla 42), generado en la preparación de datos y registrado en `data/interim/stratified_split_70_15_15.csv`, sin duplicados entre los tres conjuntos:

| Conjunto   | NORMAL | PNEUMONIA | Total |
| ---------- | -----: | --------: | ----: |
| Train      |  1,108 |     2,991 | 4,099 |
| Validation |    238 |       641 |   879 |
| Test       |    237 |       641 |   878 |
| Total      |  1,583 |     4,273 | 5,856 |

Verifica estos datos mediante código en lugar de asumirlos ciegamente.

El EDA debe analizar, según corresponda:

* cantidad de imágenes;
* distribución de clases;
* estructura de carpetas;
* extensiones;
* formatos;
* dimensiones;
* canales;
* imágenes corruptas;
* imágenes inválidas;
* archivos no deseados;
* posibles duplicados;
* tamaños de archivo;
* estadísticas básicas de píxeles;
* ejemplos representativos;
* diferencias entre clases;
* distribución del dataset;
* posibles problemas de calidad;
* cualquier hallazgo relevante para las siguientes etapas.

El análisis debe basarse en los datos reales.

NO inventes resultados.

---

# 7. NOTEBOOKS

Utiliza notebooks únicamente cuando sean apropiados para:

* exploración;
* visualización;
* análisis interactivo;
* experimentación.

Para el EDA puedes crear el notebook correspondiente dentro de:

```text
notebooks/
```

Utiliza nombres claros y ordenados.

Por ejemplo:

```text
01_comprension_datos_eda.ipynb
```

Pero NO crees notebooks innecesarios.

Si ya existe un notebook equivalente, utiliza el existente.

No coloques toda la lógica reutilizable dentro del notebook.

---

# 8. CÓDIGO REUTILIZABLE

La lógica reutilizable debe estar dentro de:

```text
src/
```

No copies y pegues las mismas funciones en varios notebooks.

Cuando una función pueda reutilizarse, colócala en el módulo apropiado.

Mantén separación de responsabilidades.

Por ejemplo:

```text
src/
├── data/
├── features/
├── models/
├── training/
└── evaluation/
```

Utiliza esta estructura solamente si resulta coherente con el código existente.

No reorganices todo el proyecto innecesariamente.

---

# 9. PROGRAMACIÓN ORIENTADA A OBJETOS

Utiliza POO cuando realmente aporte valor.

Se pueden utilizar:

* clases;
* métodos;
* encapsulamiento;
* herencia cuando tenga sentido;
* composición cuando sea apropiada.

NO conviertas todas las funciones en clases solamente para "cumplir POO".

Una función sencilla debe permanecer como función si no necesita una clase.

La arquitectura debe ser sencilla, mantenible y justificable.

---

# 10. PEP 8

Todo el código Python debe seguir **PEP 8**.

Utiliza:

* nombres claros;
* funciones pequeñas;
* clases con nombres en PascalCase;
* funciones y variables en snake_case;
* constantes en MAYÚSCULAS;
* indentación de 4 espacios;
* líneas razonablemente cortas;
* imports organizados;
* espacios adecuados;
* separación lógica entre componentes.

Ejemplo:

```python
class ImageAnalyzer:
    """Analyze image dataset properties."""

    def get_image_dimensions(self, image_path):
        """Return image dimensions."""
        ...
```

Los comentarios del código deben estar en **inglés**.

Los docstrings también deben estar en **inglés**.

---

# 11. TYPE HINTS

Utiliza type hints en funciones y métodos cuando sea apropiado.

Ejemplo:

```python
def count_images(directory: Path) -> int:
    ...
```

Evita utilizar `Any` innecesariamente.

---

# 12. DOCSTRINGS

Las funciones y clases importantes deben tener docstrings claros.

Los docstrings deben explicar:

* propósito;
* parámetros;
* retorno;
* comportamiento relevante.

Utiliza un formato consistente.

---

# 13. TESTING

El proyecto debe utilizar **pytest** para las pruebas automatizadas.

Utiliza la carpeta existente:

```text
tests/
```

NO crees otro framework de testing.

NO utilices unittest si pytest puede resolver correctamente el requisito.

NO crees scripts de prueba independientes innecesarios.

Las pruebas deben verificar principalmente:

* funciones de procesamiento;
* validaciones;
* carga de datos;
* transformaciones;
* componentes reutilizables;
* comportamiento esperado;
* manejo de errores.

Las pruebas NO deben depender innecesariamente del dataset completo si pueden utilizar datos pequeños de prueba.

Utiliza fixtures cuando aporten valor.

Ejemplo:

```python
def test_count_images():
    ...
```

Ejecuta las pruebas con:

```bash
pytest
```

No ejecutes pruebas que requieran descargar datos externos sin necesidad.

---

# 14. CALIDAD DEL CÓDIGO

El código debe ser:

* modular;
* reutilizable;
* legible;
* mantenible;
* testeable;
* reproducible.

Evita:

* código duplicado;
* variables innecesarias;
* funciones gigantes;
* clases innecesarias;
* imports sin utilizar;
* rutas absolutas;
* valores mágicos;
* código muerto;
* notebooks llenos de código repetido.

---

# 15. RUTAS Y DATOS

Nunca utilices rutas como:

```text
C:\Users\TOKO\Downloads\...
```

El proyecto debe funcionar utilizando rutas relativas al proyecto.

Utiliza `pathlib.Path` cuando sea apropiado.

Ejemplo:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "chest_xray"
```

La ruta debe funcionar para otros usuarios que clonen el repositorio.

---

# 16. DATA LEAKAGE

Ten especial cuidado con la contaminación entre conjuntos.

Nunca:

* mezcles train con validation;
* mezcles train con test;
* utilices información del test durante el entrenamiento;
* ajustes transformaciones utilizando información del test.

El conjunto `test` debe permanecer reservado para la evaluación final.

---

# 17. MODELOS

Los modelos previstos para comparación son:

* VGG16;
* ResNet50;
* MobileNetV2.

Se utilizará Transfer Learning.

NO afirmes que uno de estos modelos es el mejor hasta haber realizado los experimentos.

NO inventes métricas.

NO inventes resultados.

---

# 18. RESULTADOS

Todo resultado debe provenir de una ejecución real.

Nunca escribas valores ficticios como:

```text
Accuracy = 95%
F1 = 0.91
ROC-AUC = 0.97
```

si todavía no han sido calculados.

Si un resultado todavía no existe, indica que queda pendiente.

---

# 19. VISUALIZACIONES

Las visualizaciones deben ser útiles para responder preguntas del análisis.

Evita generar gráficos solamente para llenar el notebook.

Las figuras relevantes deben poder guardarse dentro de:

```text
reports/figures/
```

cuando corresponda.

No utilices colores arbitrarios o excesivamente numerosos.

Mantén gráficos claros y profesionales.

---

# 20. REPRODUCIBILIDAD

Mantén:

* Git para código;
* DVC para datos;
* `requirements.txt` para dependencias;
* entorno virtual existente;
* semillas aleatorias cuando corresponda;
* rutas relativas;
* configuración reproducible.

---

# 21. README

El `README.md` existente debe mantenerse actualizado con el estado REAL del proyecto.

No marques como terminada una etapa que todavía no se haya realizado.

No agregues resultados inventados.

No crees otro README.

---

# 22. NO MODIFICAR INNECESARIAMENTE

No cambies archivos que no estén relacionados con la tarea.

No reemplaces configuraciones existentes sin necesidad.

No borres archivos existentes.

No borres notebooks.

No borres código funcional.

No elimines dependencias sin comprobar primero que no se utilicen.

No cambies la estructura completa del proyecto por preferencias personales.

---

# 23. GIT

No ejecutes automáticamente:

```bash
git commit
git push
git merge
git checkout
git reset
```

El manejo de Git lo realizaremos manualmente.

Antes de terminar, indícame:

1. qué archivos creaste;
2. qué archivos modificaste;
3. qué archivos no fue necesario modificar;
4. qué dependencias nuevas fueron necesarias, si alguna;
5. qué pruebas ejecutaste;
6. qué resultados reales obtuviste;
7. qué queda pendiente.

---

# 24. FORMA DE TRABAJO

Trabaja de manera incremental.

Primero analiza.

Después implementa.

Después ejecuta.

Después prueba.

Después revisa.

No generes cientos de archivos de una sola vez si no son necesarios.

No implementes etapas futuras antes de terminar correctamente la etapa actual.

Cada etapa debe quedar funcional antes de avanzar.

---

# 25. OBJETIVO ACTUAL

El objetivo inmediato es continuar correctamente con la etapa de **Comprensión de Datos / EDA**.

Analiza el proyecto actual y determina qué archivos son realmente necesarios para completar esta etapa.

Crea o modifica únicamente los archivos necesarios.

El resultado debe integrarse con la estructura existente.

No crees duplicados.

No inventes resultados.

No modifiques el dataset original.

No subas imágenes del dataset a Git.

No cambies la configuración de DVC.

No crees un nuevo README.

No crees un nuevo requirements.

Utiliza pytest para testing.

Aplica PEP 8, type hints, docstrings y POO cuando corresponda.

Al finalizar, deja el proyecto listo para continuar posteriormente con la etapa de **Preparación de Datos**, sin implementarla todavía.
