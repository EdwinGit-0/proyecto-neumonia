# -*- coding: utf-8 -*-
#
# Archivo de configuración del build de documentación de proyecto-neumonia,
# creado por sphinx-quickstart.
#
# Este archivo se ejecuta mediante execfile() con el directorio actual
# establecido en su directorio contenedor.
#
# Ten en cuenta que no todos los posibles valores de configuración están
# presentes en este archivo autogenerado.
#
# Todos los valores de configuración tienen un valor predeterminado; los
# valores comentados sirven para mostrar el valor predeterminado.

import os
import sys

# Si las extensiones (o los módulos a documentar con autodoc) están en otro
# directorio, agrega aquí esos directorios a sys.path. Si el directorio es
# relativo a la raíz de la documentación, usa os.path.abspath para hacerlo
# absoluto, como se muestra aquí.
# sys.path.insert(0, os.path.abspath('.'))

# -- Configuración general ----------------------------------------------------

# Si tu documentación necesita una versión mínima de Sphinx, indícala aquí.
# needs_sphinx = '1.0'

# Agrega aquí los nombres de los módulos de extensión de Sphinx, como
# cadenas. Pueden ser extensiones incluidas con Sphinx (nombradas
# 'sphinx.ext.*') o extensiones propias.
extensions = []

# Agrega aquí las rutas que contienen plantillas, relativas a este
# directorio.
templates_path = ['_templates']

# El sufijo de los nombres de archivo de origen.
source_suffix = '.rst'

# La codificación de los archivos de origen.
# source_encoding = 'utf-8-sig'

# El documento maestro del toctree.
master_doc = 'index'

# Información general del proyecto.
project = u'proyecto-neumonia'

# La información de versión del proyecto que se documenta; actúa como
# reemplazo de |version| y |release|, y también se usa en varios otros
# lugares de los documentos generados.
#
# La versión corta X.Y.
version = '0.1'
# La versión completa, incluidos los sufijos alpha/beta/rc.
release = '0.1'

# El idioma para el contenido autogenerado por Sphinx. Consulta la
# documentación para ver una lista de idiomas admitidos.
# language = None

# Hay dos opciones para reemplazar |today|: o bien, estableces today en un
# valor que no sea falso, y entonces se usa:
# today = ''
# O bien, today_fmt se usa como el formato de una llamada a strftime.
# today_fmt = '%B %d, %Y'

# Lista de patrones, relativa al directorio de origen, que coinciden con
# archivos y directorios que se ignoran al buscar archivos de origen.
exclude_patterns = ['_build']

# El rol reST predeterminado (usado para este marcado: `texto`) que se
# utilizará en todos los documentos.
# default_role = None

# Si es verdadero, se agregará '()' al texto de referencias cruzadas como
# :func:.
# add_function_parentheses = True

# Si es verdadero, el nombre del módulo actual se antepondrá a todos los
# títulos de unidades de descripción (como .. function::).
# add_module_names = True

# Si es verdadero, se mostrarán en la salida las directivas sectionauthor y
# moduleauthor. Se ignoran por defecto.
# show_authors = False

# El nombre del estilo Pygments (resaltado de sintaxis) a usar.
pygments_style = 'sphinx'

# Una lista de prefijos ignorados para la ordenación del índice de módulos.
# modindex_common_prefix = []


# -- Opciones para la salida HTML ----------------------------------------------

# El tema para las páginas HTML y HTML Help. Consulta la documentación para
# ver una lista de temas incorporados.
html_theme = 'default'

# Las opciones de tema son específicas de cada tema y permiten personalizar
# el aspecto de un tema. Para ver una lista de opciones disponibles en cada
# tema, consulta la documentación.
# html_theme_options = {}

# Agrega aquí las rutas que contienen temas personalizados, relativas a este
# directorio.
# html_theme_path = []

# El nombre para este conjunto de documentos Sphinx. Si es None, usa
# "<proyecto> v<versión> documentation".
# html_title = None

# Un título más corto para la barra de navegación. El predeterminado es el
# mismo que html_title.
# html_short_title = None

# El nombre de un archivo de imagen (relativo a este directorio) para
# colocarlo en la parte superior de la barra lateral.
# html_logo = None

# El nombre de un archivo de imagen (dentro de la ruta estática) para usarlo
# como favicon de la documentación. Este archivo debe ser un icono de
# Windows (.ico) de 16x16 o 32x32 píxeles.
# html_favicon = None

# Agrega aquí las rutas que contienen archivos estáticos personalizados
# (como hojas de estilo), relativas a este directorio. Se copian después de
# los archivos estáticos incorporados, de modo que un archivo llamado
# "default.css" sobrescribirá el "default.css" incorporado.
html_static_path = ['_static']

# Si no es '', se inserta una marca de tiempo 'Actualizado por última vez
# el:' al pie de cada página, usando el formato strftime indicado.
# html_last_updated_fmt = '%b %d, %Y'

# Si es verdadero, SmartyPants se usará para convertir comillas y guiones a
# entidades tipográficamente correctas.
# html_use_smartypants = True

# Plantillas de barra lateral personalizadas; mapean nombres de documentos a
# nombres de plantillas.
# html_sidebars = {}

# Plantillas adicionales que deben renderizarse a páginas; mapean nombres de
# páginas a nombres de plantillas.
# html_additional_pages = {}

# Si es falso, no se genera el índice de módulos.
# html_domain_indices = True

# Si es falso, no se genera ningún índice.
# html_use_index = True

# Si es verdadero, el índice se divide en páginas individuales para cada
# letra.
# html_split_index = False

# Si es verdadero, se agregan enlaces a las fuentes reST a las páginas.
# html_show_sourcelink = True

# Si es verdadero, "Created using Sphinx" se muestra en el pie de página de
# HTML. El valor predeterminado es True.
# html_show_sphinx = True

# Si es verdadero, se muestra "(C) Copyright ..." en el pie de página de
# HTML. El valor predeterminado es True.
# html_show_copyright = True

# Si es verdadero, se generará un archivo de descripción OpenSearch y todas
# las páginas contendrán una etiqueta <link> que lo referencie. El valor de
# esta opción debe ser la URL base desde la que se sirve el HTML terminado.
# html_use_opensearch = ''

# Este es el sufijo de nombre de archivo para los archivos HTML
# (p. ej. ".xhtml").
# html_file_suffix = None

# Nombre base del archivo de salida para el builder de HTML Help.
htmlhelp_basename = 'proyecto-neumoniadoc'


# -- Opciones para la salida LaTeX ---------------------------------------------

latex_elements = {
    # El tamaño del papel ('letterpaper' o 'a4paper').
    # 'papersize': 'letterpaper',

    # El tamaño de fuente ('10pt', '11pt' o '12pt').
    # 'pointsize': '10pt',

    # Contenido adicional para el preámbulo de LaTeX.
    # 'preamble': '',
}

# Agrupa el árbol de documentos en archivos LaTeX. Lista de tuplas
# (archivo de origen, nombre del archivo destino, título, autor,
# clase de documento [howto/manual]).
latex_documents = [
    ('index',
     'proyecto-neumonia.tex',
     u'proyecto-neumonia Documentation',
     u"EDWIN ROQUE CERROGRANDE", 'manual'),
]

# El nombre de un archivo de imagen (relativo a este directorio) para
# colocarlo en la parte superior de la página de título.
# latex_logo = None

# Para los documentos "manual", si es verdadero, los encabezados de nivel
# superior son partes, no capítulos.
# latex_use_parts = False

# Si es verdadero, se muestran referencias a páginas después de los enlaces
# internos.
# latex_show_pagerefs = False

# Si es verdadero, se muestran direcciones URL después de los enlaces
# externos.
# latex_show_urls = False

# Documentos para agregar como apéndice a todos los manuales.
# latex_appendices = []

# Si es falso, no se genera ningún índice de módulos.
# latex_domain_indices = True


# -- Opciones para la salida de páginas de manual ----------------------------------

# Una entrada por página de manual. Lista de tuplas
# (archivo de origen, nombre, descripción, autores, sección del manual).
man_pages = [
    ('index', 'proyecto-neumonia', u'proyecto-neumonia Documentation',
     [u"EDWIN ROQUE CERROGRANDE"], 1)
]

# Si es verdadero, se muestran direcciones URL después de los enlaces
# externos.
# man_show_urls = False


# -- Opciones para la salida Texinfo ------------------------------------------------

# Agrupa el árbol de documentos en archivos Texinfo. Lista de tuplas
# (archivo de origen, nombre del archivo destino, título, autor, directorio
# menu, descripción, categoría).
texinfo_documents = [
    ('index', 'proyecto-neumonia', u'proyecto-neumonia Documentation',
     u"EDWIN ROQUE CERROGRANDE", 'proyecto-neumonia',
     'Clasificacion de imagenes de rayos X de torax para la identificacion de neumonia mediante redes neuronales convolucionales', 'Miscellaneous'),
]

# Documentos para agregar como apéndice a todos los manuales.
# texinfo_appendices = []

# Si es falso, no se genera ningún índice de módulos.
# texinfo_domain_indices = True

# Cómo mostrar las direcciones URL: 'footnote', 'no' o 'inline'.
# texinfo_show_urls = 'footnote'