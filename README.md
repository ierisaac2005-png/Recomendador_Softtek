# Recomendador documental Top-5 | Caso académico Softtek

**Autor:** Isaac Esquivel Ruiz  
**Materia:** Sistemas y tecnologías de administración del conocimiento

## Objetivo

Recomendar los cinco documentos más parecidos a un documento semilla o a una consulta de texto mediante TF-IDF y similitud coseno. La utilidad propuesta es recuperar políticas, guías y experiencias previas para reducir búsquedas repetidas y apoyar la atención de incidentes.

Esta es una actividad independiente de las tareas anteriores de ciclo de vida y ontología. Softtek se mantiene como referencia del contexto de servicios de TI. **Los 30 documentos son ficticios y fueron redactados para la actividad; no son políticas ni incidentes reales de la empresa.**

## Archivos de la entrega

- `Recomendador_Softtek.ipynb`: notebook autónomo, con resultados e interpretaciones de seis líneas.
- `recomendador.py`: implementación reutilizable y consola interactiva.
- `corpus.csv`: 30 documentos con `id`, `title`, `text` y `category`.
- `requirements.txt`: versiones de dependencias utilizadas en la ejecución local.
- `evidencias/`: las tres pruebas en PNG y TXT, resultados completos en JSON y controles de funcionamiento.
- `Reporte_Recomendador_Softtek.pdf`: reporte de dos páginas. El enlace del video sigue pendiente.
- `Guion_reflexion.md`: apoyo para grabar el video; no sustituye la grabación solicitada.

Los archivos se entregan individualmente. Al descargarlos, colocar las evidencias dentro de una carpeta llamada `evidencias` y conservar los nombres indicados aquí.

## Corpus

Se incluyen cinco temas: accesos, respaldos, despliegues, seguridad y rendimiento. Cada tema tiene seis documentos. Las categorías son **Políticas y procesos**, **Soporte e incidentes** y **Operación y guías**, con diez documentos cada una. Son categorías asignadas al redactar el corpus, no predicciones de un clasificador.

Los documentos tienen contenido distinto, con políticas, procesos, tickets, preguntas frecuentes y bitácoras. El algoritmo usa únicamente `title` y `text`; ni el ID ni la categoría influyen en la puntuación. Los temas están deliberadamente delimitados para una demostración pequeña, por lo que no representan la diversidad de un repositorio empresarial.

## Dependencias y ejecución local

Se verificó en Python 3.12. Las dependencias son NumPy, pandas, scikit-learn y Matplotlib. Se recomienda usar un entorno virtual y ejecutar estos comandos en la carpeta de la actividad:

```bash
python -m pip install -r requirements.txt
python recomendador.py --id DOC001
python recomendador.py --texto "Falló un respaldo y necesito revisar la restauración de los datos"
```

Para elegir el modo y escribir una entrada desde la consola:

```bash
python recomendador.py
```

Para generar nuevamente las tres pruebas y sus evidencias:

```bash
python recomendador.py --pruebas --salida evidencias
```

El parámetro `--k` permite cambiar la cantidad, por ejemplo `--id DOC001 --k 3`. El valor predeterminado y las pruebas de esta actividad usan cinco. La ruta del conjunto puede cambiarse con `--corpus otro_corpus.csv`.

## Google Colab

1. Abrir Colab y cargar `Recomendador_Softtek.ipynb` mediante **Archivo → Subir notebook**.
2. Ejecutar las celdas en orden. El notebook contiene el corpus y el código, y los escribe en una carpeta de trabajo llamada `recomendador_softtek_demo`.
3. Revisar las tres pruebas, sus puntuaciones y las interpretaciones.
4. En la última celda, cambiar `modo` y `entrada` para probar un ID o texto propio.

No necesita Drive, claves de API, modelos externos ni un repositorio publicado. Las dependencias se instalan únicamente si faltan; las versiones utilizadas quedan registradas en los resultados. La ejecución fue verificada localmente; no se afirma una ejecución realizada en el servicio de Colab.

## Cómo funciona

1. Se comprueban columnas, IDs únicos y textos no vacíos ni duplicados.
2. Se unen título y contenido. Se convierten a minúsculas, se normalizan tildes, se retiran símbolos y algunas palabras frecuentes en español.
3. `TfidfVectorizer` crea vectores de palabras y pares de palabras (`ngram_range=(1, 2)`), con frecuencia sublineal y normalización L2.
4. Se calcula `cosine_similarity` entre los documentos. Una consulta nueva se transforma con el mismo vocabulario, sin volver a ajustarlo.
5. Se ordenan puntuaciones de mayor a menor; en empates se utiliza el ID. Si existe una semilla, se excluye.

El score está entre 0 y 1 para estos vectores no negativos: indica similitud léxica, **no probabilidad de éxito ni comprensión completa del significado**. Se devuelven hasta K coincidencias positivas; si no hay ninguna, se avisa. No se rellena una lista con documentos de puntuación cero. Las tres pruebas incluidas sí devuelven cinco resultados.

## Pruebas y validación

| Prueba | Entrada | Relevantes temáticos en Top-5 | P@5 |
|---|---|---:|---:|
| Política/proceso | DOC001: política de acceso | 5 de 5 | 1.00 |
| Soporte/incidente | DOC008: respaldo nocturno fallido | 4 de 5 | 0.80 |
| Texto libre | Problema de despliegue y reversión | 5 de 5 | 1.00 |

La precisión temática media es **0.9333**. Los conjuntos de documentos relevantes fueron definidos por tema antes de ejecutar las pruebas y aparecen en `PRUEBAS`, dentro del script. P@5 se calcula como documentos temáticamente relevantes recuperados dividido entre cinco. No se usa para entrenar ni ajustar parámetros.

En la segunda prueba aparece DOC002, un ticket de cuenta bloqueada, con similitud 0.055285. Comparte palabras genéricas como “soporte”, “revisó”, “verificó” e “incidente”, pero no ayuda directamente a restaurar respaldos. Este falso positivo muestra que compartir términos no asegura utilidad. En la tercera prueba, pertenecer al tema de despliegues tampoco garantiza que cada resultado contenga un rollback aplicable.

La evaluación es ilustrativa: solo tres consultas, juicios temáticos del autor y un corpus sintético. No mide ahorro de tiempo, resolución de incidentes ni rendimiento en documentos reales. No se entrenó un clasificador supervisado y no hay una partición de entrenamiento/prueba; se construyó un índice de recuperación sobre el corpus completo.

También se verificaron exclusión de la semilla, orden de puntuaciones, repetibilidad, rechazo de entradas inválidas y consultas fuera del vocabulario. El JSON incluye parámetros, versiones y SHA-256 del corpus para relacionar la evidencia con el conjunto utilizado.

## Utilidad y mejoras

El recomendador puede acercar una política a ejemplos de aplicación o un incidente a procedimientos existentes. Eso podría reducir trabajo repetido y acelerar una decisión informada. No ejecuta cambios ni resuelve incidentes por sí solo.

Sus principales límites son los sinónimos, la ambigüedad, las variantes de palabras, los textos breves y el sesgo del corpus. Normalizar tildes no equivale a lematizar; “respaldo” y “respaldos” siguen siendo términos distintos. Los pares de palabras ayudan con expresiones, pero no resuelven completamente las negaciones.

En producción agregaría vigencia, versión, responsable, sistema afectado y permisos por rol. La autorización debe aplicarse antes de mostrar resultados. Evaluaría filtros temáticos, búsqueda híbrida con embeddings y retroalimentación de especialistas. Para un corpus grande evitaría almacenar toda la matriz N x N y calcularía similitudes por consulta. Mediría P@5 con más consultas, utilidad por técnico y tiempo de resolución antes de afirmar beneficios económicos.

## Reflexión en video: pendiente

La consigna pide una grabación de máximo cinco minutos. Usar `Guion_reflexion.md`, subir el video con acceso para el docente y agregar su enlace aquí y en el PDF antes de entregar. **El guion y las interpretaciones escritas no sustituyen ese requisito.**

Documentación técnica: [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) y [cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html).
