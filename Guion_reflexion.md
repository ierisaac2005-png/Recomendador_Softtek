# Guion de reflexión organizacional

Duración aproximada: 3 a 4 minutos a ritmo normal. Mostrar el notebook y las tres pruebas mientras se explica. Grabar este guion con tus palabras y agregar el enlace real al reporte.

## Contexto y enfoque

Para esta actividad desarrollé un recomendador de documentos usando un caso académico relacionado con Softtek. Preparé treinta documentos ficticios de soporte y operación, como políticas, guías e incidentes. La idea es que una persona pueda seleccionar un documento o escribir un problema y recibir cinco recomendaciones relacionadas.

Elegí TF-IDF porque permite trabajar con texto de forma sencilla y explicar cómo se obtiene el resultado. Primero limpio el contenido y después convierto las palabras en vectores. La similitud coseno compara esos vectores y ordena los documentos. Cuando uso un documento como punto de partida, el programa lo excluye para que no se recomiende a sí mismo.

## Por qué puede reducir trabajo repetido

En una organización es común que alguien investigue un problema que otra persona ya atendió. Si la solución existe, pero está perdida entre documentos y tickets, se termina repitiendo el diagnóstico o preguntando nuevamente a otros equipos. Este recomendador puede acercar esa información al momento de atender el caso.

Por ejemplo, desde una política de acceso aparecieron tickets y guías sobre cuentas, permisos y autenticación. Eso permite conectar una regla general con experiencias concretas. También puede ayudar a decidir qué revisar primero, aunque la decisión final sigue siendo del técnico. En esta práctica no medí minutos ahorrados, así que el beneficio es una posibilidad que tendría que comprobarse con usuarios reales.

## Resultados y limitaciones

Ejecuté tres pruebas. En la primera y la tercera, las cinco recomendaciones pertenecían al tema esperado. En la segunda, cuatro eran de respaldos y una era sobre una cuenta bloqueada. Ese resultado es importante porque muestra que el sistema puede confundir documentos que comparten palabras como soporte o incidente, aunque traten problemas diferentes.

La precisión temática media fue de aproximadamente noventa y tres por ciento, pero no significa que el sistema tenga esa efectividad en una empresa. El corpus es pequeño, simulado y está organizado en temas claros. Además, un documento puede ser del tema correcto sin contener la solución que se necesita.

Otra limitación es que TF-IDF depende de las palabras utilizadas. Puede fallar con sinónimos, abreviaturas, documentos muy cortos o expresiones diferentes para el mismo problema. Tampoco entiende completamente una negación ni verifica que un procedimiento siga vigente. Por eso la puntuación no es una probabilidad de que la recomendación resuelva el incidente.

## Qué mejoraría

Para un uso real agregaría metadatos como sistema, versión, responsable y fecha de revisión. También aplicaría permisos por rol para evitar que una persona reciba documentos que no está autorizada a consultar.

Probaría filtros temáticos y una búsqueda híbrida que combine palabras con embeddings, para evaluar si recupera mejor los sinónimos. Finalmente, reuniría más consultas reales y pediría a especialistas que indiquen qué recomendaciones les sirvieron. Compararía la precisión, el tiempo de búsqueda y la resolución de casos. Así podría saber si el recomendador realmente reduce trabajo y mejora las decisiones, en lugar de quedarme únicamente con una buena puntuación en la demostración.
