"""Recomendación Top-K mediante TF-IDF y similitud coseno.

Caso académico con documentos ficticios inspirados en servicios de TI.
No se utilizan category ni id como variables de similitud.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import textwrap
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOPWORDS = frozenset('a al algo antes como con cuando de del desde donde durante e el ella en entre es esta este estos fue ha hacia hasta la las le lo los mas mediante muy o para pero por porque que se segun ser si sin sobre su sus un una uno unos y'.split())


def limpiar_texto(texto: str) -> str:
    """Minúsculas, normalización de tildes y eliminación de símbolos.

    Se conserva la palabra 'no' para no borrar todas las negaciones, aunque
    TF-IDF no interpreta su sentido de forma fiable.
    """
    normal = unicodedata.normalize('NFKD', str(texto).lower())
    normal = ''.join(c for c in normal if not unicodedata.combining(c))
    normal = re.sub(r'[^a-z0-9\s]', ' ', normal)
    return ' '.join(t for t in normal.split() if t not in STOPWORDS)


class Recomendador:
    """Índice TF-IDF ajustado una sola vez al corpus completo."""

    def __init__(self, corpus: str | Path = 'corpus.csv') -> None:
        self.ruta = Path(corpus)
        self.df = pd.read_csv(self.ruta, dtype=str, keep_default_na=False)
        required = {'id', 'title', 'text'}
        if not required.issubset(self.df.columns):
            raise ValueError('El CSV necesita las columnas id, title y text.')
        if 'category' not in self.df:
            self.df['category'] = 'Sin categoría'
        if len(self.df) < 2:
            raise ValueError('Se requieren al menos dos documentos.')
        for col in required:
            self.df[col] = self.df[col].str.strip()
            if self.df[col].eq('').any():
                raise ValueError(f'Hay valores vacíos en {col}.')
        if self.df['id'].duplicated().any():
            raise ValueError('Los identificadores deben ser únicos.')
        textos = self.df['title'] + ' ' + self.df['text']
        limpios = textos.map(limpiar_texto)
        if limpios.eq('').any():
            raise ValueError('Hay documentos sin contenido después de la limpieza.')
        if limpios.duplicated().any():
            raise ValueError('Hay documentos duplicados después de la limpieza.')
        self.vectorizador = TfidfVectorizer(
            preprocessor=limpiar_texto,
            lowercase=False,
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm='l2',
            min_df=1,
        )
        self.matriz = self.vectorizador.fit_transform(textos)
        if np.any(np.asarray(self.matriz.getnnz(axis=1)) == 0):
            raise ValueError('Algún documento no genera términos válidos.')
        # Pequeño corpus: se calcula explícitamente la matriz N x N solicitada.
        self.similitudes = np.clip(cosine_similarity(self.matriz), 0.0, 1.0)
        self.indices = {doc_id: i for i, doc_id in enumerate(self.df['id'])}

    def recomendar(self, *, documento_id: str | None = None,
                   consulta: str | None = None, k: int = 5) -> pd.DataFrame:
        """Devuelve hasta K coincidencias positivas, ordenadas por score e ID.

        El score mide semejanza léxica, no probabilidad de que la solución sirva.
        Una consulta fuera del vocabulario devuelve una tabla vacía.
        """
        if (documento_id is None) == (consulta is None):
            raise ValueError('Proporciona un ID o una consulta, pero no ambos.')
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError('K debe ser un entero positivo.')
        semilla = None
        if documento_id is not None:
            documento_id = documento_id.strip()
            if documento_id not in self.indices:
                raise ValueError(f'ID desconocido: {documento_id}')
            semilla = self.indices[documento_id]
            puntuaciones = self.similitudes[semilla].copy()
            puntuaciones[semilla] = -1.0
        else:
            if not consulta or not limpiar_texto(consulta):
                raise ValueError('La consulta no puede estar vacía.')
            vector = self.vectorizador.transform([consulta])
            puntuaciones = np.clip(cosine_similarity(vector, self.matriz).ravel(), 0.0, 1.0)
        resultado = self.df[['id', 'title', 'category']].copy()
        resultado['similarity'] = puntuaciones
        resultado = resultado.loc[resultado['similarity'] > 0]
        resultado = resultado.sort_values(['similarity', 'id'], ascending=[False, True]).head(k)
        resultado.insert(0, 'rank', range(1, len(resultado) + 1))
        return resultado.reset_index(drop=True)


# Juicios de relevancia temática establecidos por el diseño del corpus.
# No se derivan de las puntuaciones ni se utilizan para ajustar el vectorizador.
PRUEBAS = [
    dict(nombre='prueba_1_politica', entrada='DOC001', tipo='id',
         relevantes=['DOC002', 'DOC003', 'DOC004', 'DOC005', 'DOC006'],
         criterio='Documentos de acceso, cuentas, permisos y autenticación.'),
    dict(nombre='prueba_2_incidente', entrada='DOC008', tipo='id',
         relevantes=['DOC007', 'DOC009', 'DOC010', 'DOC011', 'DOC012'],
         criterio='Documentos de respaldo, retención, integridad y restauración.'),
    dict(nombre='prueba_3_consulta',
         entrada='El despliegue de una nueva versión produjo errores. Necesito el procedimiento de reversión o rollback y las pruebas para recuperar el servicio.',
         tipo='texto', relevantes=['DOC013', 'DOC014', 'DOC015', 'DOC016', 'DOC017', 'DOC018'],
         criterio='Documentos de despliegue, validación de versiones y reversión.'),
]


def comentario_prueba(prueba: dict, resultados: pd.DataFrame) -> list[str]:
    aciertos = int(resultados['id'].isin(prueba['relevantes']).sum())
    top = str(resultados.iloc[0]['title']) if len(resultados) else 'Sin coincidencias'
    if prueba['nombre'].startswith('prueba_1'):
        contenido = [
            'Usé una política de acceso como punto de partida para buscar información relacionada.',
            f'La primera recomendación fue: {top}.',
            f'{aciertos} de las cinco recomendaciones pertenecen al tema de acceso y autenticación.',
            'La lista permite pasar de una política general a guías y casos que ayudan a aplicarla.',
            'El documento inicial se excluye para que no ocupe un lugar entre las recomendaciones.',
            'La similitud orienta la búsqueda, pero el técnico debe revisar los permisos y el contexto.',
        ]
    elif prueba['nombre'].startswith('prueba_2'):
        contenido = [
            'Partí de un incidente donde falló el respaldo por falta de espacio.',
            f'La primera recomendación fue: {top}.',
            f'{aciertos} de las cinco recomendaciones tratan respaldos y recuperación de datos.',
            'DOC002 trata una cuenta bloqueada y es un resultado ajeno al problema de respaldos.',
            'El vocabulario genérico compartido puede acercar incidentes que tienen causas distintas.',
            'Revisaría el contenido y añadiría filtros temáticos antes de usar una recomendación.',
        ]
    else:
        contenido = [
            'En esta prueba escribí el problema sin conocer el identificador de ningún documento.',
            f'La primera recomendación fue: {top}.',
            f'{aciertos} de las cinco recomendaciones corresponden al tema de despliegue y reversión.',
            'La consulta muestra que se pueden recuperar guías usando una descripción breve del problema.',
            'El resultado depende de palabras compartidas como versión, despliegue, pruebas y rollback.',
            'Si el usuario utiliza otros términos, podría ser necesario reformular la consulta.',
        ]
    return contenido


def generar_evidencias(modelo: Recomendador, carpeta: str | Path = 'evidencias') -> list[dict]:
    """Ejecuta las tres pruebas y exporta resultados, criterios e imágenes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    resumen = []
    for prueba in PRUEBAS:
        kwargs = {'documento_id': prueba['entrada']} if prueba['tipo'] == 'id' else {'consulta': prueba['entrada']}
        resultados = modelo.recomendar(**kwargs, k=5)
        if len(resultados) != 5:
            raise AssertionError(f"{prueba['nombre']}: se esperaban cinco resultados.")
        if prueba['tipo'] == 'id':
            assert prueba['entrada'] not in resultados['id'].tolist()
        assert resultados['similarity'].is_monotonic_decreasing
        assert resultados['id'].is_unique
        assert resultados['similarity'].between(0, 1).all()
        aciertos = int(resultados['id'].isin(prueba['relevantes']).sum())
        precision = aciertos / 5
        comentario = comentario_prueba(prueba, resultados)
        texto = ('PRUEBA: ' + prueba['nombre'] + '\nEntrada: ' + prueba['entrada']
                 + '\nCriterio de relevancia: ' + prueba['criterio']
                 + '\n\n' + resultados.to_string(index=False, float_format=lambda x: f'{x:.6f}')
                 + f'\n\nP@5 temática: {precision:.2f} ({aciertos}/5)\n'
                 + '\n'.join(comentario) + '\n')
        (carpeta / (prueba['nombre'] + '.txt')).write_text(texto, encoding='utf-8')
        fig = plt.figure(figsize=(13, 8), facecolor='white')
        fig.text(.045, .94, prueba['nombre'].replace('_', ' ').capitalize(), size=19, weight='bold')
        fig.text(.045, .895, textwrap.fill('Entrada: ' + prueba['entrada'], 115), va='top', size=10)
        ax = fig.add_axes([.045, .39, .91, .38]); ax.axis('off')
        rows = [[str(row['rank']), row['id'], textwrap.fill(row['title'], 46), f"{row['similarity']:.4f}"] for row in resultados.to_dict('records')]
        tabla = ax.table(cellText=rows, colLabels=['Top', 'ID', 'Título', 'Similitud coseno'],
                         colWidths=[.06, .11, .65, .18], cellLoc='left', bbox=[0, 0, 1, 1])
        tabla.auto_set_font_size(False); tabla.set_fontsize(10)
        for (r, c), cell in tabla.get_celld().items():
            cell.set_edgecolor('black'); cell.set_linewidth(.6)
            cell.set_facecolor('white')
            if r == 0: cell.set_text_props(weight='bold')
        fig.text(.045, .35, f'P@5 temática: {precision:.2f} ({aciertos}/5). Evaluación ilustrativa en corpus simulado.', size=11, weight='bold')
        fig.text(.045, .305, '\n'.join(textwrap.fill(x, 132) for x in comentario), size=10, va='top', linespacing=1.5)
        fig.text(.045, .045, 'Salida real del script. La similitud no representa una probabilidad de éxito.', size=9)
        fig.savefig(carpeta / (prueba['nombre'] + '.png'), dpi=160, bbox_inches='tight')
        plt.close(fig)
        resumen.append({**prueba, 'precision_at_5': precision,
                        'resultados': resultados.to_dict('records'), 'comentario': comentario})
        print(texto)
    # Registra versión, corpus y parámetros para relacionar evidencia y ejecución.
    import sklearn
    registro = {
        'corpus_sha256': hashlib.sha256(modelo.ruta.read_bytes()).hexdigest(),
        'documentos': len(modelo.df), 'terminos': modelo.matriz.shape[1],
        'versiones': {'python': sys.version.split()[0], 'scikit_learn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__},
        'parametros': {'ngram_range': [1, 2], 'sublinear_tf': True, 'norm': 'l2', 'campos': ['title', 'text'], 'k': 5},
        'evaluacion': 'Relevancia temática manual en corpus sintético. No evalúa resolución efectiva ni generalización.',
        'pruebas': resumen,
    }
    (carpeta / 'resultados.json').write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding='utf-8')
    return resumen


def validar_comportamiento(modelo: Recomendador) -> list[str]:
    """Controles de salida y de entradas que podrían producir recomendaciones engañosas."""
    controles = []
    assert np.allclose(modelo.similitudes, modelo.similitudes.T)
    assert np.allclose(np.diag(modelo.similitudes), 1)
    controles.append('Matriz coseno simétrica y diagonal igual a 1: OK.')
    salida = modelo.recomendar(documento_id='DOC001')
    assert len(salida) == 5 and 'DOC001' not in salida['id'].tolist()
    assert salida['similarity'].is_monotonic_decreasing and salida['id'].is_unique
    controles.append('Top-5 ordenado, sin duplicados ni documento semilla: OK.')
    assert salida.equals(modelo.recomendar(documento_id='DOC001'))
    controles.append('Resultado repetible con el mismo corpus: OK.')
    for entrada in [dict(documento_id='INEXISTENTE'), dict(consulta='   '),
                    dict(documento_id='DOC001', consulta='acceso'),
                    dict(documento_id='DOC001', k=0)]:
        try:
            modelo.recomendar(**entrada)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Entrada inválida aceptada: {entrada}')
    controles.append('ID inválido, consulta vacía, dos modalidades y K=0 rechazados: OK.')
    assert modelo.recomendar(consulta='zzzxxyyqq').empty
    controles.append('Consulta fuera del vocabulario: tabla vacía, sin resultados inventados: OK.')
    return controles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default='corpus.csv')
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument('--id', dest='documento_id')
    selector.add_argument('--texto')
    selector.add_argument('--pruebas', action='store_true')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--salida', default='evidencias')
    args = parser.parse_args()
    try:
        modelo = Recomendador(args.corpus)
        if args.pruebas:
            generar_evidencias(modelo, args.salida)
            controles = validar_comportamiento(modelo)
            Path(args.salida, 'validacion.txt').write_text('\n'.join(controles) + '\n', encoding='utf-8')
            print('\n'.join(controles))
            return 0
        if args.documento_id is None and args.texto is None:
            modo = input('Modo: 1 = ID, 2 = texto: ').strip()
            if modo == '1': args.documento_id = input('ID (por ejemplo DOC001): ').strip()
            elif modo == '2': args.texto = input('Describe el problema: ').strip()
            else: raise ValueError('Selecciona 1 o 2.')
        tabla = modelo.recomendar(documento_id=args.documento_id, consulta=args.texto, k=args.k)
        print('Entrada:', args.documento_id if args.documento_id is not None else args.texto)
        if tabla.empty:
            print('Sin coincidencias léxicas. Reformula con términos del dominio.')
        else:
            print(tabla.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
            if len(tabla) < args.k:
                print(f'Solo hay {len(tabla)} documentos con similitud positiva.')
        return 0
    except (ValueError, OSError, pd.errors.ParserError, EOFError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
