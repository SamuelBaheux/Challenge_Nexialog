import base64
import io
from dash import html
import pandas as pd

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col=[0])
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return html.Div(['Type de fichier non pris en charge.'])
    except Exception as e:
        print(e)
        return html.Div([
            'Il y a eu une erreur lors du traitement de ce fichier.'
        ])

    return [df, filename]
