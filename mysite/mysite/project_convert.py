from IPython.nbformat import current as nbformat
from IPython.nbconvert import PythonExporter

filepath = 'E:\\Chrome-Downloads\\NewProject\\SignatureVerification\\Siamese_Network\\predict.ipynb'
export_path = '.'

with open(filepath) as fh:
    nb = nbformat.reads_json(fh.read())

exporter = PythonExporter()

# source is a tuple of python source code
# meta contains metadata
source, meta = exporter.from_notebook_node(nb)

with open(export_path, 'w+') as fh:
    fh.writelines(source)
