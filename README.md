<!-- README.md -->
# Servicio 2 convertido en librería

Este microservicio queda empaquetado como librería instalable y ya no necesita que
el servicio 1 le invoque por `subprocess`.

## Qué cambia

- La API pública pasa a ser `ruesma_ocr_service.Bc3ClassifierLibrary`.
- El catálogo BC3 y los prompts YAML viajan dentro del paquete.
- Se conservan las variables de configuración principales por `.env`.
- Se mantiene un CLI de compatibilidad (`ruesma-bc3-classify-stdin`) solo para pruebas.

## Instalación local en el entorno del servicio 1

```bash
pip install -e .\servicio_2_bc3_library
```

o bien:

```bash
pip install .\servicio_2_bc3_library
```

## Uso desde Python

```python
from ruesma_ocr_service import Bc3ClassifierLibrary

classifier = Bc3ClassifierLibrary.from_env()
response = classifier.classify(payload)
```

## Variables de entorno relevantes

Consulta `.env.example`.

## Empaquetado con PyInstaller del servicio 1

Asegúrate de recoger también los datos del paquete:

```bash
pyinstaller --collect-data ruesma_ocr_service main_gui.py
```

## Nota importante

Los YAML incluidos en `ruesma_ocr_service/resources/` son plantillas de ejemplo para
que el paquete sea autocontenido. Sustitúyelos por tus YAML reales del servicio 2
antes de generar el wheel definitivo.
