<!-- README_COMPILACION_LIBRERIA.md -->
# Librería servicio 2 y recursos internos

## Objetivo

Los YAML del catálogo y de prompts deben viajar dentro de la librería para que el
servicio 1 compilado no pida al usuario final que seleccione ficheros.

## Estructura obligatoria

```text
ruesma_ocr_service/
  resources/
    bc3_catalog.yaml
    prompts.yaml
```

## Reglas

- Sustituye esos YAML por los reales antes de compilar.
- No dependas de rutas absolutas del proyecto para la distribución final.
- El `pyproject.toml` y `MANIFEST.in` ya están preparados para incluirlos como
  `package_data`.

## Desarrollo

Puedes seguir usando:

```bat
pip install -e ..\ocr_service
```

## Distribución final

El servicio 1 los arrastra automáticamente mediante PyInstaller usando:

- `collect_data_files("ruesma_ocr_service.resources")`

No hace falta que el usuario final vea ni seleccione los YAML.
