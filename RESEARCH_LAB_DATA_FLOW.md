# Research Lab — flujo de datos limpios

Research Lab no descarga datos directamente de MT5.

1. Gráficos carga el activo/temporalidad desde MT5.
2. Calidad de datos detecta huecos sospechosos.
3. El usuario puede eliminarlos sin interpolar ni inventar velas.
4. `Enviar datos limpios a Research Lab` persiste la serie en `data/research_lab/clean_datasets/` y la registra en `clean_dataset_registry.json`.
5. Research Lab solo muestra activos registrados para la temporalidad global.
6. Cambiar el rango en Research Lab solo recorta la serie limpia local. Nunca dispara una llamada a MT5.
