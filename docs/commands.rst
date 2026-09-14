Comandos
========

El Makefile contiene los puntos de entrada centrales para las tareas comunes relacionadas con este proyecto.

Sincronización de datos con S3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `make sync_data_to_s3` usa `aws s3 sync` para sincronizar de forma recursiva los archivos de `data/` hacia `s3://[OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')/data/`.
* `make sync_data_from_s3` usa `aws s3 sync` para sincronizar de forma recursiva los archivos desde `s3://[OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')/data/` hasta `data/`.