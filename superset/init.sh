#!/bin/bash
set -e

INIT_FLAG="/app/superset_home/.initialized"

if [ ! -f "$INIT_FLAG" ]; then
    echo "[init] First boot — initializing Superset..."

    superset db upgrade

    superset fab create-admin \
        --username admin \
        --firstname Admin \
        --lastname User \
        --email admin@example.com \
        --password admin

    superset init

    echo "[init] Registering SQLite database connection..."
    superset legacy-import-datasources -p /app/datasource.yaml

    touch "$INIT_FLAG"
    echo "[init] Done. Superset initialized."
else
    echo "[init] Already initialized, skipping setup."
fi

exec superset run -h 0.0.0.0 -p 8088 --with-threads --reload --debugger
