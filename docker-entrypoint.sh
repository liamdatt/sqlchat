#!/usr/bin/env bash
set -e

python manage.py collectstatic --noinput
python manage.py migrate --noinput

# Project module is "sql_ai_project"
exec gunicorn sql_ai_project.wsgi:application \
  --bind 0.0.0.0:8000 \
  --workers 3 \
  --timeout 90
