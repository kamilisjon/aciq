#!/usr/bin/env bash
set -e

echo "==> ruff format ."
ruff format .
echo
echo "==> ruff check --fix ."
ruff check --fix .