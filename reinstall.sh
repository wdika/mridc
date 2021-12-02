#!/usr/bin/env bash
set -e

INSTALL_OPTION=${1:-"dev"}

PIP=pip

echo 'Uninstalling old versions...'
${PIP} uninstall -y -q -r requirements/requirements.txt
${PIP} uninstall -y mridc

echo 'Installing new versions...'
${PIP} install -r requirements/requirements.txt
${PIP} install -U setuptools

echo 'Installing mridc'
if [[ "$INSTALL_OPTION" == "dev" ]]; then
    ${PIP} install --editable ".[all]"
else
    rm -rf dist/
    python setup.py bdist_wheel
    DIST_FILE=$(find ./dist -name "*.whl" | head -n 1)
    ${PIP} install "${DIST_FILE}[all]"
fi

if [ -x "$(command -v conda)" ]; then
  echo 'Installing numba'
  conda install -y -c numba numba.
fi

echo 'Finished!'
