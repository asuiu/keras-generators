del ./dist/*.whl
python setup.py bdist_wheel %*
twine upload dist/*.whl --verbose