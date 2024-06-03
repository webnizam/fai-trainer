rm -rf dist build fai_trainer.egg*
python setup.py sdist bdist_wheel
python -m twine upload --skip-existing dist/* --verbose