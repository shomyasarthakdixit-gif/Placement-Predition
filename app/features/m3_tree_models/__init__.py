import importlib
import os
from flask import Blueprint

m3_bp = Blueprint('m3_tree_models', __name__)

for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    importlib.import_module(f'.{module[:-3]}', package=__name__)
