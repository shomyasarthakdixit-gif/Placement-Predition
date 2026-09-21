import importlib
import pkgutil
from flask import Blueprint

m2_bp = Blueprint('m2_linear_models', __name__)

import os
for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    importlib.import_module(f'.{module[:-3]}', package=__name__)
