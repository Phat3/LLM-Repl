import os
import importlib.util

LLMS_DIR = os.path.join(os.path.dirname(__file__), "llms")
REPLS_DIR = os.path.join(os.path.dirname(__file__), "repls")


def load_modules_from_directory(path):
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".py") and not file.startswith("__init__"):
                module_name = file[:-3]
                module_path = os.path.join(root, file)

                # Load the module
                spec = importlib.util.spec_from_file_location(module_name, module_path)  # type: ignore
                module = importlib.util.module_from_spec(spec)  # type: ignore
                spec.loader.exec_module(module)


load_modules_from_directory(LLMS_DIR)
load_modules_from_directory(REPLS_DIR)
