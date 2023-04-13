import asyncio
import importlib
import os
import argparse

from llm_repl.repls import REPLS
from llm_repl.llms import LLMS

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


def main():
    parser = argparse.ArgumentParser(description="LLM REPL")
    parser.add_argument(
        "--llm",
        type=str,
        default="chatgpt",
        help="The LLM model to use (DEFAULT: chatgpt))",
        choices=LLMS.keys(),
    )
    parser.add_argument(
        "--repl",
        type=str,
        default="prompt_toolkit",
        help="The REPL interface to use (DEFAULT: prompt_toolkit)",
        choices=REPLS.keys(),
    )
    parser.add_argument(
        "--port", type=int, help="The port to connect to the LLM server", default=8000
    )

    args = parser.parse_args()

    repl = REPLS[args.repl](port=args.port)
    asyncio.get_event_loop().run_until_complete(repl.run(args.llm))
