import sys
import asyncio
import importlib
import os
import argparse
import pydantic

from llm_repl.repls import REPLStyle, REPLS
from llm_repl.llms import LLMS 

LLMS_DIR = os.path.join(os.path.dirname(__file__), "llms")
REPLS_DIR = os.path.join(os.path.dirname(__file__), "repls")

# FIXME: This is temporary for test. This will be passed in the configuration file
CONFIGS = {
    "client_msg_color": "bold green",
    "server_msg_color": "bold blue",
    "error_msg_color": "bold red",
    "misc_msg_color": "gray",
}


# Load all the LLMs in the llms/ directory
for file_name in os.listdir(LLMS_DIR):
    if file_name.endswith(".py") and file_name != "__init__.py":
        module_name = file_name[:-3]
        module = importlib.import_module(f"llm_repl.llms.{module_name}")

# Load all the REPLs in the repls/ directory
for file_name in os.listdir(REPLS_DIR):
    if file_name.endswith(".py") and file_name != "__init__.py":
        module_name = file_name[:-3]
        module = importlib.import_module(f"llm_repl.repls.{module_name}")


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

    args = parser.parse_args()

    try:
        style = REPLStyle(**CONFIGS)
    except pydantic.ValidationError:
        print("Invalid style configuration")
        sys.exit(1)

    repl = REPLS[args.repl](style=style)
    asyncio.get_event_loop().run_until_complete(repl.run(args.llm))
