import sys
import importlib
import os
import argparse
import pydantic

from llm_repl.repls import REPLStyle
from llm_repl.repls.prompt_toolkit import PromptToolkitRepl
from llm_repl.llms import MODELS

LLMS_DIR = os.path.join(os.path.dirname(__file__), "llms")

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


def main():
    parser = argparse.ArgumentParser(description="LLM REPL")
    parser.add_argument(
        "--llm",
        type=str,
        default="chatgpt",
        help="The LLM model to use",
        choices=MODELS.keys(),
    )

    args = parser.parse_args()

    try:
        style = REPLStyle(**CONFIGS)
    except pydantic.ValidationError:
        print("Invalid style configuration")
        sys.exit(1)

    repl = PromptToolkitRepl(style=style)

    # FIXME: Move this in a proper initialization function
    # Enable multiline with double enter
    @repl.kb.add("enter")
    def _(event):
        repl.handle_enter(event)

    # Exit gracefully with Ctrl+D
    @repl.kb.add("c-d")
    def _(_):
        repl.exit()

    # Exit gracefully with Ctrl+C
    @repl.kb.add("c-c")
    def _(_):
        repl.exit()

    # Run the REPL
    repl.run(args.llm)
