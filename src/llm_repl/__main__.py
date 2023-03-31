import argparse

from llm_repl.repl import LLMRepl
from llm_repl.llms import MODELS

# FIXME: This is temporary for test. This will be passed in the configuration file
CONFIGS = {
    "style": {
        "client": {"color": "bold green", "padding": (1, 1)},
        "server": {"color": "bold blue", "padding": (1, 1)},
    }
}


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

    repl = LLMRepl(config=CONFIGS)

    # Enable multiline with double enter
    @repl.kb.add("enter")
    def _(event):
        repl.handle_enter(event)

    # Exit gracefully with Ctrl+D
    @repl.kb.add("c-d")
    def _(_):
        repl._exit()

    # Exit gracefully with Ctrl+C
    @repl.kb.add("c-c")
    def _(_):
        repl._exit()

    # Run the REPL
    repl.run(MODELS[args.llm])
