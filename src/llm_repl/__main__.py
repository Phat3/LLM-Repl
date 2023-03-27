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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM REPL")
    parser.add_argument(
        "--model",
        type=str,
        default="chatgpt",
        help="The LLM model to use",
        choices=MODELS.keys(),
    )

    args = parser.parse_args()

    repl = LLMRepl(config=CONFIGS)

    # Add key bindings hooks
    @repl.kb.add("enter")
    def _(event):
        repl.handle_enter(event)

    # Run the REPL
    repl.run(MODELS[args.model])
