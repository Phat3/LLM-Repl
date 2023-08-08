import asyncio
import argparse

from llm_repl.repls import REPLS
from llm_repl.llms import LLMS


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
