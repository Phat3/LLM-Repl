# LLM REPL

## What is this?

The goal of this project is to create a simple, interactive **REPL** (Read-Eval-Print-Loop) that allows users to interact with a variety of Large Language Models (**LLMs**). The project is mainly built on top of two Python libraries: [langchain](https://github.com/hwchase17/langchain), which provides a convenient and flexible interface for working with LLMs, and [rich](https://github.com/Textualize/rich) which provides a user-friendly interface for the REPL.
The REPL can also be launched in **headledss** mode and it can be interacted with using **websocket**.

Currently, the project is in development and only supports interaction with the ChatGPT but it has been structure to make it easy to extend it use any LLMs, including custom ones (by extending `BaseLLM` in `./src/llm_repl/llms/__init__.py`).

ChatGPT can be interacted by using the models `gpt-3.5-turbo` and `gpt4` (For users who got GPT-4 API beta).

## Features

The REPL supports the following features:

### Streaming Mode

The REPL won't wait for the model to finish generating the output, but it will start printing the output as soon as it is available.

![Streaming Mode](./docs/gifs/streaming_mode.gif)

### Conversation Memory

The REPL supports conversation memory. This means that the model will remember the previous conversation and will use it to generate the next response.

![Memory](./docs/gifs/memory.gif)

### Pretty Printing

The REPL supports Markdown rendering both of the input and the output.

PS: In this initial version of the REPL, the full Markdown syntax is only when running the tool in `non-streaming` mode. In `streaming` mode only code sections will be pretty printed.

![Pretty Printing](./docs/gifs/pretty_printing.gif)

## Headless Mode

The REPL can be run in headless mode. This means that it can be interacted with using a websocket. This is useful for integrating the REPL with other applications / other UIs.

To launch the REPL in headless mode, run the following command:

```bash
llm-repl --repl websocket --port <PORT>
```

### Model Switching on the Fly

**COMING SOON...**

### Conversation History

**COMING SOON...**

## Installation

```bash
pip install llm-repl
```

## Usage

First export your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=<OPENAI_KEY>
```

Then run the REPL:

```bash
llm-repl
```

Or if you want to use a specific model:

```bash
llm-repl --llm chatgpt4
```

### Run inside Docker

```bash
docker run -it --rm -e OPENAI_API_KEY=<OPENAI_KEY> phate/llm-repl
```

Or if you want to source the environment variables from a file, first create a file called `.env` with the following content:

```bash
OPENAI_API_KEY=<OPENAI_KEY>
```

And then run the following command:

```bash
docker run -it --rm --env-file .env phate/llm-repl
```

## Development

To install the REPL in development mode

Then install the package in development mode:

```bash
pip install -e ".[DEV]"
```

Before contributing, please make sure to run the following commands:

```bash
pre-commit install
```
