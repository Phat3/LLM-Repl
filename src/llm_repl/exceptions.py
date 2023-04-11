# -------------------- LLMS EXCEPTIONS --------------------


class LLMException(Exception):
    """Base class for exceptions raised by LLMs."""

    def __init__(self, msg: str):
        self.msg = msg
        super().__init__(f"{msg}")


class LLMNotFound(LLMException):
    """Exception raised when an LLM is not found."""

    def __init__(self, llm_name: str):
        super().__init__(f"LLM '{llm_name}' not found.")


class MissingAPIKey(LLMException):
    """Exception raised when an API key is missing."""

    def __init__(self, msg: str):
        super().__init__(
            f"{msg} not found, please set it in your environment variables."
        )
