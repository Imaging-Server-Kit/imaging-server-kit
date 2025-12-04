import importlib.util
from typing import Optional

class AlgorithmNotFoundError(Exception):
    """Exception raised when a specified algorithm is not found."""

    def __init__(self, algorithm: str, message="Algorithm not found"):
        self.algorithm_name = algorithm
        self.message = f"{message}: {algorithm}"
        super().__init__(self.message)


class AlgorithmServerError(Exception):
    """Exception raised when a request to an algorithm server returns an unexpected status code."""

    def __init__(
        self,
        status_code,
        response_body,
        message="Algorithm server returned an unexpected status_code",
    ):
        self.status_code = status_code
        self.response_body = response_body
        self.message = f"{message}: {status_code}\nResponse: {response_body}"
        super().__init__(self.message)


class AlgorithmTimeoutError(Exception):
    """Exception raised when a request to an algorithm server exceeds the time limit."""

    def __init__(
        self,
        status_code,
        response_text,
        message="Algorithm server returned an unexpected status_code",
    ):
        self.status_code = status_code
        self.response_text = response_text
        self.message = f"{message}: {status_code}\nResponse: {response_text}"
        super().__init__(self.message)


class InvalidAlgorithmParametersError(Exception):
    """Exception raised when a request to an algorithm server is made with invalid algorithm parameters."""

    def __init__(
        self,
        status_code,
        response_text,
        message="Algorithm parameters were invalidated by the server",
    ):
        self.status_code = status_code
        self.response_text = response_text
        pydantic_details = response_text.get("detail")[0]
        pydantic_validation_msg = pydantic_details.get("msg")
        pydantic_failing_param = pydantic_details.get("loc")[0]
        pydantic_failing_param_value = pydantic_details.get("input")
        self.message = f"{message}: Parameter: {pydantic_failing_param}. {pydantic_validation_msg}. Received: {pydantic_failing_param_value}."
        super().__init__(self.message)


class ServerRequestError(Exception):
    """Exception raised when HTTP requests fail."""

    def __init__(self, url: str, error: Exception, message="Request to server failed"):
        self.url = url
        self.error = error
        self.message = f"{message} ({url=}): {error}"
        super().__init__(self.message)


class AlgorithmStreamError(Exception):
    """Exception raised when an algorithm is incorrectly used as a stream."""

    def __init__(
        self, algorithm: str, message="Algorithm incorrectly used as a stream."
    ):
        self.algorithm_name = algorithm
        self.message = f"{algorithm}: {message}"
        super().__init__(self.message)


class AlgorithmRuntimeError(Exception):
    def __init__(
        self,
        algorithm: str,
        error: Optional[Exception] = None,
        message="Algorithm did not run successfully. ",
    ):
        self.message = message + f"{algorithm=}"
        if error is not None:
            self.message = self.message + ", Error: {error}"
        super().__init__(self.message)


def napari_available():
    """Check if napari-serverkit is installed."""
    spec = importlib.util.find_spec("napari_serverkit")
    return spec is not None
