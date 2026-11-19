"""Safe classification for provider failures."""

import socket


SAFE_ERROR_CODES = {
    "authorization",
    "authentication",
    "configuration",
    "connectivity",
    "invalid_response",
    "not_configured",
    "provider_error",
    "rate_limit",
    "timeout",
}

SAFE_EXCEPTION_TYPES = frozenset({
    "ApiRequestFailure",
    "AuthenticationError",
    "InvalidCredentialsError",
    "WMLClientError",
    "ForbiddenActionForPlan",
    "ForbiddenActionForGitBasedProject",
    "MissingValue",
    "MissingWMLComponent",
    "MissingFoundationModel",
    "NoWMLCredentialsProvided",
    "MissingArgument",
    "InvalidMultipleArguments",
    "CannotSetProjectOrSpace",
    "SpaceIDandProjectIDCannotBeNone",
    "CannotAutogenerateBedrockUrl",
    "ParamOutOfRange",
    "InvalidValue",
    "UnexpectedType",
    "NotUrlNorID",
    "WrongEnvironmentVersion",
    "InvalidURL",
    "UnsupportedProtocol",
    "HTTPStatusError",
    "TimeoutError",
    "TimeoutException",
    "ConnectTimeout",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
    "ConnectError",
    "NetworkError",
    "RequestError",
    "RemoteProtocolError",
    "ProxyError",
    "SSLError",
    "ConnectionError",
    "ConnectionResetError",
    "ConnectionAbortedError",
    "BrokenPipeError",
    "OSError",
    "ValueError",
    "TypeError",
    "OtherError",
})

_CONFIGURATION_EXCEPTION_TYPES = {
    "missingvalue",
    "missingwmlcomponent",
    "missingfoundationmodel",
    "nowmlcredentialsprovided",
    "missingargument",
    "invalidmultiplearguments",
    "cannotsetprojectorspace",
    "spaceidandprojectidcannotbenone",
    "cannotautogeneratebedrockurl",
    "paramoutofrange",
    "invalidvalue",
    "unexpectedtype",
    "noturlnorid",
    "wrongenvironmentversion",
    "invalidurl",
    "unsupportedprotocol",
    "valueerror",
    "typeerror",
}


def _exception_type_names(error) -> set[str]:
    return {exception_type.__name__.lower() for exception_type in type(error).__mro__}


def safe_exception_name(error) -> str:
    """Return an allowlisted exception class name, never its message."""
    for exception_type in type(error).__mro__:
        if exception_type.__name__ in SAFE_EXCEPTION_TYPES:
            return exception_type.__name__
    return "OtherError"


def safe_http_status(error) -> int | None:
    """Extract only a numeric HTTP status; never inspect response content."""
    response = getattr(error, "response", None)
    candidates = (
        getattr(error, "status_code", None),
        getattr(response, "status_code", None) if response is not None else None,
        getattr(error, "code", None),
    )
    for candidate in candidates:
        if isinstance(candidate, bool):
            continue
        if isinstance(candidate, int):
            status_code = candidate
        elif isinstance(candidate, str) and candidate.isdecimal():
            status_code = int(candidate)
        else:
            continue
        if 100 <= status_code <= 599:
            return status_code
    return None


def classify_provider_error(error) -> str:
    """Return a non-sensitive category without inspecting exception messages."""
    if isinstance(error, str) and error in SAFE_ERROR_CODES:
        return error

    error_types = _exception_type_names(error)
    status_code = safe_http_status(error)

    if status_code in (408, 504) or any("timeout" in name for name in error_types):
        return "timeout"
    if status_code == 403 or error_types & {
        "forbiddenactionforplan",
        "forbiddenactionforgitbasedproject",
    } or any(
        marker in name
        for name in error_types
        for marker in ("forbidden", "authorization", "permissiondenied", "notauthorized")
    ):
        return "authorization"
    if status_code == 401 or any(
        marker in name
        for name in error_types
        for marker in ("auth", "credential", "unauthenticated")
    ):
        return "authentication"
    if status_code == 429 or any(
        marker in name
        for name in error_types
        for marker in ("ratelimit", "rate_limit", "resourceexhausted")
    ):
        return "rate_limit"
    if status_code in (502, 503) or isinstance(
        error, (ConnectionError, socket.error)
    ) or any(
        marker in name
        for name in error_types
        for marker in (
            "connection",
            "connecterror",
            "networkerror",
            "network",
            "requesterror",
            "remoteprotocolerror",
            "proxyerror",
            "sslerror",
            "transporterror",
            "socketerror",
            "connectionreset",
            "connectionaborted",
            "brokenpipe",
        )
    ):
        return "connectivity"
    if status_code in (400, 404, 422) or error_types & _CONFIGURATION_EXCEPTION_TYPES:
        return "configuration"
    return "provider_error"
