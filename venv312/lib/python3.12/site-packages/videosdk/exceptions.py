class SDKError(Exception):
    """Base class for all SDK exceptions."""
    pass

class InternalServerError(SDKError):
    """Raised when there is an unhandled error error."""
    def __init__(self, message="Internal Server Error"):
        self.message = message
        super().__init__(self.message)
        
class MeetingInitializationError(SDKError):
    """Raised when there is an error in meeting initialization."""
    def __init__(self, message="Failed to initialize meeting"):
        self.message = message
        super().__init__(self.message)

class InvalidMeetingConfigError(SDKError):
    """Raised when the meeting configuration is invalid."""
    def __init__(self, message="Invalid meeting configuration"):
        self.message = message
        super().__init__(self.message)

class ValidationError(SDKError):
    """Raised when the Validation Failed."""
    def __init__(self, message="Validation failed"):
        self.message = message
        super().__init__(self.message)

class RTCError(SDKError):
    """Raised when there is RTC error or server side error."""
    def __init__(self, message="Error in RTC"):
        self.message = message
        super().__init__(self.message)
        
class MeetingError:
    """class of meeting error."""
    code: int
    type: str
    message: str