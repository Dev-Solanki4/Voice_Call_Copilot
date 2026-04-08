from vsaiortc.mediastreams import MediaStreamTrack


class Stream:
    """
    Represents a media stream, initialized with a MediaStreamTrack.

    Attributes:
        track (MediaStreamTrack): The media stream track associated with the stream.
        id (str): The ID of the stream, derived from the track.
        kind (str): The kind of stream (e.g., 'audio', 'video').
        codecs (None): Placeholder for codecs information, currently not initialized.
    """

    def __init__(self, track: MediaStreamTrack) -> None:
        """
        Initialize a Stream object with a given MediaStreamTrack.

        Args:
            track (MediaStreamTrack): The media stream track associated with the stream.
        """
        self.track = track
        self.id = track.id
        self.kind = track.kind
        self.codecs = None  # Placeholder for codecs information

    def __str__(self) -> str:
        """
        Return a string representation of the Stream object.

        Returns:
            str: String representation containing stream ID and kind.
        """
        return f"Stream(ID: {self.id}, Kind: {self.kind})"
