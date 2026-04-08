import asyncio
import fractions
import time
from vsaiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av import AudioFrame
from av.frame import Frame
from av.packet import Packet

AUDIO_PTIME = 0.020

class CustomAudioTrack(MediaStreamTrack):
    """
    A dummy audio track which reads silence.
    """

    kind = "audio"

    _start: float
    _timestamp: int

    async def recv(self) -> Frame:
        """
        Receive the next :class:`~av.audio.frame.AudioFrame`.

        The base implementation just reads silence, subclass
        :class:`AudioStreamTrack` to provide a useful implementation.
        """
        if self.readyState != "live":
            raise MediaStreamError

        sample_rate = 8000
        samples = int(AUDIO_PTIME * sample_rate)

        if hasattr(self, "_timestamp"):
            self._timestamp += samples
            wait = self._start + (self._timestamp / sample_rate) - time.time()
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0

        frame = AudioFrame(format="s16", layout="mono", samples=samples)
        for p in frame.planes:
            p.update(bytes(p.buffer_size))
        frame.pts = self._timestamp
        frame.sample_rate = sample_rate
        frame.time_base = fractions.Fraction(1, sample_rate)
        
        return frame

