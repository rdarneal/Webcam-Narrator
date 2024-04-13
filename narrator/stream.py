from threading import Thread
import cv2


class VideoStream:
    """
    Class for reading frames from a video stream using multi-threading.

    Attributes:
        stream: cv2.VideoCapture: Video capture object.
        grabbed (bool): Indicates whether the frame was successfully grabbed.
        frame: numpy.ndarray: The current frame.
        stopped (bool): Indicates whether the video stream has stopped.

    Methods:
        __init__(self, src=0): Initializes the VideoStream object.
        start(self): Starts the video stream thread.
        update(self): Method called by the thread to continuously update frames.
        read(self): Returns the current frame.
        stop(self): Stops the video stream.
    """

    def __init__(self, src=0):
        """
        Initializes the VideoStream object.

        Args:
            src (int or str): Video source index or filename. Defaults to 0.
        """
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        self.stopped: bool = False

    def start(self):
        """Starts the video stream thread."""
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        """Method called by the thread to continuously update frames."""
        while True:
            if self.stopped:
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        """Returns the current frame."""
        return self.frame

    def stop(self):
        """Stops the video stream."""
        self.stopped = True
