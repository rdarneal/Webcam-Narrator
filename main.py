import cv2
import os
import argparse
from dotenv import find_dotenv, load_dotenv
from threading import Thread
from narrator.api import analyze_image, get_system_message
from narrator.stream import VideoStream
from openai import OpenAI
from elevenlabs.client import ElevenLabs


def main():
    load_dotenv((find_dotenv()))
    parser = argparse.ArgumentParser(
        description="Process an image using OpenAI and ElevenLabs."
    )
    parser.add_argument(
        "-voice_id",
        dest="voice",
        help="Specify the voice id, user the narrator.ipynb if you are unsure.",
    )
    parser.add_argument(
        "-txt_file",
        dest="txt_file",
        help="Folder location of the .txt file containing the system message.",
    )
    args = parser.parse_args()

    # Get the voice_id from args, else from the .env file
    if args.voice:
        voice = args.voice
    else:
        voice = os.getenv("VOICE_ID")

    # Allow entry of system message from text file
    system_message = get_system_message(args.txt_file)

    # Folder
    folder = "frames"

    # Create the frames folder if it doesn't exist
    frames_dir = os.path.join(os.getcwd(), folder)
    os.makedirs(frames_dir, exist_ok=True)

    # initiate OpenAI client
    oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # initiate eleven labs client
    el_client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))

    # Begin webcam
    print("starting threaded video stream...")
    vs: VideoStream = VideoStream(0).start()
    cv2.namedWindow("Narrator V1.0", cv2.WINDOW_NORMAL)

    # instantiate callback variables for Threading API calls
    processing = False
    message = ""

    def update_message_callback():
        # Used to reset callback variables after API processing
        nonlocal processing, message
        processing = False
        message = ""

    while True:
        img = vs.read()

        # Update message
        cv2.putText(
            img,
            message,
            (7, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (100, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("output", img)

        if cv2.waitKey(1) == ord("q"):
            break

        elif cv2.waitKey(1) == ord("d") and not processing:
            processing = True
            message = "Analyzing Image..."
            Thread(
                target=analyze_image,
                args=(
                    oai_client,
                    el_client,
                    img,
                    update_message_callback,
                    voice,
                    system_message,
                ),
            ).start()

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    main()
