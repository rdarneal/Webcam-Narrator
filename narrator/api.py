import base64
import errno
import os
import cv2

import numpy as np

from typing import Callable
from openai import OpenAI
from elevenlabs import stream
from elevenlabs.client import ElevenLabs


def encode_image(
    image: np.ndarray, save_path: str | os.PathLike = "frames/frame.jpg"
) -> bytes:
    """
    Encode an image into a base64-encoded string.

    Args:
        image (numpy.ndarray): The image to encode.
        save_path (str or os.PathLike, optional): The path to save the resized image.
            Defaults to "frames/frame.jpg".

    Returns:
        bytes: The base64-encoded image.

    Raises:
        IOError: If failed to capture the image.

    Example:
        >>> encoded_image = encode_image(image)
    """
    height = image.shape[0]
    width = image.shape[1]
    new_width = 250
    ratio = new_width / width
    new_height = int(height * ratio)
    dimensions = (new_width, new_height)
    try:
        image_to_save = cv2.resize(image, dimensions, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(save_path, image_to_save)
        _, buffer = cv2.imencode(".jpg", image_to_save)
        b64img = base64.b64encode(buffer).decode("utf-8")
        # print(type(b64img), type(buffer))
        return b64img
    except IOError as e:
        if e.errno != errno.EACCES:
            raise
        print("Failed to capture image, retry in a moment")


def get_system_message(txt_file):
    """
    Extracts the system message from the specified text file.

    Args:
        txt_file (str): Folder location of the .txt file containing the system message.

    Returns:
        str: The system message extracted from the text file, or the default message if the file cannot be processed.
    """
    default_message = "You are Sir David Attenborough. Narrate the picture as if it is a nature documentary. Make it snarky and funny. Don't repeat yourself. Make it short. If I do anything remotely interesting, make a big deal about it!"

    if txt_file and os.path.exists(txt_file):
        try:
            with open(txt_file, "r") as file:
                return file.read().strip()
        except Exception as e:
            print(f"Error reading text file: {e}")
            return default_message
    else:
        return default_message


def analyze_image(
    openai_client: OpenAI,
    elevenlabs_client: ElevenLabs,
    image: np.ndarray,
    update_message_callback: Callable,
    voice: str,
    system_message: str,
    max_tokens: int = 500,
    script: list = [],
) -> None:
    """
    Analyze an image using OpenAI and generate corresponding narration.

    Args:
        openai_client (OpenAI): The OpenAI client.
        elevenlabs_client (ElevenLabs): The ElevenLabs client.
        image (numpy.ndarray): The image to analyze.
        update_message_callback (Callable): Callback function to update the processing message.
        voice (str): The voice code.
        system (str): The system message you want to use for OpenAI.
        max_tokens (int, optional): Maximum tokens for OpenAI completion. Defaults to 500.
        script (list, optional): A list of previous conversation turns. Defaults to [].

    Returns:
        None

    Example:
        >>> analyze_image(openai_client, elevenlabs_client, image, update_message_callback, voice)
    """
    base64_image = encode_image(image)
    new_line = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Describe this image"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
            ],
        },
    ]

    messages = (
        [
            {
                "role": "system",
                "content": system_message,
            },
        ]
        + script
        + new_line
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=max_tokens,
    )

    update_message_callback()

    response_text = response.choices[0].message.content
    print(response_text)

    appended_script = script + [{"role": "assistant", "content": response_text}]

    audio = elevenlabs_client.generate(text=response_text, voice=voice, stream=True)

    stream(audio)

    return appended_script
