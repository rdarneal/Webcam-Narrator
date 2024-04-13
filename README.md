# AI Narration System

This project provides an AI narration system that processes images using OpenAI and generates narrations for the content of the images. Additionally, it allows users to process audio from YouTube videos and extract segments for training voice models with ElevenLabs.

## Features

- Processes images using OpenAI's natural language processing capabilities.
- Generates narrations in various voices using ElevenLabs' text-to-speech service.
- Supports real-time image processing from a webcam stream.
- Extracts audio from YouTube videos for voice model training.

## Requirements

- Python 3.x
- OpenAI API key
- ElevenLabs API key
- OpenCV
- dotenv
- narrator

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/your_username/ai-narration-system.git
    cd ai-narration-system
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables by creating a `.env` file in the project root directory and adding the following:

    ```plaintext
    OPENAI_API_KEY=your_openai_api_key
    ELEVEN_API_KEY=your_elevenlabs_api_key
    VOICE_ID=your_default_voice_id
    ```

4. Run the `main.py` script:

    ```bash
    python main.py
    ```

5. Optionally, you can specify a different voice ID using the `-voice_id` argument:

    ```bash
    python main.py -voice_id your_voice_id
    ```

6. If you want to modify the system message, you can provide the file path location of a .txt file containing the new message using the `-txt_file` argument:

    ```bash
    python main.py -txt_file /path/to/system_message.txt
    ```

    The system will read the content of the text file and use it as the new system message.

## Note

This project is for educational purposes only. It demonstrates the capabilities of natural language processing and text-to-speech synthesis using AI technologies. It's important to use AI responsibly and ethically.

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
