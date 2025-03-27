# AutoPitch Trainer v2.0

**by asoqwer**

Welcome to AutoPitch Trainer v2.0! This utility trains machine learning models to automatically generate pitch bends for USTX files used in [OpenUtau](https://github.com/stakira/OpenUtau). Learn from your manually tuned files and save time applying similar pitches to new projects.

This README provides a quick overview and setup guide. **For detailed step-by-step instructions, explanations, and recommendations, please refer to the full tutorial:**

➡️ [**AutoPitch Trainer v2.0 Full Tutorial (Google Docs)**](https://docs.google.com/document/d/1Eb43g7Tc616YRtyfLEqrwGKLS5af238-KsGQoY06oBs/edit?usp=sharing) ⬅️

## Features

*   Train a pitch generation model from scratch using your USTX files.
*   Fine-tune a pre-trained model on a new dataset.
*   Resume interrupted training sessions.
*   Process new USTX files to apply pitch bends automatically.
*   Convert trained models to ONNX format for use in UtauV.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/emeraldsingers/AutoPitchTrainer
    cd autopitch_trainer
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install PyTorch with CUDA (Highly Recommended):**
    For significantly faster training on compatible NVIDIA GPUs, install PyTorch with CUDA. Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to get the command matching your system. For CUDA 11.8:
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    For CPU-only installation (much slower):
    ```bash
    pip3 install torch torchvision torchaudio
    ```

4.  **Install Other Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

1.  **Launch the Trainer:**
    ```bash
    python main.py
    ```
2.  **Select a Tab:** The GUI has several tabs for different tasks:
    *   **Train:** Build a new model from scratch using a directory of tuned USTX files.
    *   **Fine-Tune:** Adapt an existing model (`.pth` + `phoneme_vocab.yaml`) using a new set of USTX files.
    *   **Resume Training:** Continue a previous training session using the saved checkpoint (`last_autopitch_model.pth`) and original data.
    *   **Process:** Apply pitch bends from a trained model (`.pth` + `phoneme_vocab.yaml`) to one or more USTX files.
        *   **Note:** The "Best Loss" model option may not function as expected, as the trainer might not save this specific checkpoint correctly. Using "Last" is generally recommended.
    *   **Convert to ONNX:** Export a trained model (`last_autopitch_model.pth` + `phoneme_vocab.yaml`) to the `.onnx` format needed by UtauV.
3.  **Follow On-Screen Instructions:** Each tab has fields for specifying input/output directories, model files, epochs, etc. Use the "Browse" buttons to select paths.
4.  **Need More Details?** For specific steps on preparing data, recommended settings, and understanding the process for each tab, **consult the [Full Tutorial](https://docs.google.com/document/d/1Eb43g7Tc616YRtyfLEqrwGKLS5af238-KsGQoY06oBs/edit?usp=sharing)**.

## Using with UtauV

*   After converting your model to ONNX (`autopitch.onnx` and `phoneme_vocab.yaml`), you need to place these files in the correct UtauV dependency folder (`Dependencies/autopitch`).
*   You might also need to configure the `noiseScale` in `Dependencies/autopitch/config.json` within UtauV to control pitch variation.
*   **See the [Full Tutorial](https://docs.google.com/document/d/1Eb43g7Tc616YRtyfLEqrwGKLS5af238-KsGQoY06oBs/edit?usp=sharing#heading=h.1jc0a81yjd79) for detailed UtauV installation and usage instructions.**

## Troubleshooting

*   **"Error loading AutoPitch" (in UtauV):** Ensure `autopitch.onnx` and `phoneme_vocab.yaml` are correctly placed in `Dependencies/autopitch`.
*   **No Pitch Bends Applied (in UtauV):** Make sure notes are selected before applying AutoPitch, or verify the quality of your trained model and training data.
*   **For other issues:** Check the Troubleshooting section in the [Full Tutorial](https://docs.google.com/document/d/1Eb43g7Tc616YRtyfLEqrwGKLS5af238-KsGQoY06oBs/edit?usp=sharing#heading=h.k7j7bpl9l710).

## Support

For questions or support, contact **asoqwer** via:

*   Email: `emeraldproject13@gmail.com`
*   Discord: `asoqwer`
*   Telegram: [Emerald Project Support Bot](https://t.me/EmeraldProjectSupport_bot)

## Additional Links

*   **Emerald Project:** [emeraldsingers.github.io](https://emeraldsingers.github.io/)
*   **UtauV Releases:** [Github Releases](https://github.com/emeraldsingers/UtauV/releases/latest)
*   **PreTrained AutoPitch models:** [Mega.NZ](https://mega.nz/folder/rilkRawA#urkiXGT1SsuJLhquWJegoQ)
*   **Emerald Project Telegram:** [Telegram Channel](https://t.me/UtauV)
*   **asoqwer on YT:** [asoqwer - YouTube](https://www.youtube.com/@asoqwer)

## License

This project is licensed under the GNU GPLv3 License - see the `LICENSE` file for details.
