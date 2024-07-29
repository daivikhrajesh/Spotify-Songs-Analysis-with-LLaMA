# Spotify Songs Analysis with LLaMA

This project demonstrates how to use the LLaMA model from Hugging Face to analyze and generate insights from a dataset of Spotify songs. The script processes song data from India and uses a pre-trained LLaMA model to answer questions about the dataset.

## Prerequisites

Make sure you have the following installed:

- Python 3.7 or higher
- Pip

You also need to install the required Python packages. The script uses:

- `pandas` for data manipulation
- `transformers` for working with the Hugging Face models
- `torch` for PyTorch operations
- `huggingface_hub` for model authentication

## Setup

1. **Clone the Repository**

   Clone this repository to your local machine:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**

   Install the required Python packages using pip:

   ```bash
   pip install pandas transformers torch huggingface_hub
   ```

3. **Get Your Hugging Face Token**

   Ensure you have a Hugging Face token. If you don't have one, you can obtain it by creating an account at [Hugging Face](https://huggingface.co/) and generating a token from your account settings.

4. **Update the Token in the Script**

   Replace `"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"` in the script with your Hugging Face token.

5. **Prepare the Dataset**

   Ensure the dataset `universal_top_spotify_songs.csv` is located in the same directory as the script. The dataset should include columns such as 'name', 'artists', 'daily_rank', 'daily_movement', etc.

## Running the Script

Run the script using Python:

```bash
python main.py
```

The script will:

1. Authenticate with Hugging Face.
2. Load and preprocess the dataset.
3. Tokenize the text data.
4. Initialize the text generation pipeline using the LLaMA model.
5. Generate answers to predefined questions based on the dataset.

## Questions Answered by the Model

The script asks the following questions:

1. What was the most famous song in India recently?
2. What are some of the best acoustic songs?
3. Are there any explicit songs? If yes, please list them.
4. Name some famous artists who featured regularly on this list?

## Troubleshooting

- If you encounter errors related to the tokenizer or padding, ensure that the model and tokenizer versions are compatible.
- Check that the dataset has the required columns and is properly formatted.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for providing the pre-trained LLaMA model.
- [PyTorch](https://pytorch.org/) for the deep learning framework.
