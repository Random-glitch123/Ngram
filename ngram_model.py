import random
import re
import os

'''
For python code set KEEP_FORMATTING = True and SPLIT_CHAR = "\n"
For javascript code set KEEP_FORMATTING = True and SPLIT_CHAR = ";"

'''

TRAINING_DATA_PATH = 'english_dataset/'
NGRAMSIZE = 7
GENERATED_TEXT_LENGTH = 30
SPLIT_CHAR = " "
KEEP_FORMATTING = False

class NGramModel:
    """
    A simple N-gram language model for text generation.
    """
    def __init__(self, n):

        if n < 2:
            raise ValueError("n must be at least 2 for this model.")
        self.n = n
        self.model = {}  # The dictionary to store n-gram relationships

    def _preprocess(self, text):
        if(KEEP_FORMATTING):
            pass
        else:
            text = text.lower()
            # Keep only letters and whitespace
            text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split(SPLIT_CHAR)
        return tokens

    def fit(self, text):
        """
        Trains (fits) the n-gram model on the provided text.
        """
        tokens = self._preprocess(text)

        if len(tokens) < self.n:
            print(f"Warning: Text is too short for n={self.n}. Model will not be trained.")
            return

        # Iterate through the tokens to build the model
        for i in range(len(tokens) - self.n + 1):
            prefix = tuple(tokens[i : i + self.n - 1])
            next_word = tokens[i + self.n - 1]

            if prefix not in self.model:
                self.model[prefix] = []
            self.model[prefix].append(next_word)

        print(f"Model trained with {len(self.model)} unique prefixes.")

    def generate(self, max_words=50):

# Start with a random prefix from the model's keys
        start_prefix = random.choice(list(self.model.keys()))
        result = list(start_prefix)

        for _ in range(max_words - (self.n - 1)):
            current_prefix = tuple(result[-(self.n - 1):])

            if current_prefix not in self.model:
                # If we hit a dead end, stop generating
                break

            # Get possible next words and choose one randomly
            possible_next_words = self.model[current_prefix]
            next_word = random.choice(possible_next_words)
            result.append(next_word)

        return SPLIT_CHAR.join(result)

# --- Example Usage ---
if __name__ == '__main__':
    # --- Load Training Data from File or Folder ---
    training_text = ""
    path = TRAINING_DATA_PATH

    if not os.path.exists(path):
        print(f"Error: The path '{path}' does not exist.")
        print("Please update the TRAINING_DATA_PATH variable at the top of the script.")
    elif os.path.isdir(path):
        print(f"--- Loading training data from directory: {path} ---")
        loaded_files = 0
        # Recursively walk through the directory to find all .txt files
        for root, _, files in os.walk(path):
            for filename in sorted(files):
                if filename.endswith(".txt"):
                    file_path = os.path.join(root, filename)
                    try:
                        # Use errors='ignore' for more robust file reading
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            training_text += f.read() + "\n"
                            loaded_files += 1
                            print(f"  - Loaded {os.path.relpath(file_path, path)}")
                    except Exception as e:
                        print(f"  - Error loading {filename}: {e}")
        if loaded_files == 0:
            print("Warning: No .txt files were found in the specified directory or its subdirectories.")
        else:
            print(f"--- Finished loading {loaded_files} file(s). ---")
    elif os.path.isfile(path):
        print(f"--- Loading training data from file: {path} ---")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                training_text = f.read()
            print("--- Finished loading file. ---")
        except Exception as e:
            print(f"Error: Could not read file: {e}")
            training_text = ""
    else:
        print(f"Error: The path '{path}' is not a valid file or directory.")

    if training_text:
        # --- Trigram Model (n=3) ---
        print("\n--- Training a Ngram Model (n=" + str(NGRAMSIZE) + ") ---")
        trigram_model = NGramModel(n=NGRAMSIZE)
        trigram_model.fit(training_text)
        print("\n--- Generating text with Ngram Model ---")
        generated_text = trigram_model.generate(max_words=GENERATED_TEXT_LENGTH)
        print(generated_text)
        print("-" * 50)