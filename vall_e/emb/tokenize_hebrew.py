

class TokenizeHebrew():
    def __init__(self):
        self.token_size = 1

    def tokenize_string(self, text: str):
        """
            TODO: Add support for many sorts of tokenization
        """
        return [x for x in text]

    def __call__(self, text):
        return self.tokenize_string(text)
