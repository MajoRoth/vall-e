from abc import ABC, abstractmethod
import string

import torch
from transformers import BertModel, BertTokenizerFast


class Tokenizer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def tokenize(self, input: str):
        pass

    def __call__(self, text):
        return self.tokenize(text)



class TokenizeByLetters(Tokenizer):

    def tokenize(self, input: str):
        tokens = [x for x in input]
        ignored = {" ", *string.punctuation}
        return ["_" if p in ignored else p for p in tokens]



class TokenizeByWords(Tokenizer):

    def tokenize(self, input: str):
        tokens = HebrewTextUtils.clean_punctuation(input).split()
        return tokens



class TokenizeByLettersAlephBert(Tokenizer):

    def __init__(self):
        self.alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
        self.alephbert = BertModel.from_pretrained('onlplab/alephbert-base')


    def tokenize(self, input: str):
        """
            tokenize the sentence with AlephBert vectors.
            the vectors of the model are placed after each word letter tokens.
            output: <vec cls> <l1> <l2> <l3> <l4> <vec w1> <l5> <l6> <l7> <vec 2w>
        """
        clean_input = HebrewTextUtils.clean_punctuation(input)
        words = clean_input.split()
        self.alephbert.eval()
        input = self.alephbert_tokenizer(clean_input, return_tensors="pt")
        outputs = self.alephbert(**input)
        last_hidden_states = outputs.last_hidden_state[0]

        tokens = [last_hidden_states[0].tolist()]

        for i in range(len(words)):
            tokens += [x for x in words[i]]
            tokens.append(
                last_hidden_states[i+1]
            )

        return tokens


class HebrewTextUtils:
    NIKUD_RANGE = (1425, 1479)  # Nikud in Unicode

    @staticmethod
    def remove_nikud(text: str):
        if not isinstance(text, str):
            print(f"ERROR: {text} is not str - {type(text)}. returned empty string")
            return ""
        new_text = ""
        for char in text:
            if not HebrewTextUtils.is_nikud(char):
                new_text += char
        return new_text

    @staticmethod
    def is_nikud(char):
        if ord(char) >= HebrewTextUtils.NIKUD_RANGE[0] and ord(char) <= HebrewTextUtils.NIKUD_RANGE[1]:
            return True
        return False

    @staticmethod
    def clean_punctuation(word: str):
        return word.translate(str.maketrans('', '', string.punctuation))



if __name__ == "__main__":
    tokenizer = TokenizeByLettersAlephBert()
    tokenizer.tokenize("היי מה קורה")
    # tokenizer.tokenize("היי מה שלומך")
    # tokenizer.tokenize("היי")
    # tokenizer.tokenize("ביי")
