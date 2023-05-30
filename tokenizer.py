from abc import ABC, abstractmethod
import string

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
        clean_input = HebrewTextUtils.clean_punctuation(input)
        self.alephbert.eval()
        for i in range(len(clean_input.split())):
            input = self.alephbert_tokenizer(clean_input, return_tensors="pt", position_ids=i)
            outputs = self.alephbert(**input)
            last_hidden_states = outputs.last_hidden_state
            print(last_hidden_states)

        last_hidden_states = outputs.last_hidden_state
        print(last_hidden_states.size())
        print(last_hidden_states)
        letter_tokens = [x for x in clean_input]
        tokens = last_hidden_states.tolist() + letter_tokens
        print(tokens)
        return tokens


class HebrewTextUtils:
    NIKUD_RANGE = (1425, 1479)  # Nikud in Unicode

    @staticmethod
    def remove_nikud(text: str):
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
    # tokenizer.tokenize("היי")
    # tokenizer.tokenize("ביי")
