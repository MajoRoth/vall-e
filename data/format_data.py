import os.path
import sys

import pandas as pd


NIKUD_RANGE = (1425, 1479)  # Nikud in Unicode

def is_nikud(char):
    if ord(char) >= 1425 and ord(char) <= 1479:
        return True
    return False

def remove_nikud(text: str):
    new_text = ""
    for char in text:
        if not is_nikud(char):
            new_text += char
    return new_text


def format_data(data_path, output_path):
    data_frame = pd.read_csv(os.path.join(data_path, "metadata.csv"), encoding="utf-8", sep='|')

    for index, row in data_frame.iterrows():
        with open(os.path.join(output_path, f"{row['file_id']}.normalized.txt"), 'w') as txt_file:
            txt_file.write(
                remove_nikud(row['transcript'])
            )
            txt_file.close()


if __name__ == "__main__":
    if len(sys.argv) == 3:
        data_path = sys.argv[1]
        output_path = sys.argv[2]
    else:
        data_path = "/Users/amitroth/Data/hayot_kis/saspeech_gold_standard"
        output_path = "./saspeech"