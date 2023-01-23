import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    bert_path = "/Users/admin/Desktop/bert-base-chinese-model"
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    text = '[CLS]美国的首都是[MASK][MASK]'
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Create the segments tensors.
    segments_ids = [0] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Load pre-trained model (weights)
    model = BertForMaskedLM.from_pretrained(bert_path)
    model.eval()

    masked_indexs = []

    for i in range(len(tokenized_text)):
        if tokenized_text[i] == "[MASK]":
            masked_indexs.append(i)

    for masked_index in masked_indexs:
        # Predict all tokens
        with torch.no_grad():
            predictions = model(tokens_tensor, segments_tensors)
        predicted_index = torch.argmax(predictions[0][0][masked_index]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        print(predicted_token)