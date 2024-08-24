import pandas as pd
from sklearn.utils import shuffle


def convert_example_to_feature(review, tokenizer, max_length):
    # 返回字典
    # {'input_ids': [101, 3284, 3449,......, 0, 0],
    #  'token_type_ids': [0, 0,...... 0, 0, 0],
    #  'attention_mask': [1, 1, 1,...... 0, 0, 0]}
    return tokenizer.encode_plus(review,
                                 add_special_tokens=False,  # add [CLS], [SEP]
                                 max_length=max_length,  # max length of the text that can go to BERT
                                 pad_to_max_length=True,  # add [PAD] tokens
                                 return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                 truncation=True,
                                 # return_offsets_mapping=True
                                 )


def convert_example_to_feature_aspect(category, tokenizer, max_length):
    # 返回字典
    # {'input_ids': [101, 3284, 3449,......, 0, 0],
    #  'token_type_ids': [0, 0,...... 0, 0, 0],
    #  'attention_mask': [1, 1, 1,...... 0, 0, 0]}
    return tokenizer.encode_plus(category,
                                 add_special_tokens=True,  # add [CLS], [SEP]
                                 max_length=max_length,  # max length of the text that can go to BERT
                                 pad_to_max_length=True,  # add [PAD] tokens
                                 return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                 truncation=True,
                                 # return_offsets_mapping=True
                                 )


def map_emb_to_dict(input_ids, attention_masks, aspect_input_ids, aspect_attention_mask, graphs, relation_value, label):
    return {
        "input_ids": input_ids,
        "attention_masks": attention_masks,
        "aspect_input_ids": aspect_input_ids,
        "aspect_attention_masks": aspect_attention_mask,
        "graphs": graphs,
        "relation_value": relation_value
        # "target_word": target_word
    }, label


def map_emb_to_dict_o(input_ids, attention_masks, graphs, relation_value, label):
    return {
        "input_ids": input_ids,
        "attention_masks": attention_masks,
        "graphs": graphs,
        "relation_value": relation_value
    }, label


def read_dataset(path):
    dataset = pd.read_csv(path, keep_default_na=False)
    dataset = shuffle(dataset)
    return dataset