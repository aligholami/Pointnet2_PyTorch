import pytorch_lightning as pl
from transformers import BertModel, BertTokenizer, PreTrainedTokenizer, AutoTokenizer
import torch

class BERTEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # accepts both strings and list of strings as input
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.fine_tune_bert = False

        if self.fine_tune_bert:
            self.bert_model.train()
            self.session = torch.enable_grad
        else:
            self.bert_model.eval()
            self.session = torch.no_grad

    def __test_single__(self):
        test_sequence = 'This is a BERT model. It is a nice model.'
        marked_text = "[CLS] " + test_sequence + " [SEP]"
        tokenized_sequence = self.tokenizer.tokenize(marked_text)
        token_indices = self.tokenizer.convert_tokens_to_ids(tokenized_sequence)
        print("[Single Test Results]")
        for i1, i2 in zip(tokenized_sequence, token_indices):
            print("{}: {}".format(i1, i2))

        segments_ids = [1] * len(tokenized_sequence)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([token_indices])
        segments_tensors = torch.tensor([segments_ids])

        self.bert_model.eval()
        
        with torch.no_grad():
            bert_outputs = self.bert_model(tokens_tensor, segments_tensors)
            hidden_states = bert_outputs[2]
            layer_i = 0
            batch_i = 0
            token_i = 0
            print("Number of BERT layers: ", len(hidden_states))
            print("Number of batches in layer {}: {}".format(layer_i, len(hidden_states[layer_i])))
            print("Number of tokens in layer {} and batch {}: {}".format(layer_i, batch_i, len(hidden_states[layer_i][batch_i])))
            print("Number of dimensions in layer {}, batch {} and token {}: {}".format(layer_i, batch_i, token_i, len(hidden_states[layer_i][batch_i][token_i])))
        
        encoded_text_2 = self.tokenizer.encode_plus(test_sequence, return_tensors="pt")
        with torch.no_grad():
            bert_outputs = self.bert_model(**encoded_text_2)
            hidden_states = bert_outputs[2]
            layer_i = 0
            batch_i = 0
            token_i = 0
            print("Number of BERT layers: ", len(hidden_states))
            print("Number of batches in layer {}: {}".format(layer_i, len(hidden_states[layer_i])))
            print("Number of tokens in layer {} and batch {}: {}".format(layer_i, batch_i, len(hidden_states[layer_i][batch_i])))
            print("Number of dimensions in layer {}, batch {} and token {}: {}".format(layer_i, batch_i, token_i, len(hidden_states[layer_i][batch_i][token_i])))

    def __test_batch__(self):
        batch_sequence = ['hello world.', 'lorem ipsum made in nowhere.', 'Im listening to finneas at the momment.']
        tokens = self.tokenizer.batch_encode_plus(batch_sequence, padding=True, return_tensors="pt")
        tokens_no_batch_encoded = self.tokenizer(batch_sequence, padding=True, return_tensors="pt")
        # assert tokens == tokens_no_batch_encoded
        print("\n[Batch Test Results]")
        print(tokens)

        with torch.no_grad():
            bert_outputs = self.bert_model(**tokens)

        print("\n[Generated Batch BERT Outputs.]")

    def forward(self, text_batch: list) -> torch.Tensor:
        """
            Takes a batch of text.
            Returns a tensor of size (batch_size, sequence_length, 768).
        """
        tokens = self.tokenizer.batch_encode_plus(text_batch, padding=True, return_tensors="pt")

        with self.session() as ss:
            bert_outputs = self.bert_model(**tokens)
            print("Done 1")
            hidden_states = bert_outputs[2]
        
        print("Resulting hidden states: ", hidden_states)
        exit(0)

if __name__ == '__main__':
    bert_encoder = BERTEncoder()
    bert_encoder.__test_single__()
    bert_encoder.__test_batch__()