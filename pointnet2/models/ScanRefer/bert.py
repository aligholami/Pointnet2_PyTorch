import pytorch_lightning as pl
from transformers import BertModel

class BERTScanReferFineTuner(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
        
    def forward(self, input_ids, attention_mask, token_type_ids):

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)