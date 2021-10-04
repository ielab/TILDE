import torch
from transformers import PreTrainedModel, BertConfig, BertModel, BertLMHeadModel, BertTokenizer
import pytorch_lightning as pl
from tools import get_stop_ids


class TILDE(pl.LightningModule):
    def __init__(self, model_type, from_pretrained=None, gradient_checkpointing=False):
        super().__init__()
        if from_pretrained is not None:
            self.bert = BertLMHeadModel.from_pretrained(from_pretrained, cache_dir="./cache")
        else:
            self.config = BertConfig.from_pretrained(model_type, cache_dir="./cache")
            self.config.gradient_checkpointing = gradient_checkpointing  # for trade off training speed for larger batch size
            print(self.config.gradient_checkpointing)
            # self.config.is_decoder = True
            self.bert = BertLMHeadModel.from_pretrained(model_type, config=self.config, cache_dir="./cache")

        self.tokenizer = BertTokenizer.from_pretrained(model_type, cache_dir="./cache")
        self.num_valid_tok = self.tokenizer.vocab_size - len(get_stop_ids(self.tokenizer))

    def forward(self, x):
        input_ids, token_type_ids, attention_mask = x
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            return_dict=True).logits[:, 0]
        return outputs

    # Bi-direction loss (BDQLM)
    def training_step(self, batch, batch_idx):
        # todo test all_gather: self.all_gather()
        passage_input_ids, passage_token_type_ids, passage_attention_mask, yqs, neg_yqs, \
        query_input_ids, query_token_type_ids, query_ttention_mask, yds, neg_yds = batch

        passage_outputs = self.bert(input_ids=passage_input_ids,
                                    token_type_ids=passage_token_type_ids,
                                    attention_mask=passage_attention_mask,
                                    return_dict=True).logits[:, 0]

        query_outputs = self.bert(input_ids=query_input_ids,
                                  token_type_ids=query_token_type_ids,
                                  attention_mask=query_ttention_mask,
                                  return_dict=True).logits[:, 0]

        batch_size = len(yqs)
        batch_passage_prob = torch.sigmoid(passage_outputs)
        batch_query_prob = torch.sigmoid(query_outputs)

        passage_pos_loss = 0
        query_pos_loss = 0
        for i in range(batch_size):

            # BCEWithLogitsLoss
            passage_pos_ids_plus = torch.where(yqs[i] == 1)[0]
            passage_pos_ids_minus = torch.where(yqs[i] == 0)[0]

            passage_pos_probs = batch_passage_prob[i][passage_pos_ids_plus]
            passage_neg_probs = batch_passage_prob[i][passage_pos_ids_minus]
            passage_pos_loss -= torch.sum(torch.log(passage_pos_probs)) + torch.sum(torch.log(1-passage_neg_probs))

            query_pos_ids_plus = torch.where(yds[i] == 1)[0]
            query_pos_ids_minus = torch.where(yds[i] == 0)[0]

            query_pos_probs = batch_query_prob[i][query_pos_ids_plus]
            query_neg_probs = batch_query_prob[i][query_pos_ids_minus]
            query_pos_loss -= torch.sum(torch.log(query_pos_probs)) + torch.sum(torch.log(1-query_neg_probs))

        loss = (passage_pos_loss + query_pos_loss) / (self.num_valid_tok * 2)

        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer

    def save(self, path):
        self.bert.save_pretrained(path)


class TILDEv2(PreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "tildev2"

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.tok_proj = torch.nn.Linear(config.hidden_size, 1)
        self.init_weights()

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.bert.init_weights()
        self.tok_proj.apply(self._init_weights)

    def encode(self, **features):
        assert all([x in features for x in ['input_ids', 'attention_mask', 'token_type_ids']])
        model_out = self.bert(**features, return_dict=True)
        reps = self.tok_proj(model_out.last_hidden_state)
        tok_weights = torch.relu(reps)
        return tok_weights