import torch
from transformers import PreTrainedModel, BertConfig, BertModel


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
