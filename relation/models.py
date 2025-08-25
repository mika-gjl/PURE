import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.models.camembert.modeling_camembert import CamembertModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel as _PreTrained



from allennlp.modules import FeedForward
from allennlp.nn.util import batched_index_select
import torch.nn.functional as F

def _pick_sequence_output(outputs):
    """
    兼容不同 transformers 版本的返回结构：
    - tuple/list: 取 index 0
    - dataclass: 有 last_hidden_state 就取它
    """
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    return outputs

BertLayerNorm = torch.nn.LayerNorm
class BertForRelation(_PreTrained):
    def __init__(self, config, num_rel_labels):
        super(BertForRelation, self).__init__(config)
        self.num_labels = num_rel_labels
        # 用 CamemBERT 做骨干，属性仍叫 self.bert（避免其它文件引用改动）
        self.bert = CamembertModel(config)
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        # 兼容不同 transformers 版本的权重初始化
        if hasattr(self, "post_init"):
            self.post_init()
        else:
            self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, sub_idx=None, obj_idx=None, input_position=None):
        # CamemBERT/Roberta 忽略 token_type_ids（传了也不会报错）
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
            position_ids=input_position
        )
        sequence_output = _pick_sequence_output(outputs)
        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_idx)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_idx)])
        rep = torch.cat((sub_output, obj_output), dim=1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits



class BertForRelationApprox(_PreTrained):
    def __init__(self, config, num_rel_labels):
        super(BertForRelationApprox, self).__init__(config)
        self.num_labels = num_rel_labels
        self.bert = CamembertModel(config)
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        if hasattr(self, "post_init"):
            self.post_init()
        else:
            self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, sub_obj_ids=None, sub_obj_masks=None, input_position=None):
        """
        attention_mask: [batch_size, from_seq_length, to_seq_length]（如果你真用到 3D mask）
        """
        batch_size = input_ids.size(0)
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
            position_ids=input_position
        )
        sequence_output = _pick_sequence_output(outputs)

        sub_ids = sub_obj_ids[:, :, 0].view(batch_size, -1)
        sub_embeddings = batched_index_select(sequence_output, sub_ids)
        obj_ids = sub_obj_ids[:, :, 1].view(batch_size, -1)
        obj_embeddings = batched_index_select(sequence_output, obj_ids)
        rep = torch.cat((sub_embeddings, obj_embeddings), dim=-1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            active_loss = (sub_obj_masks.view(-1) == 1)
            active_logits = logits.view(-1, logits.shape[-1])
            active_labels = torch.where(
                active_loss, labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:
            return logits

