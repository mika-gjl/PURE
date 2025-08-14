import numpy as np
import json
import logging

logger = logging.getLogger('root')

def batchify(samples, batch_size):
    """
    Batchfy samples with a batch size
    """
    num_samples = len(samples)

    list_samples_batches = []

    # if a sentence is too long, make itself a batch to avoid GPU OOM
    to_single_batch = []
    for i in range(0, len(samples)):
        if len(samples[i]['tokens']) > 350:
            to_single_batch.append(i)

    for i in to_single_batch:
        logger.info('Single batch sample: %s-%d', samples[i]['doc_key'], samples[i]['sentence_ix'])
        list_samples_batches.append([samples[i]])
    samples = [sample for i, sample in enumerate(samples) if i not in to_single_batch]

    for i in range(0, len(samples), batch_size):
        list_samples_batches.append(samples[i:i+batch_size])

    assert(sum([len(batch) for batch in list_samples_batches]) == num_samples)

    return list_samples_batches

def overlap(s1, s2):
    if s2.start_sent >= s1.start_sent and s2.start_sent <= s1.end_sent:
        return True
    if s2.end_sent >= s1.start_sent and s2.end_sent <= s1.end_sent:
        return True
    return False

def convert_dataset_to_samples(dataset, max_span_length, ner_label2id=None,
                               context_window=0, split=0, pretokenized=False):
    """
    Extract sentence samples and gold entities from a dataset.

    Parameters
    ----------
    dataset : Dataset
        Your dataset object.
    max_span_length : int
        Max length (in tokens) of candidate spans to enumerate.
    ner_label2id : dict
        Mapping from NER label string to int id.
    context_window : int
        If >0, expand tokens with left/right context up to this window size (token-level).
    split : int
        0=all, 1=first 90%, 2=last 10%.
    pretokenized : bool
        If True, 'sent.text' is assumed to be ALREADY tokenized with the SAME tokenizer
        you will use for training/inference (i.e., subword-level). Spans in gold are
        assumed to be in the SAME index space.
    """
    samples = []
    num_ner = 0
    max_len = 0
    max_ner = 0
    num_overlap = 0

    if split == 0:
        data_range = (0, len(dataset))
    elif split == 1:
        data_range = (0, int(len(dataset)*0.9))
    elif split == 2:
        data_range = (int(len(dataset)*0.9), len(dataset))
    else:
        data_range = (0, len(dataset))

    for c, doc in enumerate(dataset):
        if c < data_range[0] or c >= data_range[1]:
            continue

        for i_sent, sent in enumerate(doc):
            # --- 基础检查：确保是“列表形式的 tokens” ---
            tokens = sent.text
            if isinstance(tokens, str):
                # 兜底：如果是字符串，按空格切。强烈建议你在数据准备阶段就生成 list。
                tokens = tokens.split()

            num_ner += len(sent.ner)
            sample = {
                'doc_key': doc._doc_key,
                'sentence_ix': sent.sentence_ix,
                'pretokenized': bool(pretokenized),  # 关键信号，供下游模型决定是否跳过二次分词
            }

            # 过长句提示（仅日志）
            if context_window != 0 and len(tokens) > context_window:
                logger.info('Long sentence: {} {}'.format(
                    {'doc_key': doc._doc_key, 'sentence_ix': sent.sentence_ix}, len(tokens))
                )

            sample['tokens'] = list(tokens)  # 拷贝一份，避免就地修改
            sample['sent_length'] = len(sample['tokens'])
            sent_start = 0
            sent_end = len(sample['tokens'])

            max_len = max(max_len, len(sample['tokens']))
            max_ner = max(max_ner, len(sent.ner))

            # --- 上下文窗口（按 token 数扩展，不涉及重新分词） ---
            if context_window > 0 and len(sample['tokens']) < context_window:
                need = context_window - len(sample['tokens'])
                add_left = need // 2
                add_right = need - add_left

                # 往左补
                j = i_sent - 1
                while j >= 0 and add_left > 0:
                    left_tokens = doc[j].text
                    if isinstance(left_tokens, str):
                        left_tokens = left_tokens.split()
                    if len(left_tokens) <= add_left:
                        context_to_add = left_tokens
                    else:
                        context_to_add = left_tokens[-add_left:]
                    sample['tokens'] = list(context_to_add) + sample['tokens']
                    add_left -= len(context_to_add)
                    sent_start += len(context_to_add)
                    sent_end += len(context_to_add)
                    j -= 1

                # 往右补
                j = i_sent + 1
                while j < len(doc) and add_right > 0:
                    right_tokens = doc[j].text
                    if isinstance(right_tokens, str):
                        right_tokens = right_tokens.split()
                    if len(right_tokens) <= add_right:
                        context_to_add = right_tokens
                    else:
                        context_to_add = right_tokens[:add_right]
                    sample['tokens'] = sample['tokens'] + list(context_to_add)
                    add_right -= len(context_to_add)
                    j += 1

            sample['sent_start'] = sent_start
            sample['sent_end'] = sent_end
            sample['sent_start_in_doc'] = sent.sentence_start

            # --- 汇总 gold NER：键为 (i, j) 的闭区间 span（基于当前 tokens 的索引空间） ---
            sent_ner = {}
            for ner in sent.ner:
                # ner.span.span_sent 应是 (start, end) 闭区间，且已在与 tokens 同一索引空间
                span_ij = ner.span.span_sent
                sent_ner[span_ij] = ner.label

            # --- 枚举候选 span + 贴标签 ---
            span2id = {}
            sample['spans'] = []
            sample['spans_label'] = []

            T = len(sample['tokens'])
            for i in range(T):
                # 限制枚举长度，避免极长句爆炸
                j_max = min(T - 1, i + max_span_length - 1)
                for j in range(i, j_max + 1):
                    sample['spans'].append((i + sent_start, j + sent_start, j - i + 1))
                    span2id[(i, j)] = len(sample['spans']) - 1
                    if (i, j) in sent_ner:
                        if ner_label2id is None:
                            raise ValueError("ner_label2id is required to map labels to ids.")
                        sample['spans_label'].append(ner_label2id[sent_ner[(i, j)]])
                    else:
                        sample['spans_label'].append(0)

            # 统计被 max_span_length 截断而“覆盖不到”的 gold（仅统计用途）
            for (i_g, j_g), _lab in sent_ner.items():
                if (i_g, j_g) not in span2id:
                    num_overlap += 1

            samples.append(sample)

    avg_length = sum([len(sample['tokens']) for sample in samples]) / max(1, len(samples))
    max_length = max([len(sample['tokens']) for sample in samples]) if samples else 0
    logger.info('# Gold spans outside enumerated window (>{}): {}'.format(max_span_length, num_overlap))
    logger.info('Extracted %d samples from %d documents, with %d NER labels, %.3f avg input length, %d max length',
                len(samples), data_range[1]-data_range[0], num_ner, avg_length, max_length)
    logger.info('Max sentence length: %d, max NER per sentence: %d', max_len, max_ner)
    return samples, num_ner


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def get_train_fold(data, fold):
    print('Getting train fold %d...'%fold)
    l = int(len(data) * 0.1 * fold)
    r = int(len(data) * 0.1 * (fold+1))
    new_js = []
    new_docs = []
    for i in range(len(data)):
        if i < l or i >= r:
            new_js.append(data.js[i])
            new_docs.append(data.documents[i])
    print('# documents: %d --> %d'%(len(data), len(new_docs)))
    data.js = new_js
    data.documents = new_docs
    return data

def get_test_fold(data, fold):
    print('Getting test fold %d...'%fold)
    l = int(len(data) * 0.1 * fold)
    r = int(len(data) * 0.1 * (fold+1))
    new_js = []
    new_docs = []
    for i in range(len(data)):
        if i >= l and i < r:
            new_js.append(data.js[i])
            new_docs.append(data.documents[i])
    print('# documents: %d --> %d'%(len(data), len(new_docs)))
    data.js = new_js
    data.documents = new_docs
    return data

# === Replacement for allennlp.nn.util.batched_index_select ===
def batched_index_select(sequence_output, spans):
    batch_size, seq_len, hidden_size = sequence_output.size()
    num_spans = spans.size(1)
    span_embeddings = []
    for b in range(batch_size):
        span_embedding = sequence_output[b].index_select(0, spans[b].view(-1))
        span_embeddings.append(span_embedding)
    span_embeddings = torch.stack(span_embeddings, dim=0)
    span_embeddings = span_embeddings.view(batch_size, num_spans, -1)
    return span_embeddings

# === Replacement for allennlp.modules.FeedForward ===
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dims, activations, dropout):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dims
            out_dim = hidden_dims
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU() if activations == nn.ReLU or activations == F.relu else activations)
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# === Replacement for allennlp.nn.Activation ===
Activation = {
    'relu': nn.ReLU(),
    'gelu': nn.GELU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
}
