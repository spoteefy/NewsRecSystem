import torch.nn as nn

from transformers.models.bert.modeling_bert import BertModel, BertConfig


class NewsEncoder(nn.Module):
    def __init__(self, config):
        super(NewsEncoder, self).__init__()
        self.config = config

        bert_config = BertConfig.from_pretrained(config.bert_model)
        bert_config.num_hidden_layers = 8
        self.bert_model = BertModel.from_pretrained(config.bert_model, config=bert_config)

        self.category_dropout = nn.Dropout(config.classifier_dropout)
        self.category_classifier = nn.Linear(bert_config.hidden_size, config.num_categories)
        self.category_loss_fn = nn.CrossEntropyLoss()
        nn.init.xavier_uniform_(self.category_classifier.weight, gain=1)

        self.ner_dropout = nn.Dropout(config.classifier_dropout)
        self.ner_classifier = nn.Linear(bert_config.hidden_size, config.num_entities)
        self.ner_loss_fn = nn.CrossEntropyLoss()
        nn.init.xavier_uniform_(self.ner_classifier.weight, gain=1)

    def forward(self, title_ids, attention_mask, token_type_ids, category_labels, ner_labels, **kwargs):
        outputs = self.bert_model(title_ids, attention_mask, token_type_ids)

        cls_output = outputs[0][:, 0, :]
        pooled_output = self.category_dropout(outputs[1])
        sequence_output = self.ner_dropout(outputs[0])

        category_logits = self.category_classifier(pooled_output)
        ner_logits = self.ner_classifier(sequence_output)

        category_loss = self.category_loss_fn(category_logits.view(-1, self.config.num_categories), category_labels.view(-1))
        ner_loss = self.ner_loss_fn(ner_logits.view(-1, self.config.num_entities), ner_labels.view(-1))

        return cls_output, category_loss, ner_loss
