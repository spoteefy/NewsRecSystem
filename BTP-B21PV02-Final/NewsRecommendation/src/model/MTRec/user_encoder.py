import torch.nn as nn

from transformers import BertConfig

from model.general.attention.additive import AdditiveAttention


class UserEncoder(nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.config = config
        bert_config = BertConfig.from_pretrained(config.bert_model)
        self.attention = AdditiveAttention(config.query_vector_dim, bert_config.hidden_size)

    def forward(self, news_vectors):
        """
            Args:
                news_vectors: batch_size, num_clicked_news_a_user, hidden_size
            Returns:
                (shape) batch_size, hidden_size
        """
        return self.attention(news_vectors)
