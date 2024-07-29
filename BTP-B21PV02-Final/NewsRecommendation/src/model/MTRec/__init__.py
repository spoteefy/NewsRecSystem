import torch
import torch.nn as nn

from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder
from .fast_user_encoder import FastUserEncoder
from model.general.click_predictor.dot_product import DotProductClickPredictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MTRec(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config)
        self.user_encoder = FastUserEncoder(config) if config.fastformer else UserEncoder(config)
        self.click_predictor = DotProductClickPredictor()

    def forward(self, candidate_news, clicked_news):
        """
            Args:
                candidate_news:
                    [
                        {
                            "title_ids": batch_size * num_words_title,
                            "attention_mask": batch_size * num_words_title,
                            "token_type_ids": batch_size * num_words_title,
                            "category_labels": batch_size,
                            "ner_labels": batch_size * num_words_title
                        } * (1 + K)
                    ]
                clicked_news:
                    [
                        {
                            "title_ids": batch_size * num_words_title,
                            "attention_mask": batch_size * num_words_title,
                            "token_type_ids": batch_size * num_words_title,
                            "category_labels": batch_size,
                            "ner_labels": batch_size * num_words_title
                        } * num_clicked_news_a_user
                    ]
            Returns:
                click_probability: batch_size
        """
        candidate_news_vectors = []
        clicked_news_vectors = []
        overall_category_loss = 0.0
        overall_ner_loss = 0.0

        for candidate in candidate_news:
            candidate = {k: v.to(device) for k, v in candidate.items()}
            cls_outputs, category_loss, ner_loss = self.news_encoder(**candidate)
            overall_category_loss += category_loss
            overall_ner_loss += ner_loss
            candidate_news_vectors.append(cls_outputs)

        # batch_size * (1 + K) * hidden_size
        candidate_news_vectors = torch.stack(candidate_news_vectors, dim=1)

        for click in clicked_news:
            click = {k: v.to(device) for k, v in click.items()}
            cls_outputs, category_loss, ner_loss = self.news_encoder(**click)
            overall_category_loss += category_loss
            overall_ner_loss += ner_loss
            clicked_news_vectors.append(cls_outputs)

        # batch_size * num_clicked_news_a_user * hidden_size
        clicked_news_vectors = torch.stack(clicked_news_vectors, dim=1)
        # batch_size * hidden_size
        user_vector = self.user_encoder(clicked_news_vectors)
        # batch_size * (1 + K)
        click_probability = self.click_predictor(candidate_news_vectors, user_vector)

        total_news = len(candidate_news) + len(clicked_news)

        return click_probability, (overall_category_loss + overall_ner_loss) / total_news

    def get_news_vector(self, news):
        """
        Args:
            news:
                {
                    "title_ids": batch_size * num_words_title,
                    "attention_mask": batch_size * num_words_title,
                    "token_type_ids": batch_size * num_words_title,
                    "category_labels": batch_size,
                    "ner_labels": batch_size * num_words_title
                }
        Returns:
            (shape) batch_size, hidden_size
        """
        news = {k: v.to(device) for k, v in news.items()}
        # batch_size, hidden_size
        return self.news_encoder(**news)[0]

    def get_user_vector(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        clicked_news_vector = clicked_news_vector.to(device)
        # batch_size, word_embedding_dim
        return self.user_encoder(clicked_news_vector)

    def get_prediction(self, news_vector, user_vector):
        """
        Args:
            news_vector: candidate_size, word_embedding_dim
            user_vector: word_embedding_dim
        Returns:
            click_probability: candidate_size
        """
        news_vector = news_vector.to(device)
        user_vector = user_vector.to(device)
        # candidate_size
        return self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)).squeeze(dim=0)
