import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

from utils import prefilter_items


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighting=True):
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        self.data = data
        self.user_item_matrix = self.prepare_matrix(self.data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(
            self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data):

        data = prefilter_items(data)

        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',  # Можно пробоват ьдругие варианты
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)  # необходимый тип матрицы для implicit

        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=-1)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=-1):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)

        model.fit(csr_matrix(user_item_matrix).tocsr())

        return model

    def get_top_items(self, user, N=5):
        """Находим топ-N покупок юзера"""

        top_items = self.data[self.data['user_id'] == user].groupby(['user_id', 'item_id'])[
            'quantity'].count().reset_index().sort_values('quantity', ascending=False)
        top_items.drop(top_items[top_items['item_id'] == 999999].index, axis=0, inplace=True)

        return top_items[:N]

    def get_similar_items(self, item, N=5):
        recs = self.model.similar_items(self.itemid_to_id[item], N=N)[0][1]

        return self.id_to_itemid[recs]

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров.
        Находим топ покупок юзера. Для каждого товара нужно найти похожий. Находим два похожих.
        Первый в списке будет сам купленный товар, поэтому берем второй"""

        top_items = self.get_top_items(user=user)
        top_items['simular_items'] = top_items['item_id'].apply(lambda x: self.get_similar_items(item=x, N=2))
        res = top_items['simular_items'].to_list()

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        similar_users = self.model.similar_users(self.userid_to_id[user], N)[0]
        res = [self.get_top_items(self.id_to_userid[user], 1)['item_id'].values[0] for user in similar_users]

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res