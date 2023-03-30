import pandas as pd


def prefilter_items(data, n_popular=5000):

    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top_k = popularity.sort_values('n_sold', ascending=False).head(n_popular).item_id.tolist()

    data.loc[~data['item_id'].isin(top_k), 'item_id'] = 999999

    return data