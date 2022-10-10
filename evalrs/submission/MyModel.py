import cvxpy
import numpy as np
import pandas as pd
from typing import Optional, Tuple

from implicit.als import AlternatingLeastSquares
from jurity.mitigation import BinaryMitigation
from reclist.abstractions import RecModel
from scipy.sparse import csr_matrix

from tqdm import tqdm

POST_PROC_TYPE_USER_ACTIVITY_FAIRNESS = 'user_activity_fairness'
POST_PROC_TYPE_TRACK_ACTIVITY_FAIRNESS = 'track_activity_fairness'
ALL_USERS_CAT = 'all'


class MyModel(RecModel):
    
    def __init__(self, items: pd.DataFrame, users: pd.DataFrame, top_k: int = 100, factors: int = 50,
                 regularization: float = 0.1, alpha: float = 0.1,
                 post_proc_type: Optional[str] = POST_PROC_TYPE_USER_ACTIVITY_FAIRNESS,
                 post_proc_num_popular: int = 200,  # only valid for `user_activity_fairness`
                 post_proc_positive_quantile: float = 0.8,  # only valid for `user_activity_fairness`
                 is_member_max_positive_activity_bin: int = 2,  # only valid for `user_activity_fairness`
                 use_average: bool = False, **kwargs):
        super(MyModel, self).__init__()
        self.items = items
        self.users = users
        self.top_k = top_k
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.user_id_map = {}
        self.item_id_map = {}
        self.reverse_item_map = {}
        self.impl = {}
        self.cat_key = 'user_activity'
        self.n_sum = 300
        self.train_data = {}
        self.user_summary = None
        self.user_activity_summary = None
        self.track_popularity_summary = None
        self.artist_popularity_summary = None
        self.bins = {'user_activity': np.array([1, 100, 1000]), 'track_popularity': np.array([1, 10, 100, 1000]),
                     'artist_popularity': np.array([1, 100, 1000, 10000])}

        # Post-process params
        self.post_proc_type = post_proc_type
        self.post_proc_num_popular = post_proc_num_popular
        # The quantile value that a score has to beat to be called "positive"
        # Higher is more selective
        self.post_proc_positive_quantile = post_proc_positive_quantile
        self.is_member_max_positive_activity_bin = is_member_max_positive_activity_bin

        # Averaging params
        self.use_average = use_average

        # kwargs may contain additional arguments in case, for example, you
        # have data augmentation strategies
        print("Received additional arguments: {}".format(kwargs))
        return

    def get_user_attribute(self, train_df: pd.DataFrame, groupby_key: str, col_name: str) -> pd.DataFrame:
        summary_df = train_df.groupby(col_name, as_index=True, sort=False)[['user_track_count']].sum().rename(
            columns={'user_track_count': groupby_key})
        summary_df[f'{groupby_key}_bin_index'] = np.digitize(summary_df.values.reshape(-1), self.bins[groupby_key])
        summary_df[f'{groupby_key}_bins'] = self.bins[groupby_key][summary_df[f'{groupby_key}_bin_index'].values - 1]
        setattr(self, f'{groupby_key}_summary', summary_df)
        train_df = pd.merge(train_df, summary_df, left_on=col_name, right_index=True)
        return train_df

    def generate_scores_for_item(self, user_ids, item_id):
        _, likelihoods = self.impl[ALL_USERS_CAT].recommend(user_ids, self.train_data[ALL_USERS_CAT], N=1,
                                             filter_already_liked_items=False, items=[item_id])
        return np.squeeze(likelihoods)

    def generate_scores_for_user(self, user_ids, item_ids=None):
        if item_ids:
            likelihoods = np.dot(self.impl[ALL_USERS_CAT].user_factors[user_ids, :], self.impl[ALL_USERS_CAT].item_factors[item_ids].T)
        else:
            likelihoods = np.dot(self.impl[ALL_USERS_CAT].user_factors[user_ids, :], self.impl[ALL_USERS_CAT].item_factors.T)
        return np.squeeze(likelihoods)

    def generate_scores(self, user_ids):
        recommendation_ids, scores = self.impl[ALL_USERS_CAT].recommend(user_ids, self.train_data[ALL_USERS_CAT][user_ids, :], N=self.top_k,
                                                         filter_already_liked_items=False)
        return recommendation_ids, scores

    def _train_cat(self, train_df: pd.DataFrame, cat: str):
        user_id_map = train_df['user_id'].unique()
        self.user_id_map[cat] = pd.Series(np.arange(user_id_map.shape[0]), index=user_id_map)
        item_id_map = train_df['track_id'].unique()
        self.item_id_map[cat] = pd.Series(np.arange(item_id_map.shape[0]), index=item_id_map)
        self.reverse_item_map[cat] = pd.Series(self.item_id_map[cat].index, index=self.item_id_map[cat].values)

        train_data = csr_matrix((np.ones(train_df.shape[0]),
                                 (train_df['user_id'].map(self.user_id_map[cat]),
                                  train_df['track_id'].map(self.item_id_map[cat]))),
                                shape=(self.user_id_map[cat].shape[0], self.item_id_map[cat].shape[0]))
        self.train_data[cat] = train_data
        self.impl[cat] = AlternatingLeastSquares(factors=self.factors, regularization=self.regularization,
                                                 alpha=self.alpha)
        self.impl[cat].fit(self.train_data[cat])
        print(f"Training completed for Category {cat}!")
        return

    def _train_postprocess(self, train_data):
        if self.post_proc_type == POST_PROC_TYPE_USER_ACTIVITY_FAIRNESS:

            track_popularity = np.squeeze(np.asarray(train_data.sum(axis=0)))
            self.popular_items = np.argsort(track_popularity)[::-1][:self.post_proc_num_popular]

            self.mitigations = dict()
            for item_id in self.popular_items:
                labels = np.squeeze(train_data[:, item_id].toarray())
                # print('labels', labels)

                # likelihoods for the item
                likelihoods = self.generate_scores_for_item(self.user_id_map[ALL_USERS_CAT].values, item_id)
                likelihoods = np.squeeze(likelihoods)
                # take sigmoid
                likelihoods = 1. / (1. + np.exp(-likelihoods))
                # print('likelihoods', likelihoods)

                # use only the top 10% as a positive prediction
                binary_cutoff = np.quantile(likelihoods, self.post_proc_positive_quantile)
                predictions = (likelihoods > binary_cutoff).astype(int)
                # print('predictions', predictions)

                # user activity summary is sorted by internal user id
                is_member = (self.user_activity_summary['user_activity_bin_index'] <= self.is_member_max_positive_activity_bin).astype(int).values
                # print('is member', is_member)

                mitigation = BinaryMitigation.EqualizedOdds()
                mitigation.fit(labels, predictions, likelihoods, is_member)
                is_fitted = mitigation.p2p_prob_0 is not None
                if is_fitted:
                    self.mitigations[item_id] = mitigation

            print(f"Fitted {len(self.mitigations)} popular items")

        elif self.post_proc_type == POST_PROC_TYPE_TRACK_ACTIVITY_FAIRNESS:
            self.mitigations = dict()

            batch_size = 1000
            num_batches = self.user_id_map[ALL_USERS_CAT].shape[0] // batch_size

            is_member = (self.track_popularity_summary['track_popularity_bin_index'] <= self.is_member_max_positive_activity_bin).astype(int).values

            for i in tqdm(range(num_batches)):
                batch_ids = np.arange(i * batch_size, min((i + 1) * batch_size, self.user_id_map[ALL_USERS_CAT].shape[0]))
                labels_batch = train_data[batch_ids, :].toarray()

                likelihoods_batch = self.generate_scores_for_user(batch_ids)
                # take sigmoid
                likelihoods_batch = 1. / (1. + np.exp(-likelihoods_batch))

                # use only the top 10% as a positive prediction
                binary_cutoff = np.quantile(likelihoods_batch, self.post_proc_positive_quantile, axis=1)
                predictions_batch = (likelihoods_batch > binary_cutoff.reshape(-1, 1)).astype(int)

                for j, user_id in enumerate(batch_ids):
                    mitigation = BinaryMitigation.EqualizedOdds()
                    try:
                        mitigation.fit(labels_batch[j], predictions_batch[j], likelihoods_batch[j], is_member)
                        is_fitted = mitigation.p2p_prob_0 is not None
                        if is_fitted:
                            self.mitigations[user_id] = mitigation
                    except cvxpy.error.SolverError:
                        pass

    def train(self, train_df: pd.DataFrame):
        """
        Implement here your training logic. Since our example method is a simple random model,
        we actually don't use any training data to build the model, but you should ;-)

        At the end of training, make sure the class contains a trained model you can use in the predict method.
        """
        user_id_map = train_df['user_id'].unique()
        self.user_id_map[ALL_USERS_CAT] = pd.Series(np.arange(user_id_map.shape[0]), index=user_id_map)
        item_id_map = train_df['track_id'].unique()
        self.item_id_map[ALL_USERS_CAT] = pd.Series(np.arange(item_id_map.shape[0]), index=item_id_map)
        self.reverse_item_map[ALL_USERS_CAT] = pd.Series(self.item_id_map[ALL_USERS_CAT].index,
                                                         index=self.item_id_map[ALL_USERS_CAT].values)

        user_summary = self.users.loc[self.user_id_map[ALL_USERS_CAT].index, ['country', 'gender']].fillna('NaN')
        self.user_summary = user_summary
        train_df = pd.merge(train_df, user_summary, left_on='user_id', right_index=True)

        train_df = self.get_user_attribute(train_df, 'user_activity', 'user_id')
        train_df = self.get_user_attribute(train_df, 'track_popularity', 'track_id')
        train_df = self.get_user_attribute(train_df, 'artist_popularity', 'artist_id')

        self._train_cat(train_df, ALL_USERS_CAT)
        if self.use_average:
            for bin in self.bins[self.cat_key]:
                train_df_cat = train_df[train_df[f'{self.cat_key}_bins'] == bin]
                cat = self.cat_key + '_' + str(bin)
                self._train_cat(train_df_cat, cat)

        self._train_postprocess(self.train_data[ALL_USERS_CAT])
        return 

    def _predict_cat(self, user_ids: pd.DataFrame, cat: str) -> (pd.DataFrame, pd.DataFrame):
        test_ids = user_ids['user_id'].map(self.user_id_map[cat]).values
        recommendation_ids, scores = self.impl[cat].recommend(test_ids, self.train_data[cat][test_ids, :],
                                                              N=self.n_sum,
                                                              filter_already_liked_items=True)

        recs_df = pd.DataFrame(recommendation_ids).apply(lambda x: x.map(self.reverse_item_map[cat]), axis=1)
        recs_df.insert(0, 'user_id', user_ids['user_id'].values)
        recs_df.columns = ['user_id', *[str(i) for i in range(self.n_sum)]]

        scores_df = pd.DataFrame(scores).apply(lambda x: x.map(self.reverse_item_map[cat]), axis=1)
        scores_df.insert(0, 'user_id', user_ids['user_id'].values)
        scores_df.columns = ['user_id', *[str(i) for i in range(self.n_sum)]]
        return recs_df, scores_df

    def _take_average_score(self, recs_df_1: pd.DataFrame, scores_df_1: pd.DataFrame, recs_df_2: pd.DataFrame,
                            scores_df_2: pd.DataFrame) -> pd.DataFrame:

        recs_df_1 = pd.melt(recs_df_1, id_vars=['user_id'], value_name="rec")
        scores_df_1 = pd.melt(scores_df_1, id_vars=['user_id'], value_name='score')
        recs_df_1['score'] = scores_df_1['score']

        recs_df_2 = pd.melt(recs_df_2, id_vars=['user_id'], value_name="rec")
        scores_df_2 = pd.melt(scores_df_2, id_vars=['user_id'], value_name='score')
        recs_df_2['score'] = scores_df_2['score']

        sum_matrix = recs_df_1.set_index(['user_id', 'rec'])[['score']].add(
            recs_df_2.set_index(['user_id', 'rec'])[['score']], fill_value=0)

        top_scores = sum_matrix.groupby('user_id')['score'].nlargest(self.top_k)

        top_score_index = top_scores.index.droplevel(0)

        sum_matrix = sum_matrix.loc[top_score_index].reset_index()

        recommendation_ids = sum_matrix.groupby('user_id')['rec'].apply(lambda x: x.map(self.item_id_map[ALL_USERS_CAT]).to_list()).to_list()
        scores = sum_matrix.groupby('user_id')['score'].apply(lambda x: x.to_list()).to_list()

        return recommendation_ids, scores

    def _postprocess(self, user_ids, test_ids):
        if self.post_proc_type == POST_PROC_TYPE_USER_ACTIVITY_FAIRNESS:
            recommendation_ids, scores = self.generate_scores(test_ids)

            for item_id in self.popular_items:
                if item_id in self.mitigations:
                    item_scores = self.generate_scores_for_item(test_ids, item_id)
                    # print('item scores', item_scores.shape, item_scores)
                    # take sigmoid
                    likelihoods = 1. / (1. + np.exp(-item_scores))
                    # print('likelihoods', likelihoods.shape, likelihoods)

                    # use only the top 10% as a positive prediction
                    binary_cutoff = np.quantile(likelihoods, self.post_proc_positive_quantile)
                    predictions = likelihoods > binary_cutoff
                    # print('predictions', predictions.shape, predictions)

                    # calculate is_member
                    is_member = (self.user_activity_summary.loc[user_ids['user_id']]['user_activity_bin_index'] <= self.is_member_max_positive_activity_bin).astype(int).values
                    # print('is_member', is_member.shape, is_member)

                    # Run bias mitigation. Fair likelihoods contain flips
                    fair_predictions, fair_likelihoods = self.mitigations[item_id].transform(predictions, likelihoods, is_member)
                    # print('fair_predictions', fair_predictions.shape, fair_predictions)
                    # print('fair_likelihoods', fair_likelihoods.shape, fair_likelihoods)
                    # overlap = likelihoods == fair_likelihoods
                    # print('overlap', overlap.shape, overlap)
                    # print('num overlap', overlap.sum())

                    # To go from likelihoods to score, we can use the logit function
                    fair_scores = np.log(fair_likelihoods) - np.log(1 - fair_likelihoods)
                    # print('fair scores', fair_scores.shape, fair_scores)

                    # Rescore the most popular items using fairness

                    scores[recommendation_ids == item_id] = -1
                    scores = np.hstack((scores, fair_scores.reshape((-1, 1))))
                    recommendation_ids = np.hstack((recommendation_ids, np.full((fair_scores.shape[0], 1), item_id)))

            # Reorganize scores modified with fairness to get updated reco ids
            order = np.argsort(scores)[:, ::-1][:, :self.top_k]
            recommendation_ids = np.take_along_axis(recommendation_ids, order, 1)
            scores = np.take_along_axis(scores, order, 1)

        elif self.post_proc_type == POST_PROC_TYPE_TRACK_ACTIVITY_FAIRNESS:
            is_member = (self.track_popularity_summary['track_popularity_bin_index'] <= self.is_member_max_positive_activity_bin).astype(int).values

            recommendation_ids_all = []
            scores_all = []
            for user_id in test_ids:
                if user_id in self.mitigations:
                    likelihoods = self.generate_scores_for_user(user_id)
                    likelihoods = np.squeeze(likelihoods)
                    # take sigmoid
                    likelihoods = 1. / (1. + np.exp(-likelihoods))
                    # print('likelihoods', likelihoods)

                    # use only the top 10% as a positive prediction
                    binary_cutoff = np.quantile(likelihoods, self.post_proc_positive_quantile)
                    predictions = (likelihoods > binary_cutoff).astype(int)
                    # print('predictions', predictions)

                    mitigation = self.mitigations[user_id]
                    fair_predictions, fair_likelihoods = mitigation.transform(predictions, likelihoods, is_member)

                    # To go from likelihoods to score, we can use the logit function
                    fair_scores = np.log(fair_likelihoods) - np.log(1 - fair_likelihoods)
                    recommendation_ids = np.argsort(fair_scores)[::-1][:self.top_k]
                    scores = fair_scores[recommendation_ids]
                    recommendation_ids_all.append(np.squeeze(recommendation_ids))
                    scores_all.append(np.squeeze(scores))

                else:
                    recommendation_ids, scores = self.generate_scores(user_id)
                    recommendation_ids_all.append(np.squeeze(recommendation_ids))
                    scores_all.append(np.squeeze(scores))

            recommendation_ids = np.vstack(recommendation_ids_all)
            scores = np.vstack(scores_all)
        else:
            recommendation_ids, scores = self.generate_scores(test_ids)
        return recommendation_ids, scores

    def _recommend(self, user_ids: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:

        if self.use_average:
            res_dfs = []
            scores_dfs = []

            for bin in self.bins[self.cat_key]:
                cat = self.cat_key + '_' + str(bin)
                user_ids_cat = user_ids.loc[user_ids['user_id'].isin(self.user_id_map[cat].keys())]
                res_df_cat, scores_df_cat = self.predict_cat(user_ids_cat, cat)
                res_dfs.append(res_df_cat)
                scores_dfs.append(scores_df_cat)

            recs_df_combined = pd.concat(res_dfs).fillna(0)
            scores_df_combined = pd.concat(scores_dfs).fillna(0)

            recs_df_all, scores_df_all = self._predict_cat(user_ids, ALL_USERS_CAT)
            recommendation_ids, scores = self._take_average_score(recs_df_combined, scores_df_combined, recs_df_all,
                                                                 scores_df_all)
        else:
            test_ids = user_ids['user_id'].map(self.user_id_map[ALL_USERS_CAT]).values
            recommendation_ids, scores = self.impl[ALL_USERS_CAT].recommend(test_ids,
                                                                            self.train_data[ALL_USERS_CAT][test_ids, :],
                                                                            N=self.top_k,
                                                                            filter_already_liked_items=False)
        return recommendation_ids, scores

    def predict(self, user_ids: pd.DataFrame) -> pd.DataFrame:
        """
        
        This function takes as input all the users that we want to predict the top-k items for, and 
        returns all the predicted songs.

        While in this example is just a random generator, the same logic in your implementation 
        would allow for batch predictions of all the target data points.

        """

        # all_scores = np.dot(test_factors, self.impl.item_factors.T)
        # print('all scores', all_scores.shape, all_scores)

        if self.use_average:
            recommendation_ids, scores = self.recommend(user_ids)
        else:
            test_ids = user_ids['user_id'].map(self.user_id_map[ALL_USERS_CAT]).values
            recommendation_ids, scores = self._postprocess(user_ids, test_ids)

        recs_df = pd.DataFrame(recommendation_ids).apply(lambda x: x.map(self.reverse_item_map[ALL_USERS_CAT]), axis=1)
        recs_df.insert(0, 'user_id', user_ids['user_id'].values)
        recs_df.columns = ['user_id', *[str(i) for i in range(self.top_k)]]
        return recs_df.set_index('user_id')
