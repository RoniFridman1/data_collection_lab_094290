import os
import random
import tqdm
from linkedin_api import Linkedin
import time
import json
import pandas as pd

DEFAULT_OUTPUT_PATH = 'data/posts.json'
DEFAULT_INPUT_PATH = 'data/data_users_with_posts.csv'

class LinkedinAPI:
    """
    API for pulling all relevant info about user, and it's latest available posts.
    """

    def __init__(self, username, password):
        self.api = Linkedin(username, password)
        self.ids = pd.unique(pd.read_csv(DEFAULT_INPUT_PATH)['id'])
        self.output_path = DEFAULT_OUTPUT_PATH
        self.post_to_pred = {}
        if os.path.exists(self.output_path):
            try:
                self.posts = json.loads(open(self.output_path).read())
            except Exception as e:
                print(f"Failed loading posts.\n Error: {e}")
        else:
            self.posts = {}

    def get_all_posts(self):
        for i, id in enumerate(tqdm.tqdm(self.ids)):
            if id in self.posts.keys():
                continue
            if len(self.posts) % 5 == 0:
                print(f"Finished {i} users already!")
                self.save_posts()

            try:
                self.get_posts(id)

            except PermissionError:
                print(f"permission denied for id: {id}. Skipping.")
            except Exception as e:
                print(f'Error: {e}')
                self.save_posts()
                return

    def get_posts(self, id, n_posts=10, post_to_predict: dict | None = None):
        """
        pull available posts from user's account.
        :param n_posts: Number of posts to pull.
        :param id: LinkedIn id
        :param post_to_predict: If None, meaning that we need to scrap posts from user. Else, should be a
        dictionary with these fields: text, number_of_images.
        :return: None if post_to_predict is none. Else, returns
        """
        profile = self.api.get_profile(id)
        if profile == {}:
            self.posts[id] = {'profile': {}, 'posts': {}}
            raise PermissionError
        urn = profile.get('member_urn')
        connections = self.api.get_profile_network_info(id)
        if connections == {}:
            self.posts[id] = {'profile': {}, 'posts': {}}
            raise PermissionError
        followers_count, connections_count = connections.get('followersCount'), connections.get('connectionsCount')
        profiles_info = {}
        for att in ['headline', 'geoLocationName', 'summary', 'industryName']:
            if att in profile.keys():
                profiles_info[att] = profile.get(att)
            else:
                profiles_info[att] = ''
        profiles_info['followers'] = followers_count
        profiles_info['connections'] = connections_count

        if post_to_predict is None:
            posts = [self.extract_info_on_post(post) for post in self.api.get_profile_posts(public_id=id, post_count=10)
                     if
                     post['actor']['urn'] == urn]
            if len(posts) > n_posts:
                posts = posts[:n_posts]

            self.posts[id] = {'profile': profiles_info, 'posts': posts}
            m = random.random() * 5
            time.sleep(m)
        elif isinstance(post_to_predict, dict):
            self.post_to_pred[id] = {'profile': profiles_info, 'posts': [post_to_predict]}
            df = self.load_posts_into_df(predict=True)
            return df
        else:
            raise TypeError

    def load_posts_into_df(self, predict=False):
        df_list = []
        posts_to_df = self.post_to_pred.items() if predict else self.posts.items()
        for k, v in posts_to_df:
            if len(v['posts']) == 0:
                continue
            id = k
            headline = v['profile']['headline']
            geoLocationName = v['profile']['geoLocationName']
            summary = v['profile']['summary']
            industryName = v['profile']['industryName']
            followes = v['profile']['followers']
            connections = v['profile']['connections']

            for post in v['posts']:
                text = post['text']
                likes = None if predict else post['likes']
                comments =  None if predict else post['comments']
                shares =  None if predict else post['shares']
                number_of_images = post['number_of_images']
                df_list.append(
                    [id, headline, geoLocationName, summary, industryName, followes, connections, text, likes, comments,
                     shares, number_of_images])
        df = pd.DataFrame(df_list, columns=['id', 'headline', 'geoLocationName', 'summary', 'industryName', 'followers',
                                            'connections', 'text', 'likes', 'comments', 'shares', 'number_of_images'])
        return df

    def extract_info_on_post(self, post):
        social_details = post['socialDetail']['totalSocialActivityCounts']
        likes = social_details['numLikes']
        comments = social_details['numComments']
        shares = social_details['numShares']
        text = ''
        if 'commentary' in post.keys():
            if 'text' in post['commentary'].keys():
                text = post['commentary']['text']['text']
        number_of_images = 0
        if 'content' in post.keys():
            for c in post['content'].values():
                if 'images' in c.keys():
                    number_of_images += len(c['images'])

        return {'text': text, 'likes': likes, 'comments': comments, 'shares': shares,
                'number_of_images': number_of_images}

    def save_posts(self):
        with open(self.output_path, 'w+') as outfile:
            json.dump(self.posts, outfile)
            outfile.close()
