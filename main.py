from preprocessing import preprocess
from train import train
import pandas as pd

if __name__ == '__main__':
    post_df = pd.read_csv("posts_res.csv")
    post_df = preprocess(post_df, train_flag=True)
    model_name = "Logistic Regression"
    train(post_df, model_name)
