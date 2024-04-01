import pickle
from preprocessing import preprocess
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

def load_model():
    # Load the trained model
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # Load the scaler
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Load the OneHotEncoder
    with open('one_hot_encoder.pkl', 'rb') as encoder_file:
        one_hot_encoder = pickle.load(encoder_file)

    return model, scaler, one_hot_encoder

def features_name(x, one_hot_encoder,x_new_headline_embeddings, x_new_about_embeddings, x_new_post_embeddings):
    scalar_feature_names = ['followers', 'number_of_images', 'sentiment', 'word_count',
                            'line_count', 'hashtag_count', 'url_count', 'adjective_count',
                            'total_keyword_count', 'flesch_score', 'gunning_fog',
                            'coleman_liau', 'combined_score', "unique_keyword_count", "ellipsis"]

    categorical_feature_names = ['industryName', 'topic', 'cluster', "geoLocationName"]

    # Feature names: embeddings + scalar features + one-hot encoded feature names
    embedding_feature_names = ['headline_embedding_{}'.format(i) for i in range(x_new_headline_embeddings.shape[1])] + \
                              ['about_embedding_{}'.format(i) for i in range(x_new_about_embeddings.shape[1])] + \
                              ['post_embedding_{}'.format(i) for i in range(x_new_post_embeddings.shape[1])]

    one_hot_feature_names = one_hot_encoder.get_feature_names_out(categorical_feature_names).tolist()
    feature_names = embedding_feature_names + scalar_feature_names + one_hot_feature_names

    return feature_names, scalar_feature_names, categorical_feature_names

def get_percentile_bin(value, feature_values):
    # Function to categorize the value into percentile bins
    percentiles = np.percentile(feature_values, [20, 40, 60, 80])
    if value <= percentiles[0]:
        return "very low"
    elif value <= percentiles[1]:
        return "low"
    elif value <= percentiles[2]:
        return "medium"
    elif value <= percentiles[3]:
        return "high"
    else:
        return "very high"

def flatten_if_nested(embedding):
    if len(embedding) == 1:
        return embedding[0]
    return embedding

def encode_text_with_tinybert(text, model, tokenizer):
    if text is not None and text.strip():
        encoded_input = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        with torch.no_grad():
            output = model(**encoded_input)
        return output.last_hidden_state[:, 0, :].numpy()  # Get embeddings for [CLS] token
    else:
        # Return a zero vector for empty strings or strings that contain only whitespace
        return np.zeros(model.config.hidden_size)

def make_inference(x_new):
    x_new = preprocess(x_new)
    model, scaler, one_hot_encoder = load_model()

    # Preparing x_new for the model
    x_new_headline_embeddings = np.array([flatten_if_nested(e) for e in x_new['headline_embeddings']])
    x_new_about_embeddings = np.array([flatten_if_nested(e) for e in x_new['about_embeddings']])
    x_new_post_embeddings = np.array([flatten_if_nested(e) for e in x_new['post_embeddings']])
    x_new_embeddings = np.hstack((x_new_headline_embeddings, x_new_about_embeddings, x_new_post_embeddings))

    # Feature names
    feature_names, scalar_feature_names, categorical_feature_names = features_name(x_new, one_hot_encoder,
                                                                                   x_new_headline_embeddings,
                                                                                   x_new_about_embeddings,
                                                                                   x_new_post_embeddings)

    x_new_scalar = x_new[scalar_feature_names].values
    x_new_categorical_transformed = one_hot_encoder.transform(x_new[categorical_feature_names]).toarray()
    x_new_vec = np.hstack((x_new_embeddings, x_new_scalar, x_new_categorical_transformed))

    x_new_scaled = scaler.transform(x_new_vec)
    # Make predictions using the loaded model
    prediction = model.predict(x_new_scaled)

    # Score
    x_new_score = model.predict_proba(x_new_scaled.reshape(1, -1))[0][1]


    # Get the coefficients from the model
    coefficients = model.coef_[0]

    # Create a dictionary mapping feature names to their coefficients
    feature_coefficients = {feature: coef for feature, coef in
                                                         zip(feature_names, coefficients) if
                                                         feature in scalar_feature_names}

    # Categorize features
    general_features = ['followers', 'number_of_images']
    post_improvement_features = [
        'sentiment', 'word_count', 'line_count', 'hashtag_count', 'url_count',
        'adjective_count', 'total_keyword_count', 'flesch_score', 'gunning_fog',
        'coleman_liau', 'combined_score', 'unique_keyword_count', 'ellipsis'
    ]

    # Iterate through each feature to generate recommendations
    recommendations = []
    for idx, feature in enumerate(scalar_feature_names):
      if feature in feature_coefficients:
        current_value = x_new_scaled[idx]
        coef = feature_coefficients[feature]
        current_bin = get_percentile_bin(current_value, X_train[:, idx])

        # Recommendations for general features for broader strategy
        if feature in general_features:
          recommendation = f"To improve future posts' performance, "
          if coef > 0:
            recommendation += f"increasing '{feature}' (currently '{current_bin}') can be beneficial. "
            recommendation += "Consider strategies to enhance this feature over time."
          else:
            recommendation += f"decreasing '{feature}' (currently '{current_bin}') can be beneficial. "
            recommendation += "Consider strategies to manage or mitigate this feature's influence."
        # Recommendations for content-related features for immediate post improvement
        else:
          recommendation = f"For immediate post improvement, "
          if coef > 0:
            if current_bin in ["very low", "low", "medium"]:
              recommendation += f"increasing '{feature}' may have a positive impact. "
            else:
              recommendation += f"'{feature}' is already at a beneficial level. "
          else:
            if current_bin in ["high", "very high", "medium"]:
              recommendation += f"reducing '{feature}' may improve performance. "
            else:
              recommendation += f"'{feature}' is already at a beneficial level. "

        recommendations.append(recommendation)

    return prediction, x_new_score, recommendation



df = pd.read_csv("posts_res.csv")
# Function to find user by username and return DataFrame with one row
def find_user_df(df, username):
    user_row = df[df['id'] == username]
    if len(user_row) == 0:
        return None
    else:
        return user_row

# Example usage
username_to_find = 'siavash-yasini'
user_df = find_user_df(df, username_to_find)
user_df = user_df[user_df["likes"]==176.0]
# df = preprocess(user_df)
prediction, x_new_score, recommendation = make_inference(user_df)
print(prediction)
print(x_new_score*100)
print(recommendation)