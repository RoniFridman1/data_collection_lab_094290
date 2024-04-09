import pickle
from preprocessing import preprocess
import numpy as np
import torch
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings('ignore', category=InconsistentVersionWarning)


def load_model():
    # Load the trained model
    with open('logistic_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # Load the scaler
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Load the OneHotEncoder
    with open('one_hot_encoder.pkl', 'rb') as encoder_file:
        one_hot_encoder = pickle.load(encoder_file)

    # Load the saved percentiles
    with open('percentiles.pkl', 'rb') as file:
        percentiles = pickle.load(file)

    return model, scaler, one_hot_encoder, percentiles


def features_name(x, one_hot_encoder, x_new_headline_embeddings, x_new_about_embeddings, x_new_post_embeddings):
    scalar_feature_names = ['followers', 'number_of_images', 'sentiment', 'word_count',
                            'line_count', 'hashtag_count', 'url_count', 'adjective_count',
                            'total_keyword_count', 'flesch_score', 'gunning_fog',
                            'coleman_liau', 'combined_score', "unique_keyword_count", "ellipsis"]

    categorical_feature_names = ['industryName', 'topic', 'cluster', "geoLocationName"]

    # Feature names: embeddings + scalar features + one-hot encoded feature names
    embedding_feature_names = ['headline_embedding_{}'.format(i) for i in range(x_new_headline_embeddings.shape[1])] + \
                              ['about_embedding_{}'.format(i) for i in range(x_new_about_embeddings.shape[1])] + \
                              ['post_embedding_{}'.format(i) for i in range(x_new_post_embeddings.shape[1])]

    one_hot_feature_names = ['industryName_Education and Training',
                             'industryName_Financial and Investment',
                             'industryName_Government and Public Policy',
                             'industryName_Healthcare and Medical',
                             'industryName_Manufacturing',
                             'industryName_Media and Entertainment',
                             'industryName_Miscellaneous',
                             'industryName_Other',
                             'industryName_Real Estate and Construction',
                             'industryName_Retail and Consumer Goods',
                             'industryName_Services',
                             'industryName_Technology',
                             'industryName_Transportation and Logistics',
                             'topic_0',
                             'topic_1',
                             'topic_2',
                             'topic_3',
                             'topic_4',
                             'cluster_0',
                             'cluster_1',
                             'cluster_2',
                             'cluster_3',
                             'cluster_4',
                             'cluster_5',
                             'cluster_6',
                             'cluster_7',
                             'cluster_8',
                             'cluster_9',
                             'geoLocationName_Alabama',
                             'geoLocationName_Arizona',
                             'geoLocationName_Arkansas',
                             'geoLocationName_California',
                             'geoLocationName_Colorado',
                             'geoLocationName_Connecticut',
                             'geoLocationName_Delaware',
                             'geoLocationName_District of Columbia',
                             'geoLocationName_Florida',
                             'geoLocationName_Hawaii',
                             'geoLocationName_Idaho',
                             'geoLocationName_Illinois',
                             'geoLocationName_Iowa',
                             'geoLocationName_Kansas',
                             'geoLocationName_Kentucky',
                             'geoLocationName_Louisiana',
                             'geoLocationName_Maine',
                             'geoLocationName_Maryland',
                             'geoLocationName_Massachusetts',
                             'geoLocationName_Michigan',
                             'geoLocationName_Minnesota',
                             'geoLocationName_Mississippi',
                             'geoLocationName_Missouri',
                             'geoLocationName_Montana',
                             'geoLocationName_Nebraska',
                             'geoLocationName_Nevada',
                             'geoLocationName_New Hampshire',
                             'geoLocationName_New Jersey',
                             'geoLocationName_New York',
                             'geoLocationName_Non-United States',
                             'geoLocationName_North Carolina',
                             'geoLocationName_North Dakota',
                             'geoLocationName_Ohio',
                             'geoLocationName_Oklahoma',
                             'geoLocationName_Oregon',
                             'geoLocationName_Pennsylvania',
                             'geoLocationName_Rhode Island',
                             'geoLocationName_South Carolina',
                             'geoLocationName_South Dakota',
                             'geoLocationName_Tennessee',
                             'geoLocationName_Texas',
                             'geoLocationName_Utah',
                             'geoLocationName_Vermont',
                             'geoLocationName_Virginia',
                             'geoLocationName_Washington',
                             'geoLocationName_West Virginia',
                             'geoLocationName_Wisconsin']

    feature_names = embedding_feature_names + scalar_feature_names + one_hot_feature_names

    return feature_names, scalar_feature_names, categorical_feature_names


# Function to categorize the value into percentile bins
def get_percentile_bin(percentiles, value, feature_idx):
    feature_percentiles = percentiles[:, feature_idx]
    if value <= feature_percentiles[0]:
        return "very low"
    elif value <= feature_percentiles[1]:
        return "low"
    elif value <= feature_percentiles[2]:
        return "medium"
    elif value <= feature_percentiles[3]:
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
    model, scaler, one_hot_encoder, percentiles = load_model()

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
            current_value = x_new_scaled[0][idx]
            coef = feature_coefficients[feature]
            current_bin = get_percentile_bin(percentiles, current_value, idx)
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

    return prediction, x_new_score, recommendations


def run_inference(user_df):
    prediction, x_new_score, recommendations = make_inference(user_df)
    print(f"post prediction: {prediction[0]}")
    print(f"post score: {x_new_score * 100}")
    # for recommendation in recommendations:
    #     print(recommendation)
    return prediction[0], x_new_score * 100, recommendations
