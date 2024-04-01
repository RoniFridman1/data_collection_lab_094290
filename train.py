from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle
from preprocessing import preprocess


def preprocess_data(post_df):
    def flatten_if_nested(embedding):
        if len(embedding) == 1:
            return embedding[0]
        return embedding

    # Flatten the embeddings and stack them horizontally
    X_headline_embeddings = np.array([flatten_if_nested(e) for e in post_df['headline_embeddings']])
    X_about_embeddings = np.array([flatten_if_nested(e) for e in post_df['about_embeddings']])
    X_post_embeddings = np.array([flatten_if_nested(e) for e in post_df['post_embeddings']])
    X_embeddings = np.hstack((X_headline_embeddings, X_about_embeddings, X_post_embeddings))

    # Prepare other scalar features
    scalar_feature_names = ['followers', 'number_of_images', 'sentiment', 'word_count', 'line_count', 'hashtag_count', 'url_count', 'adjective_count', 'total_keyword_count', 'flesch_score', 'gunning_fog', 'coleman_liau', 'combined_score', 'unique_keyword_count', 'ellipsis']
    X_scalar = post_df[scalar_feature_names].values

    # One-hot encode categorical features
    categorical_feature_names = ['industryName', 'topic', 'cluster', 'geoLocationName']
    one_hot_encoder = OneHotEncoder()
    X_categorical = one_hot_encoder.fit_transform(post_df[categorical_feature_names]).toarray()

    # Save the encoder
    with open('one_hot_encoder.pkl', 'wb') as encoder_file:
        pickle.dump(one_hot_encoder, encoder_file)

    # Combine all features
    X = np.hstack((X_embeddings, X_scalar, X_categorical))
    y = post_df['popularity_score'].values

    # Feature names
    embedding_feature_names = ['headline_embedding_{}'.format(i) for i in range(X_headline_embeddings.shape[1])] + ['about_embedding_{}'.format(i) for i in range(X_about_embeddings.shape[1])] + ['post_embedding_{}'.format(i) for i in range(X_post_embeddings.shape[1])]
    one_hot_feature_names = one_hot_encoder.get_feature_names_out(categorical_feature_names).tolist()
    feature_names = embedding_feature_names + scalar_feature_names + one_hot_feature_names

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save the scaler
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred):
  metrics = {
    'Accuracy': accuracy_score(y_true, y_pred),
    'Precision': precision_score(y_true, y_pred, average='binary'),
    'Recall': recall_score(y_true, y_pred, average='binary'),
    'F1 Score': f1_score(y_true, y_pred, average='binary')
  }
  return metrics

def train(post_df, model_name):

    X_train, X_test, y_train, y_test = preprocess_data(post_df)

    rf_classifier = RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=25,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        class_weight='balanced',
    )

    logistic_model = LogisticRegression(max_iter=1000, solver="liblinear", C=0.5, random_state=42,
                                        class_weight="balanced")

    if model_name == "Logistic Regression":
        model = logistic_model
    else:
        model = rf_classifier

    # Train
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    results = evaluate_model(y_test, y_pred)
    for metric, value in results.items():
        print(f'{metric}: {value:.4f}')

    # Save the model
    with open('model.pkl', 'wb') as file:
        pickle.dump(logistic_model, file)
