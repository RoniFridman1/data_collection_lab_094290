import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.cluster import KMeans
from gensim import corpora, models
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import textstat
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('stopwords')


def remove_emojis(text):
  # Regular expression to match all emojis
  emoji_pattern = re.compile("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F700-\U0001F77F"  # alchemical symbols
                          u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                          u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                          u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                          u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                          u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                          u"\U00002702-\U000027B0"  # Dingbats
                          u"\U000024C2-\U0001F251"
                          "]+", flags=re.UNICODE)
  return emoji_pattern.sub(r'', text)  # Replace emojis with empty string


def apply_decision_rule(df_processed):
    # Apply the decision rule to compute popularity_score
    df_processed['popularity_score'] = ((df_processed['likes'] * 1) + (df_processed['comments'] * 3) + (
            df_processed['shares'] * 9)) / (np.log(df_processed['followers'] + 1) + 1)
    df_processed['popularity_score'] = df_processed['popularity_score'].astype(int)
    df_processed['popularity_score'] = (df_processed['popularity_score'] - df_processed['popularity_score'].min()) / (
            df_processed['popularity_score'].max() - df_processed['popularity_score'].min())

    # Calculate the mean and standard deviation
    mean_popularity = df_processed['popularity_score'].mean()
    std_dev_popularity = df_processed['popularity_score'].std()

    # Calculate the threshold and apply it to tag the posts
    threshold = mean_popularity + 0.5 * std_dev_popularity
    df_processed['popularity_score'] = np.where(df_processed['popularity_score'] > threshold, 1.0, 0.0)

    return df_processed


# Function to encode a batch of text using TinyBERT, handling empty strings
def encode_text_batch_with_tinybert(texts, model, tokenizer):
    embeddings = []
    for text in texts:
        if text is not None and text == text and text.strip():
            encoded_input = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
            with torch.no_grad():
                output = model(**encoded_input)
            embeddings.append(output.last_hidden_state[:, 0, :].numpy())  # Get embeddings for [CLS] token
        else:
            # Return a zero vector for empty strings or strings that contain only whitespace
            embeddings.append(np.zeros(model.config.hidden_size))
    return np.vstack(embeddings)


# Function to process data in batches and apply the encoding function
def process_in_batches(data, batch_size, column_name, model, tokenizer, post_df):
    # Initialize an empty list to store the embeddings
    embeddings = []

    # Process each batch
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_embeddings = encode_text_batch_with_tinybert(batch, model, tokenizer)
        embeddings.extend(batch_embeddings)

    # Assign the embeddings to the DataFrame
    post_df[column_name] = embeddings


meta_industries_12 = {
    'Furniture and Home Furnishings Manufacturing': 'Manufacturing',
    'Investment Banking': 'Financial and Investment',
    'Architecture and Planning': 'Services',
    'Wholesale': 'Services',
    'Travel Arrangements': 'Services',
    'Ranching': 'Miscellaneous',
    'Hospitals and Health Care': 'Healthcare and Medical',
    'Book and Periodical Publishing': 'Services',
    'Printing Services': 'Services',
    'Professional Training and Coaching': 'Services',
    'Computers and Electronics Manufacturing': 'Manufacturing',
    'Shipbuilding': 'Manufacturing',
    'Public Policy Offices': 'Government and Public Policy',
    'Software Development': 'Technology',
    'Outsourcing and Offshoring Consulting': 'Services',
    'Retail Groceries': 'Retail and Consumer Goods',
    'Education Administration Programs': 'Education and Training',
    'Plastics Manufacturing': 'Manufacturing',
    'Renewable Energy Semiconductor Manufacturing': 'Manufacturing',
    'Computer Networking Products': 'Technology',
    'Events Services': 'Services',
    'Information Services': 'Services',
    'Food and Beverage Services': 'Services',
    'Semiconductor Manufacturing': 'Manufacturing',
    'Business Consulting and Services': 'Services',
    'Insurance': 'Services',
    'Financial Services': 'Services',
    'Wireless Services': 'Services',
    'Computer Hardware Manufacturing': 'Technology',
    'Public Safety': 'Services',
    'Maritime Transportation': 'Transportation and Logistics',
    'Tobacco Manufacturing': 'Manufacturing',
    'Writing and Editing': 'Services',
    'Veterinary Services': 'Services',
    'Staffing and Recruiting': 'Services',
    'Accounting': 'Services',
    'International Affairs': 'Government and Public Policy',
    'Spectator Sports': 'Miscellaneous',
    'Glass, Ceramics and Concrete Manufacturing': 'Manufacturing',
    'Chemical Manufacturing': 'Manufacturing',
    'Mining': 'Miscellaneous',
    'E-Learning Providers': 'Technology',
    'Security and Investigations': 'Services',
    'Translation and Localization': 'Services',
    'Automation Machinery Manufacturing': 'Technology',
    'Computer and Network Security': 'Technology',
    'Political Organizations': 'Government and Public Policy',
    'Environmental Services': 'Government and Public Policy',
    'Oil and Gas': 'Miscellaneous',
    'Real Estate': 'Real Estate and Construction',
    'Think Tanks': 'Government and Public Policy',
    'Executive Offices': 'Miscellaneous',
    'Law Practice': 'Services',
    'Nanotechnology Research': 'Miscellaneous',
    'International Trade and Development': 'Government and Public Policy',
    'Personal Care Product Manufacturing': 'Manufacturing',
    'Philanthropic Fundraising Services': 'Services',
    'Entertainment Providers': 'Media and Entertainment',
    'Market Research': 'Media and Entertainment',
    'Movies, Videos, and Sound': 'Media and Entertainment',
    'Sporting Goods Manufacturing': 'Manufacturing',
    'Graphic Design': 'Services',
    'Technology, Information and Internet': 'Technology',
    'IT Services and IT Consulting': 'Technology',
    'Retail Office Equipment': 'Retail and Consumer Goods',
    'Wholesale Import and Export': 'Services',
    'Capital Markets': 'Financial and Investment',
    'Law Enforcement': 'Services',
    'Freight and Package Transportation': 'Transportation and Logistics',
    'Industrial Machinery Manufacturing': 'Manufacturing',
    'Non-profit Organizations': 'Miscellaneous',
    'Retail Art Supplies': 'Retail and Consumer Goods',
    'Animation and Post-production': 'Media and Entertainment',
    'Transportation, Logistics, Supply Chain and Storage': 'Transportation and Logistics',
    'Aviation and Aerospace Component Manufacturing': 'Transportation and Logistics',
    'Fundraising': 'Financial and Investment',
    'Railroad Equipment Manufacturing': 'Transportation and Logistics',
    'Construction': 'Real Estate and Construction',
    'Investment Management': 'Financial and Investment',
    'Utilities': 'Miscellaneous',
    'Retail Luxury Goods and Jewelry': 'Retail and Consumer Goods',
    'Warehousing and Storage': 'Transportation and Logistics',
    'Media Production': 'Media and Entertainment',
    'Gambling Facilities and Casinos': 'Media and Entertainment',
    'Defense and Space Manufacturing': 'Manufacturing',
    'Facilities Services': 'Services',
    'Government Relations Services': 'Government and Public Policy',
    'Advertising Services': 'Media and Entertainment',
    'Paper and Forest Product Manufacturing': 'Manufacturing',
    'Packaging and Containers Manufacturing': 'Manufacturing',
    'Telecommunications': 'Technology',
    'Medical Equipment Manufacturing': 'Healthcare and Medical',
    'Beverage Manufacturing': 'Manufacturing',
    'Restaurants': 'Retail and Consumer Goods',
    'Leasing Non-residential Real Estate': 'Real Estate and Construction',
    'Newspaper Publishing': 'Media and Entertainment',
    'Armed Forces': 'Miscellaneous',
    'Appliances, Electrical, and Electronics Manufacturing': 'Manufacturing',
    'Hospitality': 'Services',
    'Pharmaceutical Manufacturing': 'Healthcare and Medical',
    'Research Services': 'Services',
    'Retail Apparel and Fashion': 'Retail and Consumer Goods',
    'Photography': 'Media and Entertainment',
    'Wellness and Fitness Services': 'Services',
    'Truck Transportation': 'Transportation and Logistics',
    'Consumer Services': 'Services',
    'Wholesale Building Materials': 'Services',
    'Human Resources Services': 'Services',
    'Airlines and Aviation': 'Transportation and Logistics',
    'Machinery Manufacturing': 'Manufacturing',
    'Individual and Family Services': 'Services',
    'Motor Vehicle Manufacturing': 'Manufacturing',
    'Performing Arts': 'Media and Entertainment',
    'Museums, Historical Sites, and Zoos': 'Media and Entertainment',
    'Broadcast Media Production and Distribution': 'Media and Entertainment',
    'Banking': 'Financial and Investment',
    'Recreational Facilities': 'Miscellaneous',
    'Government Administration': 'Government and Public Policy',
    'Public Relations and Communications Services': 'Media and Entertainment',
    'Fisheries': 'Miscellaneous',
    'Medical Practices': 'Healthcare and Medical',
    'Religious Institutions': 'Miscellaneous',
    'Online Audio and Video Media': 'Media and Entertainment',
    'Artists and Writers': 'Miscellaneous',
    'Biotechnology Research': 'Healthcare and Medical',
    'Legal Services': 'Services',
    'Retail': 'Retail and Consumer Goods',
    'Civil Engineering': 'Services',
    'Libraries': 'Miscellaneous',
    'Alternative Dispute Resolution': 'Miscellaneous',
    'Manufacturing': 'Miscellaneous',
    'Design Services': 'Services',
    'Dairy Product Manufacturing': 'Manufacturing',
    'Higher Education': 'Education and Training',
    'Civic and Social Organizations': 'Miscellaneous',
    'Textile Manufacturing': 'Manufacturing',
    'Venture Capital and Private Equity Principals': 'Financial and Investment',
    'Mental Health Care': 'Healthcare and Medical',
    'Musicians': 'Media and Entertainment',
    'Farming': 'Miscellaneous',
    'Computer Games': 'Media and Entertainment',
    'Strategic Management Services': 'Services',
    'Food and Beverage Manufacturing': 'Manufacturing',
    'Primary and Secondary Education': 'Education and Training',
    'Alternative Medicine': 'Healthcare and Medical',
    'Legislative Offices': 'Services',
    'Administration of Justice': 'Services',
    'Mobile Gaming Apps': 'Media and Entertainment'
}

additional_mapping = {
    'Information Technology & Services': 'Technology',
    'Consumer Goods': 'Retail and Consumer Goods',
    'Food Production': 'Manufacturing',
    'Computer Software': 'Technology',
    'Hospital & Health Care': 'Healthcare and Medical',
    'Biotechnology': 'Healthcare and Medical',
    'Food & Beverages': 'Retail and Consumer Goods',
    'Human Resources': 'Services',
    'Primary/Secondary Education': 'Education and Training',
    'Defense & Space': 'Miscellaneous',
    'Aviation & Aerospace': 'Transportation and Logistics',
    'Management Consulting': 'Services',
    'Semiconductors': 'Technology',
    'Transportation/Trucking/Railroad': 'Transportation and Logistics',
    'Non-profit Organization Management': 'Miscellaneous',
    'Internet': 'Technology',
    'E-learning': 'Education and Training',
    'Electrical & Electronic Manufacturing': 'Manufacturing',
    'Financial and Investment': 'Financial and Investment',
    'Real Estate and Construction': 'Real Estate and Construction',
    'Medical Practice': 'Healthcare and Medical',
    'Military': 'Miscellaneous',
    'Logistics & Supply Chain': 'Transportation and Logistics',
    'Airlines/Aviation': 'Transportation and Logistics',
    'Automotive': 'Manufacturing',
    'Professional Training & Coaching': 'Services',
    'Executive Office': 'Miscellaneous',
    'Entertainment': 'Media and Entertainment',
    'Staffing & Recruiting': 'Services',
    'Apparel & Fashion': 'Retail and Consumer Goods',
    'Computer & Network Security': 'Technology',
    'Security & Investigations': 'Services',
    'Research': 'Services',
    'Health, Wellness & Fitness': 'Healthcare and Medical',
    'Political Organization': 'Government and Public Policy',
    'Venture Capital & Private Equity': 'Financial and Investment',
    'Mechanical Or Industrial Engineering': 'Manufacturing',
    'Oil & Energy': 'Miscellaneous',
    'Renewables & Environment': 'Miscellaneous',
    'Public Relations & Communications': 'Services',
    'Online Media': 'Media and Entertainment',
    'Sports': 'Media and Entertainment',
    'Music': 'Media and Entertainment',
    'Broadcast Media': 'Media and Entertainment',
    'Computer Hardware': 'Technology',
    'Architecture & Planning': 'Services',
    'Packaging & Containers': 'Manufacturing',
    'Import & Export': 'Services',
    'Chemicals': 'Manufacturing',
    'Outsourcing/Offshoring': 'Services',
    'Design': 'Services',
    'Writing & Editing': 'Services',
    'Medical Device': 'Healthcare and Medical',
    'Mining & Metals': 'Miscellaneous',
    'Cosmetics': 'Retail and Consumer Goods',
    'Luxury Goods & Jewelry': 'Retail and Consumer Goods',
    'Wine & Spirits': 'Retail and Consumer Goods',
    'Printing': 'Services',
    'Publishing': 'Media and Entertainment',
    'International Trade & Development': 'Services',
    'Wireless': 'Technology',
    'Package/Freight Delivery': 'Transportation and Logistics',
    'Pharmaceuticals': 'Healthcare and Medical',
    'Computer Networking': 'Technology',
    'Education Management': 'Education and Training',
    'Marketing & Advertising': 'Services',
}

# List of all U.S. states for reference
us_states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida',
             'Georgia',
             'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
             'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
             'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
             'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee',
             'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']

# Create the mapping dictionary
geo_location_mapping = {state: state for state in us_states}

# Add mappings for city and metropolitan areas known to be in the United States
additional_us_locations = {
    'Washington DC-Baltimore Area': 'District of Columbia',
    'Greater Seattle Area': 'Washington',
    'Greater Boston': 'Massachusetts',
    'San Francisco Bay Area': 'California',
    'New York City Metropolitan Area': 'New York',
    'Salt Lake City Metropolitan Area': 'Utah',
    'South Carolina Metropolitan Area': 'South Carolina',
    'Los Angeles Metropolitan Area': 'California',
    'Kansas City Metropolitan Area': 'Kansas',
    'Texas Metropolitan Area': 'Texas',
    'Dallas-Fort Worth Metroplex': 'Texas',
    'Greater Madison Area': 'Wisconsin',
    'Greater Phoenix Area': 'Arizona',
    'Johnson City-Kingsport-Bristol Area': 'Tennessee',
    'Wausau-Stevens Point Area': 'Wisconsin'
}


# Sentiment Analysis Function
def calculate_sentiment(dataframe, text_column='text'):
    sid = SentimentIntensityAnalyzer()
    dataframe['sentiment'] = dataframe[text_column].apply(lambda x: sid.polarity_scores(x)['compound'])
    return dataframe


# Readability Scores Function
def calculate_readability_scores(dataframe, text_column='text'):
    dataframe['flesch_score'] = dataframe[text_column].apply(textstat.flesch_reading_ease)
    dataframe['gunning_fog'] = dataframe[text_column].apply(textstat.gunning_fog)
    dataframe['coleman_liau'] = dataframe[text_column].apply(textstat.coleman_liau_index)
    max_flesch_score = dataframe['flesch_score'].max()
    dataframe['inverted_flesch_score'] = (max_flesch_score + 1) - dataframe['flesch_score']
    dataframe['combined_score'] = dataframe[['inverted_flesch_score', 'gunning_fog', 'coleman_liau']].mean(axis=1)
    return dataframe


# Text Cleaning and LDA Topic Modeling Function
def assign_lda_topics(dataframe, text_column='text', num_topics=5):
    stopwords_set = set(nltk.corpus.stopwords.words('english'))
    custom_stopwords = ['one', 'time', 'great', 'like', 'see', 'en', 'la', 'que', 'el', "de", "los", "para", "las",
                        "una", "im", "un", "con", "es", "tu", "se", "1", "get", "use", "thank", "us"]
    all_stopwords = stopwords_set.union(custom_stopwords)

    # Clean text
    def clean_text(text):
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        return ' '.join([word for word in words if word.lower() not in all_stopwords])

    dataframe['cleaned_text'] = dataframe[text_column].apply(clean_text)

    # LDA Topic Modeling
    texts = dataframe['cleaned_text'].apply(lambda x: x.lower().split())
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15, random_state=42)
    dataframe['topic'] = [max(lda_model[corpus[i]], key=lambda x: x[1])[0] for i in range(len(dataframe))]

    return dataframe, lda_model


def text_analysis(df, text_column='text'):
    df['word_count'] = df[text_column].apply(lambda x: len(x.split()))
    df['line_count'] = df[text_column].apply(lambda x: x.count('\n'))
    df['multiple_exclamation_marks'] = df[text_column].apply(lambda x: len(re.findall(r'!{2,}', x)))
    df['ellipsis'] = df[text_column].apply(lambda x: len(re.findall(r'\.{3,}', x)))
    df['hashtag_count'] = df[text_column].apply(lambda x: len(re.findall(r'#(\w+)', x)))
    df['url_count'] = df[text_column].apply(lambda x: len(
        re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', x)))
    return df


def adjective_count(df, text_column='text'):
    df['adjective_count'] = df[text_column].apply(
        lambda x: len([word for word, pos in pos_tag(word_tokenize(x)) if pos.startswith('JJ')]))
    return df


keywords = [
    # Core Data Science Concepts
    'data science', 'data analysis', 'predictive analytics', 'descriptive analytics', 'prescriptive analytics',
    'data modeling', 'data wrangling', 'data mining', 'data cleaning', 'data visualization',

    # Machine Learning
    'machine learning', 'classification', 'regression', 'clustering', 'dimensionality reduction',
    'ensemble methods', 'random forest', 'gradient boosting', 'xgboost', 'lightgbm', 'catboost',
    'support vector machine', 'k-nearest neighbors', 'logistic regression', 'linear regression',
    'decision trees', 'neural networks', 'deep learning', 'convolutional neural networks', 'recurrent neural networks',
    'long short-term memory', 'transfer learning', 'reinforcement learning', 'feature selection', 'feature engineering',

    # Natural Language Processing
    'natural language processing', 'nlp', 'text mining', 'sentiment analysis', 'text classification',
    'language model', 'tokenization', 'lemmatization', 'stemming', 'word embedding', 'bag of words',
    'tf-idf', 'topic modeling', 'named entity recognition', 'part-of-speech tagging', 'machine translation',

    # Computer Vision
    'computer vision', 'image processing', 'object detection', 'image classification', 'image segmentation',
    'facial recognition', 'optical character recognition', 'edge detection', 'pattern recognition',

    # Data Visualization Tools
    'matplotlib', 'seaborn', 'plotly', 'ggplot', 'bokeh', 'd3.js', 'tableau', 'power bi', 'qlikview',
    'chart', 'graph', 'plot', 'histogram', 'scatterplot',

    # Statistical Analysis
    'statistics', 'bayesian statistics', 'hypothesis testing', 'anova', 'correlation', 'regression analysis',
    'time series analysis', 'multivariate analysis', 'factor analysis', 'principal component analysis',

    # Big Data Technologies
    'big data', 'hadoop', 'spark', 'kafka', 'flink', 'hive', 'pig', 'cassandra', 'mongodb',

    # AI and Robotics
    'artificial intelligence', 'ai', 'robotics', 'autonomous vehicles', 'drone technology',

    # Programming Languages and Tools
    'python', 'r', 'sql', 'java', 'scala', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',

    # Cloud Computing & DevOps
    'aws', 'azure', 'google cloud', 'docker', 'kubernetes', 'cloud computing', 'virtualization',

    # Blockchain and Cryptocurrency
    'blockchain', 'bitcoin', 'ethereum', 'smart contracts', 'cryptocurrency',

    # IoT and Edge Computing
    'internet of things', 'iot', 'edge computing', 'sensor networks', 'smart cities',

    # Ethics and Privacy
    'data privacy', 'data security', 'cybersecurity', 'ethical ai', 'data governance',

    # Quantum Computing
    'quantum computing', 'quantum mechanics', 'qubits', 'quantum cryptography',

    # Miscellaneous
    'augmented reality', 'virtual reality', 'digital twin', '5g', 'bioinformatics', 'genomics',
]


def keyword_features(df, text_column='text', keywords=[]):
    def count_keywords(text, keyword_list):
        keyword_counts = {}
        for keyword in keyword_list:
            keyword_counts[keyword] = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text.lower()))
        return keyword_counts

    df['keyword_counts'] = df[text_column].apply(lambda x: count_keywords(x, keywords))
    df['total_keyword_count'] = df['keyword_counts'].apply(lambda x: sum(x.values()))
    df['unique_keyword_count'] = df['keyword_counts'].apply(lambda x: len([k for k, v in x.items() if v > 0]))
    df['keyword_diversity'] = df['unique_keyword_count'] / (df['total_keyword_count'] + 1)
    return df


def apply_clustering(df, embeddings_column='post_embeddings', n_clusters=10):
    embeddings = np.vstack(df[embeddings_column].values)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(embeddings)
    return df


def preprocess(posts_df, train_flag=False):
    # posts_df = apply_decision_rule(posts_df)

    # Apply the function to each text entry
    posts_df['text'] = posts_df['text'].apply(remove_emojis)
    # only emojis
    posts_df = posts_df[posts_df['text'].apply(lambda x: any(char.isalnum() for char in x))]

    # Remove text shorter than 4 words
    posts_df['short_sentence'] = posts_df['text'].apply(lambda x: len(x.split()))
    posts_df = posts_df[posts_df['short_sentence'] >= 4].drop('short_sentence', axis=1)

    # Only relevant records
    if train_flag:
      posts_df = posts_df[posts_df['prediction'] == 'Other Topics']
      posts_df = posts_df.drop(['prediction', 'Unnamed: 13', 'Unnamed: 0'], axis=1)

    # Missing
    posts_df['geoLocationName'] = posts_df['geoLocationName'].fillna('Other')
    posts_df['industryName'] = posts_df['industryName'].fillna('Other')

    # Text embeddings
    # Initialize TinyBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
    model = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
    # Apply the function to each text column
    process_in_batches(posts_df['text'].tolist(), 1000, 'post_embeddings', model, tokenizer, posts_df)
    process_in_batches(posts_df['summary'].tolist(), 1000, 'about_embeddings', model, tokenizer, posts_df)
    process_in_batches(posts_df['headline'].tolist(), 1000, 'headline_embeddings', model, tokenizer, posts_df)

    # industryName
    posts_df['industryName'] = posts_df['industryName'].map(meta_industries_12).fillna(posts_df['industryName'])
    # Update the meta_industries_12 dictionary with the additional mappings
    meta_industries_12.update(additional_mapping)
    # Apply the updated mapping to the industryName column
    posts_df['industryName'] = posts_df['industryName'].map(meta_industries_12).fillna(posts_df['industryName'])

    # geoLocationName
    # The 'geoLocationName' column has the format "City, Country"
    posts_df['geoLocationName'] = posts_df['geoLocationName'].apply(lambda x: x.split(', ')[-1] if pd.notnull(x) else x)
    # Update the mapping dictionary with the additional U.S. locations
    geo_location_mapping.update(additional_us_locations)
    # Apply the mapping
    posts_df['geoLocationName'] = posts_df['geoLocationName'].map(geo_location_mapping).fillna('Non-United States')

    # Creating training features from the posts
    posts_df = calculate_sentiment(posts_df, 'text')
    posts_df = calculate_readability_scores(posts_df, 'text')
    posts_df, lda_model = assign_lda_topics(posts_df, 'text')
    posts_df = text_analysis(posts_df, 'text')
    posts_df = adjective_count(posts_df, 'text')
    posts_df = keyword_features(posts_df, 'text', keywords)

    with open('kmeans_model.pkl', 'rb') as file:
        kmeans_loaded = pickle.load(file)
    embeddings = np.vstack(posts_df["post_embeddings"].values)
    new_data_clusters = kmeans_loaded.predict(embeddings)
    posts_df['cluster'] = new_data_clusters

    return posts_df


