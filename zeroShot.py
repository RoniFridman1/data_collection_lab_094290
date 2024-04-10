import pandas as pd
from transformers import pipeline

# Initialize the Zero-Shot Classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def classify_texts_in_batches(df, batch_size=500):
    predictions = []
    for start in range(0, df.shape[0], batch_size):
        end = start + batch_size
        batch = df.iloc[start:end]
        for text in batch['text']:
            result = classifier(text, ["Job Announcement", "Other Topics"])
            prediction = result['labels'][0]
            predictions.append(prediction)
        print(f"Processed {end} texts...")
    df['prediction'] = predictions
    return df


def main():
    df = pd.read_csv('posts_9000_new.csv')  # Adjust the path as necessary
    result_df = classify_texts_in_batches(df)
    print(result_df)
    result_df.to_csv('posts_res.csv', index=False)


if __name__ == "__main__":
    main()
