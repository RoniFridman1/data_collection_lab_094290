# LinkedIn Post Analyzer

Welcome to the LinkedIn Post Analyzer, an innovative Python application developed as part of a university course assignment. This application is designed to evaluate the attractiveness of LinkedIn posts using advanced Natural Language Processing (NLP) and machine learning techniques. By analyzing posts based on various parameters, our app provides valuable insights into what makes a LinkedIn post successful.

## Project Overview

Our dataset comprises 1.5 million LinkedIn users, specifically filtered to include only those in the data field who have made some posts or have a significant number of followers. Utilizing a custom-built `LinkedinAPI` class, we automated the process of scraping LinkedIn posts, adhering to a maximum of 10 posts per user, which resulted in a comprehensive collection of 9,000 posts. Each record includes detailed information about both the post and its author.

The core functionality of our application revolves around a sophisticated pipeline that processes and analyzes post data to predict post attractiveness. This prediction is based on a supervised learning model that considers a linear combination of likes, comments, and shares as the target variable. Additionally, the app evaluates and reports on various aspects of the post, such as sentiment, text length, and keyword significance.

### Feature Extraction Pipeline

Our feature extraction pipeline incorporates several cutting-edge NLP processes:

- **Topic Classification:** Utilizes the Latent Dirichlet Allocation (LDA) algorithm for effective topic identification.
- **Level of Language:** Assessment of the complexity and sophistication of the post's language (TBD).
- **Keyword Analysis:** Identification and evaluation of important keywords within the text.
- **Mathematical Expression Detection:** Checks for the presence of mathematical expressions, leveraging LaTeX parsing.
- **Text Embeddings:** Converts text to embeddings using the compact but powerful Little BERT model.
- **K-Means Clustering:** Applies K-Means clustering with k=5 to the embeddings, incorporating the cluster group as an additional feature for analysis.

![alt text](https://github.com/RoniFridman1/data_collection_lab_094290/blob/main/model.png?raw=true)

_Please insert an appropriate visualization of the model or data processing pipeline here._

## Getting Started

To get started with the LinkedIn Post Analyzer, follow these steps:

1. **Clone the Repository:**
git clone https://github.com/your-repo/linkedin-post-analyzer.git

2. **Install Dependencies:**
Navigate to the project directory and install the required dependencies:
pip install -r requirements.txt

3. **Setup LinkedIn API Credentials:**
Ensure you have valid LinkedIn API credentials and update the `config.py` file with your credentials.

4. **Running the Application:**
Execute the main script to start analyzing LinkedIn posts:


## Contributing

We welcome contributions from the community. If you're interested in enhancing the LinkedIn Post Analyzer, feel free to fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- This project was made possible thanks to the extensive dataset provided by LinkedIn and the guidance of our university course instructors.
- Special thanks to all team members who contributed to the development and success of this project.

## Contact

For any queries or further discussions, feel free to contact the project team through the repository's issue tracker.

_We hope you find the LinkedIn Post Analyzer both useful and enlightening in understanding the dynamics of successful LinkedIn posts!_
