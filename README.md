
<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.5em"> LinkedIn Post Analyzer<br> </h1>
<p align='center' style="text-align:center;font-size:1em;">
    <a>Shelly Serafimovich</a>&nbsp;,&nbsp;
    <a>Arik Ontman</a>&nbsp;,&nbsp;
    <a>Roni Fridman</a>&nbsp;
    <br/> 
    Technion - Israel Institute of Technology<br/> 
    
</p>

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

![alt text](https://github.com/RoniFridman1/data_collection_lab_094290/blob/main/lab_in_data_collection_model.png?raw=true)


## Getting Started
Initially, we'd like to point out that within the collab notebook, all our code is accompanied by comprehensive explanations detailing every step we've undertaken.

To get started with the LinkedIn Post Analyzer, follow these steps:

1. **Clone the Repository:**
git clone https://github.com/your-repo/linkedin-post-analyzer.git

2. **Install Dependencies:**
Navigate to the project directory and install the required dependencies:
pip install -r requirements.txt

3. **Running the model:**
    - In order to run the model, you cab run the file `main.py` and if you want to change the model type, you can do it in the `model_name` field.

4. **Running the Application:**
    - 4.1 First, connect to your LinkedIn user in a browser in the background.
    - 4.2 Update the `api_key` to your gpt api key in `lab_gui.py` in line 14.
    - 4.3 Run the following command in the terminal:
      ```
      streamlit run lab_gui.py
      ```
    - 4.4 After registering in the interface, confirm the verification email you receive.
      
![alt text](https://github.com/RoniFridman1/data_collection_lab_094290/blob/main/GUI.png?raw=true)

## Contributing

We welcome contributions from the community. If you're interested in enhancing the LinkedIn Post Analyzer, feel free to fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

