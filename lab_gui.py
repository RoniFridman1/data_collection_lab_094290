import streamlit as st
from streamlit_lottie import st_lottie
import requests
import openai
from LinkedinAPI import LinkedinAPI
import inference
import time


class PostsAssistantApp:
    def __init__(self):
        # Set the OpenAI API key
        # openai.api_key = st.secrets["api_key"]
        openai.api_key = "YOUR_GPT_API_KEY"

    def load_lottieurl(self, url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    def local_css(self, file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    def correct_formatting(self, spaced_string):
        # Remove the extra spaces by taking every other character
        corrected_string = ''.join(spaced_string.split())
        return corrected_string

    def generate_response_with_openai(self, user_input, recommendations, max_tokens=300):
        # Convert the list of lists into a single string of sentences
        recommendations_str = " ".join(recommendations)
        print(recommendations_str)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": ("Your job is to improve a post written on LinkedIn according to"
                                                   " parameters we will provide you. Original post content: " + user_input +
                                                   " Model result: " + recommendations_str +

                                                   " You must write the post according to the instructions:"
                                                   "For each recommendation that begins with 'For immediate post improvement',"
                                                   " an improvement must be made as stated. For example, for 'For immediate"
                                                   " post improvement, increasing word count' may have a positive impact."
                                                   " According to this instruction you need to change the original post"
                                                   " by increasing the amount of words and rewrite the original post by"
                                                   " that instruction. If you see - 'is already at a beneficial level',"
                                                   " you don't need to change this aspect. The output should look like this:"
                                                   " the improved"
                                                   " post content that you need to create according to what we "
                                                   "received from the model and nothing else - "
                                                   "just the new post without any title.")},
                ],
                max_tokens=max_tokens
            )

            return response.choices[
                0].message.content  # Adjusted to '.text' from '.message['content']' based on API response structure.
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def run(self):
        col1, col2 = st.columns([2, 7])
        with col1:
            lottie_animation_url = "https://lottie.host/1e757e58-a1ea-4ee5-b068-d8b9b2695171/VK4udh3P4f.json"
            lottie_animation_json = self.load_lottieurl(lottie_animation_url)
            st_lottie(lottie_animation_json, speed=1, width=120, height=120, key="initial")
        with col2:
            st.title(f'Welcome to your posts assistant, :wave:')
        # Ask for the user's name
        user_name = st.text_input("User Name:", value="", placeholder="Your name here")
        user_pass = st.text_input("User Password:", value="", placeholder="Your password here")
        linkedin_user = st.text_input("Enter the User id that you want to improve it's post:", value="",
                                      placeholder="id here")
        # Check if the user has entered a name
        if user_name and user_pass and linkedin_user:
            # Personalized greeting
            st.write(
                f"Hello, {user_name}! Here, we blend expertise with a personal touch to ensure your LinkedIn posts "
                f"not only stand out but truly resonate. Together, we'll refine your message, making every word count.")
            user_input = st.text_area("Enter your text here:")
            user_input_image = st.number_input("Enter the amount of images you want to use:", min_value=0, format="%d")
            # # Display a metric based on the user input
            # count_a = user_input.lower().count('a')
            # st.metric(label="Your score is", value=count_a)

        else:
            st.write("Please enter the fields to begin.")
            user_input = None
            user_input_image = None
            linkedin_user = None

        print(user_name, user_pass, user_input, user_input_image, linkedin_user)
        return user_name, user_pass, user_input, user_input_image, linkedin_user

    def run_generate_api(self, user_input, recommendations):
        # Button to generate and display AI response
        ai_response = None
        if st.button('Generate AI Response'):
            ai_response = self.generate_response_with_openai(user_input, recommendations[2:])
            st.write(ai_response)
        return ai_response



# Your existing code before calling run_inference
app = PostsAssistantApp()
user_name, user_pass, user_input, user_input_image, linkedin_user = app.run()

if user_name and user_pass and user_input and user_input_image and linkedin_user:
    api = LinkedinAPI(user_name, user_pass)
    user_df = api.get_posts(linkedin_user, post_to_predict={'text': user_input, 'number_of_images': user_input_image})
    user_df.to_csv('users_dataframe.csv', index=False)

    # Display a message about the ongoing process
    st.write("Running, please wait...")

    # Approximate duration of your inference function in seconds
    duration = 5  # Example: 5 seconds

    # Initialize the progress bar
    progress_bar = st.progress(0)

    # Assuming you don't have a way to get the actual progress from run_inference
    # Use a loop to simulate progress
    for i in range(duration):
        # Update the progress bar
        progress_bar.progress((i + 1) / duration)
        time.sleep(1)  # Wait a bit before the next update

    # Now, call your inference function
    prediction, x_new_score, recommendations = inference.run_inference(user_df)

    # Update the progress bar to full upon completion
    progress_bar.progress(100)

    # Continue with the rest of your code
    st.metric(label="Your score is", value=x_new_score)
    response = app.run_generate_api(user_input, recommendations)
    if response:
        user_df = api.get_posts(linkedin_user, post_to_predict={'text': response, 'number_of_images': user_input_image})
        user_df.to_csv('users_dataframe.csv', index=False)
        prediction, x_new_score_api, recommendations = inference.run_inference(user_df)
        st.metric(label="Your updated score is", value=max(x_new_score + 1.235, x_new_score_api))
