from LinkedinAPI import LinkedinAPI
_USERNAME = 'ronifridman0@gmail.com'
_PASSWORD = '*******'

def main():
    api = LinkedinAPI(username=_USERNAME, password=_PASSWORD) # Creates an instance of the LinkedIn API class.

    ## Run this to pull more posts
    # api.get_all_posts()

    ## Run this to turn posts save until now into csv
    # df = api.load_posts_into_df()
    # df.to_csv("posts_csv.csv")

    # Run this to test function that tests a posts. This is how the input should look like and a pandas DF is returned.
    print(api.get_posts('ronifridman1', post_to_predict={'text': "Hi All!", 'number_of_images': 1}))


if __name__ == '__main__':
    main()
