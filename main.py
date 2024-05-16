import streamlit as st
import boto3
import botocore.config
import json

def generate_blog_on_topic(blogtopic: str) -> str:
    """
    Generates a blog post on a specified topic using the Bedrock AI AWS Servises.
    
    Args:
    blogtopic (str): The topic on which the blog post should be written.
    
    Returns:
    str: The generated blog content or None if an error occurs.
    """
    # Define the prompt with detailed instructions for generating the blog post
    prompt = f"""
    <s>[INST]
    Human: Please write a detailed, engaging 200-word blog about '{blogtopic}'. The blog should be insightful and informative, tailored for an audience interested in current trends and practical advice. Use a friendly and professional tone, include a compelling introduction, provide valuable insights in the main body, and end with a thoughtful conclusion that encourages further reading or action.
    Assistant:[/INST]
    </s>
    """
    
    # Configuration for the AI model request
    body = {
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.2,
        "top_p": 0.7
    }

    try:
        # Create a client to interact with the Bedrock AI service
        bedrock = boto3.client("bedrock-runtime", region_name="us-east-1",
                               config=botocore.config.Config(read_timeout=300, retries={'max_attempts': 3}))
        # Invoke the AI model and get the response
        response = bedrock.invoke_model(body=json.dumps(body), modelId="meta.llama3-8b-chat-v1")
        response_content = response.get('body').read()
        response_data = json.loads(response_content)
        # Return the generated blog content
        return response_data['generation']
    except Exception as e:
        # Handle exceptions and display an error message in Streamlit
        st.error(f"Error generating the blog: {e}")
        return None

def main():
    """Main function to run the Streamlit app, providing a UI for blog generation."""
    st.title('AI-powered Blog Generator Using LLama 3 8B Calling From Bedrock')  # Title of the app
    st.write('This app Developed By Jillani Soft Tech 😎.')  # Description
    blog_topic = st.text_input('Enter the topic for the blog:', '')  # Input field for the blog topic

    if st.button('Generate Blog'):  # Button to trigger blog generation
        with st.spinner('Generating the blog... Please wait.'):  # Spinner to indicate processing
            blog_content = generate_blog_on_topic(blog_topic)
            if blog_content:
                st.success('Blog generated successfully!')  # Success message
                st.subheader('Generated Blog Content:')  # Subheader for displaying blog content
                st.write(blog_content)  # Display the generated content
            else:
                st.error('Failed to generate the blog. Please try again.')  # Error message if generation fails

if __name__ == '__main__':
    main()
