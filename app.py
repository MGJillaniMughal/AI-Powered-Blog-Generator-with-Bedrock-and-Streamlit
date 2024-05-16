import boto3
import botocore.config
import json
from datetime import datetime

def generate_blog_on_topic(topic: str) -> str:
    """Generate a blog post on a specified topic using the Bedrock AI model."""
    prompt = f"<s>[INST]Human: Write a concise 200-word blog post on {topic}.\nAssistant:[/INST]</s>"
    body = {
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9
    }

    try:
        bedrock = boto3.client("bedrock-runtime", region_name="us-east-1",
                               config=botocore.config.Config(read_timeout=300, retries={'max_attempts': 3}))
        response = bedrock.invoke_model(body=json.dumps(body), modelId="meta.llama3-8b-instruct-v1:0")
        response_content = response.get('body').read()
        blog_content = json.loads(response_content)['generation']
        print("Blog generation successful.")
        return blog_content
    except Exception as e:
        print(f"Error during blog generation: {e}")
        return None

def save_blog_to_s3(s3_bucket: str, s3_key: str, blog_content: str) -> None:
    """Save the generated blog content to an S3 bucket."""
    s3 = boto3.client('s3')
    try:
        s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=blog_content)
        print("Blog content successfully saved to S3.")
    except Exception as e:
        print(f"Error saving blog to S3: {e}")

def lambda_handler(event, context):
    """AWS Lambda handler function for processing blog generation requests."""
    event_data = json.loads(event['body'])
    topic = event_data['blog_topic']

    blog_content = generate_blog_on_topic(topic=topic)
    if blog_content:
        current_time = datetime.now().strftime('%Y%m%d%H%M%S')
        s3_key = f"blog-posts/{current_time}.txt"
        s3_bucket = 'aws-bedrock-course1'
        save_blog_to_s3(s3_bucket, s3_key, blog_content)
    else:
        print("Failed to generate blog content.")

    return {
        'statusCode': 200,
        'body': json.dumps('Blog generation and storage completed.')
    }
