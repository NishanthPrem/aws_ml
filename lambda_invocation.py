import boto3
import json
import os
from dotenv import load_dotenv


load_dotenv()

lambda_client = boto3.client('lambda')
event = {
    'bucket_name': os.getenv('BUCKET_NAME'),
    'train_file_key': 'data/train.csv',
    'test_file_key': 'data/test.csv'
}

response = lambda_client.invoke(
    FunctionName='aws-ml',
    InvocationType='Event',
    Payload=json.dumps(event)
)

print(response)
