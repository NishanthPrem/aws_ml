import boto3
import os
from dotenv import load_dotenv

load_dotenv()
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_default_region = os.getenv('AWS_DEFAULT_REGION')

s3 = boto3.client('s3',
                  aws_access_key_id=aws_access_key_id,
                  aws_secret_access_key=aws_secret_access_key,
                  region_name=aws_default_region )
bucket_name = '0708-aws-ml'

s3.upload_file('train.csv', bucket_name, 'data/train.csv')
s3.upload_file('test.csv', bucket_name, 'data/test.csv')
print(f'File upload complete')