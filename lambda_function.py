import os
import boto3
import pandas as pd
from io import StringIO
from botocore.exceptions import NoCredentialsError

s3_client = boto3.client('s3')
sagemaker_client = boto3.client('sagemaker')
sagemaker_runtime_client = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):
    try:
        # Get bucket and file information from event
        bucket_name = event['bucket_name']
        train_file_key = event['train_file_key']
        test_file_key = event['test_file_key']
        output_predictions_key = 'output/predictions.csv'

        # Retrieve train and test files from S3
        train_obj = s3_client.get_object(Bucket=bucket_name, Key=train_file_key)
        test_obj = s3_client.get_object(Bucket=bucket_name, Key=test_file_key)

        train_data = pd.read_csv(StringIO(train_obj['Body'].read().decode('utf-8')))
        test_data = pd.read_csv(StringIO(test_obj['Body'].read().decode('utf-8')))

        # Preprocess the data
        train_data = preprocess_data(train_data)
        test_data = preprocess_data(test_data)

        # Save the preprocessed data back to S3
        s3_client.put_object(Bucket=bucket_name, Key='output/preprocessed_train.csv', Body=train_data.to_csv(index=False))
        s3_client.put_object(Bucket=bucket_name, Key='output/preprocessed_test.csv', Body=test_data.to_csv(index=False))

        # Initiate SageMaker training job
        training_job_name = train_sagemaker(bucket_name, 'output/preprocessed_train.csv')

        # Wait for the training job to complete
        waiter = sagemaker_client.get_waiter('training_job_completed_or_stopped')
        waiter.wait(TrainingJobName=training_job_name)

        # Use the trained model to predict the test data
        predictions = predict_sagemaker(training_job_name, test_data)

        # Save the predictions to S3
        predictions.to_csv('/tmp/predictions.csv', index=False)
        s3_client.upload_file('/tmp/predictions.csv', bucket_name, output_predictions_key)

        return {
            'statusCode': 200,
            'body': 'Preprocessing, training, and prediction completed successfully.'
        }
    except NoCredentialsError:
        return {
            'statusCode': 403,
            'body': 'Credentials not available.'
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': str(e)
        }

def preprocess_data(df):
    # Example preprocessing steps
    df = df.fillna(0)
    # Add more preprocessing steps as needed
    return df

def train_sagemaker(bucket, train_key, test_key, output_key):
    role = 'arn:aws:iam::123456789012:role/SageMakerExecutionRole'  # The SageMaker execution role ARN
    training_job_name = 'aws-ml-train'
    
    response = sagemaker_client.create_training_job(
        TrainingJobName=training_job_name,
        AlgorithmSpecification={
            'TrainingImage': '632365934929.dkr.ecr.us-west-1.amazonaws.com/knn:1',
            'TrainingInputMode': 'File'
        },
        RoleArn=role,
        InputDataConfig=[
            {
                'ChannelName': 'train',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': f's3://{bucket}/{train_key}',
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                },
                'ContentType': 'application/x-recordio-protobuf'
            },
            {
                'ChannelName': 'test',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': f's3://{bucket}/{test_key}',
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                },
                'ContentType': 'application/x-recordio-protobuf'
            }
        ],
        OutputDataConfig={
            'S3OutputPath': f's3://{bucket}/{output_key}'
        },
        ResourceConfig={
            'InstanceType': 'ml.m5.large',
            'InstanceCount': 1,
            'VolumeSizeInGB': 10
        },
        StoppingCondition={
            'MaxRuntimeInSeconds': 3600
        }
    )
    
    return response
