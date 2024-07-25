import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
import pandas as pd
import mxnet as mx
from io import BytesIO

# Initialize Glue context and job
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# S3 paths
input_train_path = 's3://your-bucket-name/train.csv'
input_test_path = 's3://your-bucket-name/test.csv'
output_train_path = 's3://your-bucket-name/output/train.rec'
output_test_path = 's3://your-bucket-name/output/test.rec'

# Function to convert CSV to RecordIO
def convert_csv_to_recordio(input_path, output_path):
    # Read CSV data
    df = pd.read_csv(input_path)
    
    # Convert to RecordIO
    records = []
    for _, row in df.iterrows():
        record = mx.recordio.IRHeader(0, row.to_dict(), 0, 0)
        s = mx.recordio.pack(record)
        records.append(s)
    
    # Write to S3
    buffer = BytesIO()
    with mx.recordio.MXRecordIO(buffer, 'w') as record_file:
        for record in records:
            record_file.write(record)
    
    buffer.seek(0)
    s3_client = boto3.client('s3')
    bucket_name, key_name = output_path.replace('s3://', '').split('/', 1)
    s3_client.upload_fileobj(buffer, bucket_name, key_name)

# Convert and upload
convert_csv_to_recordio(input_train_path, output_train_path)
convert_csv_to_recordio(input_test_path, output_test_path)

job.commit()
