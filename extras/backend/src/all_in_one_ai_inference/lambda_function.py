import boto3
import uuid
import base64
import traceback
from botocore.client import Config

s3_client = boto3.client('s3', config=Config(signature_version='s3v4'))

config = Config(
    read_timeout=120,
    retries={
        'max_attempts': 0
    }
)

sagemaker_runtime_client = boto3.client('sagemaker-runtime', config = config)
sagemaker_client = boto3.client('sagemaker')

account_id = boto3.client("sts").get_caller_identity().get("Account")
region_name = boto3.session.Session().region_name
bucket = f'sagemaker-{region_name}-{account_id}'
prefix = f's3://{bucket}/stable-diffusion-webui/asyncinvoke/in/'


def lambda_handler(event, context):
    if event['httpMethod'] == 'POST':
        payload = event['body']

        print(event['headers'])
        print(event['queryStringParameters'])

        if('Content-Type' in event['headers']):
            content_type = event['headers']['Content-Type']
        elif('content-type' in event['headers']):
            content_type = event['headers']['content-type']
        else:
            content_type = None

        endpoint_name = event['queryStringParameters']['endpoint_name']

        try:
            body = payload if(content_type == 'application/json') else base64.b64decode(payload)

            response = sagemaker_client.describe_endpoint(
                EndpointName = endpoint_name
            )

            infer_type = 'async' if ('AsyncInferenceConfig' in response) else 'sync'

            if(infer_type == 'sync'):
                response = sagemaker_runtime_client.invoke_endpoint(
                    EndpointName = endpoint_name,
                    ContentType = content_type,
                    Body = body)

                print(response)
                body = response['Body'].read()
            else:
                key = f'{prefix}{uuid.uuid4()}.json'
                s3_client.put_object(
                    Body=payload,
                    Bucket=bucket,
                    Key=key
                )
                response = sagemaker_runtime_client.invoke_endpoint_async(
                    EndpointName=endpoint_name,
                    ContentType="application/json",
                    InputLocation=key
                )
                print(response)
                body = response["OutputLocation"]

            return {
                'statusCode': 200,
                'body': body
            }

        except Exception as e:
            traceback.print_exc()
            return {
                'statusCode': 400,
                'body': str(e)
            }
    else:
        return {
            'statusCode': 400,
            'body': "Unsupported HTTP method"
        }
