import boto3
import traceback
import json

sagemaker_client = boto3.client('sagemaker')

def lambda_handler(event, context):
    print(event['body'])

    try:
        body = json.loads(event['body'])
        training_job_name = body['training_job_name']
        hyperparameters = body['hyperparameters']
        hyperparameters['train-args'] = json.dumps(hyperparameters['train-args'])
        algorithm_specification = body['algorithm_specification']
        role_arn = body['role_arn']
        input_data_config = body['input_data_config']
        output_data_config = body['output_data_config']
        resource_config = body['resource_config']
        tags = body['tags']
    
        response = sagemaker_client.create_training_job(
            TrainingJobName = str(training_job_name),
            HyperParameters = json_encode_hyperparameters(hyperparameters),
            AlgorithmSpecification = algorithm_specification,
            RoleArn = role_arn,
            InputDataConfig = input_data_config,
            OutputDataConfig = output_data_config,
            ResourceConfig = resource_config,
            StoppingCondition = {
                'MaxRuntimeInSeconds': 86400
            },
            EnableNetworkIsolation = False,
            EnableInterContainerTrafficEncryption = False,
            EnableManagedSpotTraining = False,
            Tags = tags
        )
    
        print(response)
        return {
            'statusCode': 200,
            'body': response
        }

    except Exception as e:
        traceback.print_exc()
        return {
            'statusCode': 400,
            'body': str(e)
        }
    
def json_encode_hyperparameters(hyperparameters):
    for (k, v) in hyperparameters.items():
        print(k, v)

    return {str(k): json.dumps(v) for (k, v) in hyperparameters.items()}
