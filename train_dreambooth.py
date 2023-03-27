import os
import argparse
import json
import shutil
import traceback
import boto3
from botocore.exceptions import ClientError
import glob

def upload_s3files(s3uri, file_path_with_pattern):
    pos = s3uri.find('/', 5)
    bucket = s3uri[5 : pos]
    key = s3uri[pos + 1 : ]

    s3_resource = boto3.resource('s3')
    s3_bucket = s3_resource.Bucket(bucket)

    try:
        for file_path in glob.glob(file_path_with_pattern):
            file_name = os.path.basename(file_path)
            __s3file = f'{key}{file_name}'
            print(file_path, __s3file)
            s3_bucket.upload_file(file_path, __s3file)
    except ClientError as e:
        print(e)
        return False
    return True

def upload_s3folder(s3uri, file_path):
    pos = s3uri.find('/', 5)
    bucket = s3uri[5 : pos]
    key = s3uri[pos + 1 : ]

    s3_resource = boto3.resource('s3')
    s3_bucket = s3_resource.Bucket(bucket)

    try:
        for path, _, files in os.walk(file_path):
            for file in files:
                dest_path = path.replace(file_path,"")
                __s3file = f'{key}{dest_path}/{file}'
                __local_file = os.path.join(path, file)
                print(__local_file, __s3file)
                s3_bucket.upload_file(__local_file, __s3file)
    except Exception as e:
        print(e)

parser = argparse.ArgumentParser(description='Process dreambooth training.')
parser.add_argument('--train-args', type=str, help='Train arguments')
parser.add_argument('--sd-models-s3uri', default='', type=str, help='SD Models S3Uri')
parser.add_argument('--db-models-s3uri', default='', type=str, help='DB Models S3Uri')
parser.add_argument('--lora-models-s3uri', default='', type=str, help='Lora Models S3Uri')
parser.add_argument('--dreambooth-config-id', default='', type=str, help='Dreambooth config ID')
parser.add_argument("--dreambooth-models-path", type=str, help="Path to directory to store Dreambooth model file(s).", default=None)
parser.add_argument("--lora-models-path", type=str, help="Path to directory to store Lora model file(s).", default=None)

args = parser.parse_args()

from extensions.sd_dreambooth_extension.dreambooth.db_config import DreamboothConfig
from extensions.sd_dreambooth_extension.scripts.dreambooth import start_training_from_config, create_model
from extensions.sd_dreambooth_extension.scripts.dreambooth import performance_wizard, training_wizard
from extensions.sd_dreambooth_extension.dreambooth.db_concept import Concept

train_args = json.loads(args.train_args)
db_create_new_db_model = train_args['train_dreambooth_settings']['db_create_new_db_model']
db_use_txt2img = train_args['train_dreambooth_settings']['db_use_txt2img']
db_train_wizard_person = train_args['train_dreambooth_settings']['db_train_wizard_person']
db_train_wizard_object = train_args['train_dreambooth_settings']['db_train_wizard_object']
db_performance_wizard = train_args['train_dreambooth_settings']['db_performance_wizard']

if db_create_new_db_model:
    db_new_model_name = train_args['train_dreambooth_settings']['db_new_model_name']
    db_new_model_src = train_args['train_dreambooth_settings']['db_new_model_src']
    db_new_model_scheduler = train_args['train_dreambooth_settings']['db_new_model_scheduler']
    db_create_from_hub = train_args['train_dreambooth_settings']['db_create_from_hub']
    db_new_model_url = train_args['train_dreambooth_settings']['db_new_model_url']
    db_new_model_token = train_args['train_dreambooth_settings']['db_new_model_token']
    db_new_model_extract_ema = train_args['train_dreambooth_settings']['db_new_model_extract_ema']
    db_train_unfrozen = train_args['train_dreambooth_settings']['db_train_unfrozen']
    db_512_model = train_args['train_dreambooth_settings']['db_512_model']
    db_save_safetensors = train_args['train_dreambooth_settings']['db_save_safetensors']

    db_model_name, db_model_path, db_revision, db_epochs, db_scheduler, db_src, db_has_ema, db_v2, db_resolution = create_model(
        db_new_model_name,
        db_new_model_src,
        db_new_model_scheduler,
        db_create_from_hub,
        db_new_model_url,
        db_new_model_token,
        db_new_model_extract_ema,
        db_train_unfrozen,
        db_512_model
    )
    dreambooth_config_id = args.dreambooth_config_id
    try:
        with open(f'/opt/ml/input/data/config/{dreambooth_config_id}.json', 'r') as f:
            content = f.read()
    except Exception:
        content = None

    if content:
        params_dict = json.loads(content)

        params_dict['db_model_name'] = db_model_name
        params_dict['db_model_path'] = db_model_path
        params_dict['db_revision'] = db_revision
        params_dict['db_epochs'] = db_epochs
        params_dict['db_scheduler'] = db_scheduler
        params_dict['db_src'] = db_src
        params_dict['db_has_ema'] = db_has_ema
        params_dict['db_v2'] = db_v2
        params_dict['db_resolution'] = db_resolution

        if db_train_wizard_person or db_train_wizard_object:
            db_num_train_epochs, \
            c1_num_class_images_per, \
            c2_num_class_images_per, \
            c3_num_class_images_per, \
            c4_num_class_images_per = training_wizard(db_train_wizard_person if db_train_wizard_person else db_train_wizard_object)

            params_dict['db_num_train_epochs'] = db_num_train_epochs
            params_dict['c1_num_class_images_per'] = c1_num_class_images_per
            params_dict['c1_num_class_images_per'] = c2_num_class_images_per
            params_dict['c1_num_class_images_per'] = c3_num_class_images_per
            params_dict['c1_num_class_images_per'] = c4_num_class_images_per
        if db_performance_wizard:
            attention, \
            gradient_checkpointing, \
            gradient_accumulation_steps, \
            mixed_precision, \
            cache_latents, \
            sample_batch_size, \
            train_batch_size, \
            stop_text_encoder, \
            use_8bit_adam, \
            use_lora, \
            use_ema, \
            save_samples_every, \
            save_weights_every = performance_wizard()

            params_dict['attention'] = attention
            params_dict['gradient_checkpointing'] = gradient_checkpointing
            params_dict['gradient_accumulation_steps'] = gradient_accumulation_steps
            params_dict['mixed_precision'] = mixed_precision
            params_dict['cache_latents'] = cache_latents
            params_dict['sample_batch_size'] = sample_batch_size
            params_dict['train_batch_size'] = train_batch_size
            params_dict['stop_text_encoder'] = stop_text_encoder
            params_dict['use_8bit_adam'] = use_8bit_adam
            params_dict['use_lora'] = use_lora
            params_dict['use_ema'] = use_ema
            params_dict['save_samples_every'] = save_samples_every
            params_dict['params_dict'] = save_weights_every

        db_config = DreamboothConfig(db_model_name)
        concept_keys = ["c1_", "c2_", "c3_", "c4_"]
        concepts_list = []
        # If using a concepts file/string, keep concepts_list empty.
        if params_dict["db_use_concepts"] and params_dict["db_concepts_path"]:
            concepts_list = []
            params_dict["concepts_list"] = concepts_list
        else:
            for concept_key in concept_keys:
                concept_dict = {}
                for key, param in params_dict.items():
                    if concept_key in key and param is not None:
                        concept_dict[key.replace(concept_key, "")] = param
                concept_test = Concept(concept_dict)
                if concept_test.is_valid:
                    concepts_list.append(concept_test.__dict__)
            existing_concepts = params_dict["concepts_list"] if "concepts_list" in params_dict else []
            if len(concepts_list) and not len(existing_concepts):
                params_dict["concepts_list"] = concepts_list

        db_config.load_params(params_dict)
else:
    db_model_name = train_args['train_dreambooth_settings']['db_model_name']
    db_config = DreamboothConfig(db_model_name)

print(vars(db_config))
start_training_from_config(
    db_config,
    db_use_txt2img,
    False
)

cmd_sd_models_path = args.ckpt_dir
sd_models_dir = cmd_sd_models_path

try:
    cmd_dreambooth_models_path = args.dreambooth_models_path
except:
    cmd_dreambooth_models_path = None

try:
    cmd_lora_models_path = args.lora_models_path
except:
    cmd_lora_models_path = None

db_model_dir = os.path.dirname(cmd_dreambooth_models_path)
db_model_dir = os.path.join(db_model_dir, "dreambooth")

lora_model_dir = os.path.dirname(cmd_lora_models_path)
lora_model_dir = os.path.join(lora_model_dir, "lora")

print('---models path---', sd_models_dir, lora_model_dir)
os.system(f'ls -l {sd_models_dir}')
os.system('ls -l {0}'.format(os.path.join(sd_models_dir, db_model_name)))
os.system(f'ls -l {lora_model_dir}')

sd_models_s3uri = args.sd_models_s3uri
db_models_s3uri = args.db_models_s3uri
lora_models_s3uri = args.lora_models_s3uri

try:
    print('Uploading SD Models...')
    upload_s3files(
        args.sd_models_s3uri,
        os.path.join(sd_models_dir, db_model_name, f'{db_model_name}_*.yaml')
    )
    if db_save_safetensors:
        upload_s3files(
            sd_models_s3uri,
            os.path.join(sd_models_dir, db_model_name, f'{db_model_name}_*.safetensors')
        )
    else:
        upload_s3files(
            sd_models_s3uri,
            os.path.join(sd_models_dir, db_model_name, f'{db_model_name}_*.ckpt')
        )
    print('Uploading DB Models...')
    upload_s3folder(
        f'{db_models_s3uri}{db_model_name}',
        os.path.join(db_model_dir, db_model_name)
    )
    if db_config.use_lora:
        print('Uploading Lora Models...')
        upload_s3files(
            lora_models_s3uri,
            os.path.join(lora_model_dir, f'{db_model_name}_*.pt')
        )
    #automatic tar latest checkpoint and upload to s3 by zheng on 2023.03.22
    os.makedirs(os.path.dirname("/opt/ml/model/"), exist_ok=True)
    train_steps=int(db_config.revision)
    model_file_basename = f'{db_model_name}_{train_steps}_lora' if db_config.use_lora else f'{db_model_name}_{train_steps}'
    f1=os.path.join(sd_models_dir, db_model_name, f'{model_file_basename}.yaml')
    if os.path.exists(f1):
        shutil.copy(f1,"/opt/ml/model/")
    if db_save_safetensors:
        f2=os.path.join(sd_models_dir, db_model_name, f'{model_file_basename}.safetensors')
        if os.path.exists(f2):
            shutil.copy(f2,"/opt/ml/model/")
    else:
        f2=os.path.join(sd_models_dir, db_model_name, f'{model_file_basename}.ckpt')
        if os.path.exists(f2):
            shutil.copy(f2,"/opt/ml/model/")
except Exception as e:
    traceback.print_exc()
    print(e)
