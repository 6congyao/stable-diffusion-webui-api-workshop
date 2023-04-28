#!/bin/bash
set -v
set -e
#

# This script should be run from the repo's backend directory
#
#
# Get s3uri and region from command line
s3uri=$1
region=$2

# Get reference for all important folders
backend_dir="$PWD"
project_dir="$backend_dir/.."
build_dist_dir="$backend_dir/build/codes"
source_dir="$backend_dir/src"
build_dir="$backend_dir/build/tmp"

echo "------------------------------------------------------------------------------"
echo "[Init] Clean old dist folders"
echo "------------------------------------------------------------------------------"
echo "rm -rf $build_dist_dir"
rm -rf $build_dist_dir
echo "mkdir -p $build_dist_dir"
mkdir -p $build_dist_dir

mkdir -p ${build_dir}/

echo "------------------------------------------------------------------------------"
echo "[Rebuild] all_in_one_ai_sagemaker lambda functions"
echo "------------------------------------------------------------------------------"

echo ${source_dir}
cd ${source_dir}/all_in_one_ai_sagemaker
rm -r ${build_dir}

mkdir -p ${build_dir}/python/

cp -R * ${build_dir}/python/
cd ${build_dir}
zip -r9 all_in_one_ai_sagemaker.zip python
cp ${build_dir}/all_in_one_ai_sagemaker.zip $build_dist_dir/all_in_one_ai_sagemaker.zip
rm ${build_dir}/all_in_one_ai_sagemaker.zip

echo "------------------------------------------------------------------------------"
echo "[Rebuild] other all_in_one_ai_* lambda functions"
echo "------------------------------------------------------------------------------"


lambda_folders="
all_in_one_ai_inference
all_in_one_ai_inference_post_process
all_in_one_ai_invoke_endpoint
"

for lambda_folder in $lambda_folders; do
    # build and copy console distribution files
    echo ${source_dir}
    cd ${source_dir}/${lambda_folder}
    echo ${build_dir}
    rm -r ${build_dir}

    mkdir -p ${build_dir}/
    cp -R * ${build_dir}/
    cd ${build_dir}
    zip -r9 ${lambda_folder}.zip .
    cp ${build_dir}/${lambda_folder}.zip $build_dist_dir/${lambda_folder}.zip
    rm ${build_dir}/${lambda_folder}.zip
done

aws s3 cp ${project_dir}/backend/build/codes ${s3uri}/codes --recursive --region ${region}
