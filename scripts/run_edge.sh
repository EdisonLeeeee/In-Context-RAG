#!/bin/bash

# Configuration parameters
topk_values=(3)  # Size of top-k
datasets=('Enron' 'Googlemap_CT' 'Stack_elec' 'Stack_ubuntu' 'Amazon_movies' 'Yelp')  # List of datasets
gpu_ids="3"  # Specified GPU IDs, separated by commas
models=("Meta-Llama-3.1-8B-Instruct" "Qwen2.5-7B-Instruct" "Mistral-7B-Instruct-v0.3" "Phi-3.5-mini-instruct" "gemma-2-9b-it"  )  # List of models
NPROC_PER_NODE=$(echo $gpu_ids | awk -F',' '{print NF}')

# Create results folder
output_base_dir="edge_results"
input_dir="edge_prompts"

# The vllm backend requires vllm installation, but pt does not
infer_backend="pt" # or vllm

# Whether to allow overwriting existing output files
override=False  # If set to True, files will be overwritten; if False, existing files will be skipped

# Iterate over each model
for model_name in "${models[@]}"; do
    # Set the path for each model
    model_path="../../$model_name" # Path where the model is stored
    output_dir="${output_base_dir}/${model_name}"
    mkdir -p "${output_dir}"  # Create an output folder for each model

    # Iterate over datasets and variants
    for dataset in "${datasets[@]}"; do
        for topk in "${topk_values[@]}"; do
            # Iterate over different prompt inputs
            file_names=(
                "${dataset}_raw.jsonl"
                "${dataset}_label_rag.jsonl"
                "${dataset}_query_rag.jsonl"
                "${dataset}_few_shot_rag.jsonl"
            )
            for file_name in "${file_names[@]}"; do
                # Input file name
                input_file="${input_dir}/${file_name}"
                
                # Output file name and path
                output_file="${output_dir}/${file_name}"
                
                # Check if the output file already exists
                if [ -f "$output_file" ]; then
                    if [ "$override" == "True" ]; then
                        echo "Output file ${output_file} exists. Overriding..."
                        rm "$output_file"  # Delete the existing file
                    else
                        echo "Output file ${output_file} exists and override is False. Skipping..."
                        continue  # Skip the current file and move to the next one
                    fi
                fi

                # Run the command
                echo "============================================================================================================="
                echo "Inference on dataset: ${file_name} with topk=${topk}, model=${model_name}, using GPUs=${gpu_ids}"
                echo "============================================================================================================="
                CUDA_VISIBLE_DEVICES=$gpu_ids NPROC_PER_NODE=$NPROC_PER_NODE swift infer \
                    --model $model_path \
                    --infer_backend $infer_backend \
                    --model_revision master \
                    --torch_dtype bfloat16 \
                    --attn_impl flash_attn \
                    --max_new_tokens 100 \
                    --temperature 0.0 \
                    --top_p 0.7 \
                    --repetition_penalty 1.05 \
                    --result_path $output_file \
                    --val_dataset $input_file
            done
        done
    done
done

echo "Inference complete! Results saved in ${output_base_dir}"