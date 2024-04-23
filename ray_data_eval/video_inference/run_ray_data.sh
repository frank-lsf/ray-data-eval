python ray_data_pipeline_map.py --source local > video_inference_local_g5_xlarge_batch_32.out 2>&1
python ray_data_pipeline_map.py --source s3 > video_inference_s3_g5_xlarge_batch_32.out 2>&1
