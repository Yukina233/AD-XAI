from adbench_modified.run import RunPipeline
from adbench.baseline.DeepSAD.src.run import DeepSAD
# return the results including [params, model_name, metrics, time_fit, time_inference]
# besides, results will be automatically saved in the dataframe and ouputted as csv file in adbench/result folder

pipeline = RunPipeline(suffix='DeepSAD', parallel='semi-supervise', realistic_synthetic_mode=None, noise_type=None)
results = pipeline.run(clf=DeepSAD)