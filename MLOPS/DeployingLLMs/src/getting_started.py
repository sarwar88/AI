
import mlflow
import mlflow.pytorch
import numpy as np
import os

import os
import re
import math
from tqdm import tqdm
from huggingface_hub import login
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed
from peft import LoraConfig, PeftModel
from datetime import datetime
import mlflow
import mlflow.pytorch
# Different types of artifacts
import pandas as pd
import plotly.express as px

import mlflow
import mlflow.pytorch
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from mlflow.models.signature import infer_signature



from transformers import AutoModelForSeq2SeqLM

from huggingface_hub import login
login("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")

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# create new expriment 

experiment_id = mlflow.create_experiment("My first Experiment")

# create a new run 
with mlflow.start_run(experiment_id=experiment_id):
    # which part of your code will go for tracking 
    pass 

# create run with custome name 

run = mlflow.start_run(experiment_id=experiment_id,run_name="my first run")

# logging parameters 
mlflow.log_param("learning rate",0.01)
mlflow.log_param("batch size ",32)
num_epochs = 10
mlflow.log_param("num_epeochs",num_epochs)

# logging matrix 

for epoch in range(num_epochs) :
    mlflow.log_metric("accuracy",np.random.random(),step=epoch)
    mlflow.log_metric("loss ",np.random.random(),step=epoch)

# logging time series metrix 
for t in range(100):
    metrix_value = np.sin(t*np.pi/50)
    mlflow.log_metric("time_series_metric",metrix_value,step=t)

# Make sure the directory exists
os.makedirs("data", exist_ok=True)

# logging artifact 
with open("./data/dataset.csv" ,"w") as f :
    f.write("x,y\n")
    for x in range(100):
        f.write(f"{x},{x+1}\n")
mlflow.log_artifact("./data/dataset.csv","data")




confusion_matrix = np.random.randint(0, 100, size=(5,5))
labels = ["Cat", "Dog", "Rabbit", "Horse", "Pig"]
df_cm = pd.DataFrame(confusion_matrix, index=labels, columns=labels)

fig = px.imshow(df_cm, text_auto=True, labels=dict(x="Predicted", y="Actual"))
html_file = "confusion_matrix.html"
fig.write_html(html_file)
mlflow.log_artifact(html_file)



model = AutoModelForSeq2SeqLM.from_pretrained("TheFuzzyScientist/T5-base_Amazon-product-reviews")
mlflow.pytorch.log_model(model, name = "transformer_t5_model")

from transformers import AutoModelForSeq2SeqLM
from mlflow.models.signature import infer_signature
import torch

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained("TheFuzzyScientist/T5-base_Amazon-product-reviews")
model.to("cpu")



# Option 1: Install SentencePiece first (run this in a cell)
# !pip install sentencepiece

# Option 2: If you can't install SentencePiece, log without input example
# and define signature manually

from mlflow.models.signature import infer_signature
from mlflow.types.schema import Schema, TensorSpec
import numpy as np

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained("TheFuzzyScientist/T5-base_Amazon-product-reviews")

# Create a manual signature for T5 model (typical input format)
input_schema = Schema([
    TensorSpec(np.dtype(np.int64), (-1, -1), "input_ids"),
    TensorSpec(np.dtype(np.int64), (-1, -1), "attention_mask")
])
output_schema = Schema([
    TensorSpec(np.dtype(np.float32), (-1, -1, -1), "logits")
])

from mlflow.models.signature import ModelSignature
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Log the model with manual signature
mlflow.pytorch.log_model(
    pytorch_model=model,
    name="transformer_t5_model",
    signature=signature
)

mlflow.end_run()



