import os
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def main():
    print("Loading Voxel51/PIDray dataset...")
    # Loading a small subset to ensure successful demonstration run
    dataset = load_dataset("Voxel51/PIDray", split="train[:10]")
    
    # Analyze map structure. We assume binary classification of prohibited items 
    # based on presence of bounding boxes or category ID.
    def extract_label(example):
        # The exact structure depends on how HF exposes Voxel51/PIDray.
        # usually `objects`, `bboxes`, or just `label`
        # Let's handle a general case where it could be either.
        if "label" in example:
            return {"label": example["label"]}
        
        objects = example.get('objects', {})
        if isinstance(objects, dict) and 'category' in objects:
            has_threat = int(len(objects['category']) > 0)
        elif isinstance(objects, list):
            has_threat = int(len(objects) > 0)
        else:
            # fallback
            has_threat = 0
            
        return {"label": has_threat}
        
    dataset = dataset.map(extract_label)
    
    # Basic train/test split
    dataset = dataset.train_test_split(test_size=0.2)
    train_ds = dataset['train']
    val_ds = dataset['test']
    
    model_name = "google/vit-base-patch16-224-in21k"
    processor = AutoImageProcessor.from_pretrained(model_name)
    
    id2label = {0: "Safe", 1: "Threat"}
    label2id = {"Safe": 0, "Threat": 1}
    
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    size = (
        processor.size["shortest_edge"]
        if "shortest_edge" in processor.size
        else (processor.size["height"], processor.size["width"])
    )
    if isinstance(size, int):
        size = (size, size)
        
    transforms = Compose([
        Resize(size),
        ToTensor(),
        normalize,
    ])
    
    def transforms_fn(examples):
        # apply transforms to PIL images
        examples['pixel_values'] = [transforms(img.convert("RGB")) for img in examples['image']]
        return examples
        
    train_ds.set_transform(transforms_fn)
    val_ds.set_transform(transforms_fn)
    
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)
        
    # Set output dir
    output_dir = "./models/vit-pidray"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        evaluation_strategy="steps",
        eval_steps=5,
        save_strategy="steps",
        save_steps=5,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        max_steps=5, # Limit for demo
        warmup_ratio=0.1,
        logging_steps=1,
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    final_output = "./models/final_model"
    print(f"Saving model to {final_output}")
    trainer.save_model(final_output)
    processor.save_pretrained(final_output)

if __name__ == "__main__":
    main()
