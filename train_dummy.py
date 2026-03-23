import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from datasets import Dataset

def main():
    print("Generating mock dataset representing Voxel51/PIDray inputs...")
    
    # Generate mock images and labels
    # Voxel51/PIDray usually features X-ray images. 
    # For training ViT, we need (3, 224, 224) standard or will be resized by processor
    np.random.seed(42)
    mock_data = {
        # Using PIL images directly by generating dummy pixels
        "image": [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)],
        "label": [np.random.randint(0, 2) for _ in range(16)]
    }
    from PIL import Image
    mock_data["image"] = [Image.fromarray(img) for img in mock_data["image"]]
    
    dataset = Dataset.from_dict(mock_data)
    
    # Train test split
    dataset = dataset.train_test_split(test_size=0.2)
    train_ds = dataset['train']
    val_ds = dataset['test']
    
    # Use standard ViT base (we can use a smaller one if needed, but vit-base-patch16 is fast enough to init)
    model_name = "google/vit-base-patch16-224-in21k"
    print("Loading processor and model...")
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
    
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    size = (processor.size["height"], processor.size["width"]) if "height" in processor.size else (224, 224)
        
    transforms = Compose([
        Resize(size),
        ToTensor(),
        normalize,
    ])
    
    def transforms_fn(examples):
        examples['pixel_values'] = [transforms(img.convert("RGB")) for img in examples['image']]
        del examples['image']
        return examples
        
    train_ds.set_transform(transforms_fn)
    val_ds.set_transform(transforms_fn)
    
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)
        
    output_dir = "./models/vit-pidray"
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=1,
        save_strategy="steps",
        save_steps=1,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        max_steps=2,
        warmup_ratio=0.1,
        logging_steps=1,
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
    print("Training pipeline finished successfully.")

if __name__ == "__main__":
    main()
