from datasets import load_dataset
# Inspect first item structure via streaming
dataset = load_dataset("Voxel51/PIDray", streaming=True)
train_iter = iter(dataset['train'])
first_item = next(train_iter)
print("Features:", dataset['train'].features)
print("First item keys:", first_item.keys())
print("First item objects/labels:", [(k, v) for k, v in first_item.items() if k != 'image'])
