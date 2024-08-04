import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore", message="A parameter name that contains")


# Load the data from the CSV file
data = pd.read_csv('dataset.csv')

# Define the texts (tweets) and labels (0 = Non-antisemitic, 1 = Antisemitic) from the dataset
texts = data['Text'].tolist()
labels = data['Biased'].tolist()

# Tokenize the texts using the BERT tokenizer and encode the labels as integers 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

# Create a Dataset object
dataset = Dataset.from_dict({'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask'], 'labels': labels})

# Split the data into training and testing sets (80% training, 20% testing) 
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Define the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_steps=10,
    eval_strategy='epoch',
    save_strategy='epoch'
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda p: {
        'accuracy': accuracy_score(p.label_ids, p.predictions.argmax(-1)),
        'precision': precision_score(p.label_ids, p.predictions.argmax(-1)),
        'recall': recall_score(p.label_ids, p.predictions.argmax(-1)),
        'f1': f1_score(p.label_ids, p.predictions.argmax(-1))
    }
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Function to classify a new text using the BERT model
def classify_text_bert(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the correct device
    with torch.no_grad():  # No need to track gradients for inference
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    label = 'Antisemitic' if prediction == 1 else 'Non-antisemitic'
    return label

# Continuously prompt the user to enter text for classification
while True:
    user_input = input("Enter a text to classify (or type 'exit' to stop): ")
    if user_input.lower() == 'exit':
        break
    # Get prediction from the BERT model
    result_bert = classify_text_bert(user_input)

    print("-----------------")
    print(f"Text: {user_input} | BERT Classification: {result_bert}")
