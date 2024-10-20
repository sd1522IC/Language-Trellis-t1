import concurrent.futures
from tqdm import tqdm 
import re
import emoji
import torch
from datasets import load_dataset, Dataset
import torch.nn.functional as F
import json
import os
import time
import random
import numpy as np
from transformers import (
    AlbertTokenizer,
    AlbertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    AlbertConfig
)
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, mean_absolute_error, r2_score

class ALBERTSentimentBaseProcessor:
    """
    Base class containing shared functionality for both fine-tuning and inference.
    """
    def __init__(self, max_seq_length=128, model_size='xlarge'):
        self.max_seq_length = max_seq_length
        self.model_size = model_size
        model_name = 'albert-xlarge-v2' if model_size == 'xlarge' else 'albert-base-v2'
        self.tokenizer = AlbertTokenizer.from_pretrained(model_name)
        self.model = None
        self.output_mode = None   
        self.num_labels = None     
        self.dataset_config = None 

    def clean_text(self, text, remove_mentions=True, remove_urls=True, segment_hashtags=True, replace_emojis=True):
        """
        Cleans text for sentiment analysis.
        """
        if remove_mentions:
            text = re.sub(r'@\w+', '', text)
        if remove_urls:
            text = re.sub(r'http\S+|www.\S+', '', text)
        if segment_hashtags:
            text = re.sub(r'#(\w+)', lambda x: ' '.join(re.findall(r'[A-Z][a-z]+|\w+', x.group(1))), text)
        if replace_emojis:
            text = emoji.demojize(text)

        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def tokenize_text(self, text):
        """
        Tokenizes text using the ALBERT tokenizer with fixed padding and truncation.
        """
        return self.tokenizer(
            text,
            truncation=True,          # Truncate sequences longer than max_length
            padding='max_length',     # Pad sequences shorter than max_length
            max_length=self.max_seq_length,  # The fixed maximum length for all sequences
            return_tensors='pt'
        )

    def parallel_tokenization(self, texts):
        """
        Tokenizes texts in parallel using concurrent futures.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            tokenized_texts = list(executor.map(self.tokenize_text, texts))
        return tokenized_texts

    def load_dataset_config(self, dataset_name):
        """
        Loads dataset-specific configurations from a JSON file.
        """
        with open('dataset_configs.json', 'r') as f:
            configs = json.load(f)
        if dataset_name in configs:
            self.dataset_config = configs[dataset_name]
            self.num_labels = self.dataset_config.get('num_labels')
            return self.dataset_config
        else:
            raise ValueError(f"Dataset configuration for '{dataset_name}' not found.")

    def initialize_model(self, sentiment_context=None):
        model_name = 'albert-xlarge-v2' if self.model_size == 'xlarge' else 'albert-base-v2'
        config = AlbertConfig.from_pretrained(model_name, num_labels=self.num_labels)

        if sentiment_context is not None:
            print(f"Loading fine-tuned weights for sentiment context '{sentiment_context}'")
            try:
                # Load weights_filepath and output_mode from metadata.json
                metadata_file_path = 'metadata.json'
                with open(metadata_file_path, 'r') as f:
                    metadata_list = json.load(f)

                # Find the entry with the matching sentiment_context
                metadata = next((item for item in metadata_list if item['version_name'] == sentiment_context), None)
                if metadata is None:
                    raise FileNotFoundError(f"No metadata found for sentiment context '{sentiment_context}'.")

                weights_filepath = metadata['weights_filepath']
                weights_dir = os.path.dirname(weights_filepath)
                if not os.path.isdir(weights_dir):
                    raise FileNotFoundError(f"The specified model directory '{weights_dir}' does not exist.")

                # Load the model from the specified directory
                self.output_mode = metadata.get('output_mode')
                if self.output_mode is None:
                    raise ValueError(f"'output_mode' not found in metadata for '{sentiment_context}'.")
                self.model = AlbertForSequenceClassification.from_pretrained(weights_dir)
                config_message = f"Running with fine-tuned weights for the {self.model_size} model, sentiment context '{sentiment_context}'."

            except Exception as e:
                print(f"Failed to load fine-tuned weights for sentiment context '{sentiment_context}'. Defaulting to pre-trained weights. Error: {e}")
                self.model = AlbertForSequenceClassification.from_pretrained(model_name, config=config)
                self.output_mode = self.num_labels  # Default to the dataset's num_labels
                config_message = f"Model initialized with {model_name}, configured for num_labels {self.num_labels}."

        else:
            self.model = AlbertForSequenceClassification.from_pretrained(model_name, config=config)
            self.output_mode = self.num_labels
            config_message = f"Model initialized with {model_name}, configured for num_labels {self.num_labels}."

        print(config_message)

class ALBERTSentimentFineTuner(ALBERTSentimentBaseProcessor):
    """
    Class for fine-tuning the ALBERT model.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_function = None
        self.dataset_name = None

    def prepare_finetuning_dataset(self, dataset_name='sst', split='train', sample_size=None, shuffle_data=True):
        """
        Loads, samples, cleans, tokenizes, and prepares the dataset for fine-tuning.
        Returns a dataset with input_ids, attention_mask, and labels.
        """
        self.dataset_name = dataset_name
        self.load_dataset_config(dataset_name)  # Load configuration from dataset_configs.json
        config = self.dataset_config

        # Load num_labels from dataset configuration (must be set correctly here)
        self.num_labels = config.get('num_labels')
        
        if self.num_labels is None:
            raise ValueError("num_labels is not set in the dataset configuration!")

        # Load the dataset from Hugging Face
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)

        # Handle sampling
        if sample_size:
            total_examples = len(dataset)
            if sample_size > total_examples:
                sample_size = total_examples

            if shuffle_data:
                indices = random.sample(range(total_examples), sample_size)
            else:
                indices = list(range(sample_size))

            dataset = dataset.select(indices)
        else:
            if shuffle_data:
                dynamic_seed = random.randint(1, 10000)  # Use dynamic seed for randomness
                dataset = dataset.shuffle(seed=dynamic_seed)

        text_field = config.get('text_field')
        label_field = config.get('label_field')

        # Clean the text
        texts = [self.clean_text(example[text_field]) for example in dataset]

        label_field = config.get('label_field')  # Fetch the field name where labels are stored
        label_mapping = config.get('label_mapping')  # Fetch the mapping from dataset config

        if label_mapping:
            labels = []
            for example in dataset:
                label = example[label_field]  # Access the actual label using label_field
                # Convert label to string and check if it's in the mapping
                if str(label) in label_mapping:
                    labels.append(label_mapping[str(label)])
                elif int(label) in label_mapping:
                    # If the label as an integer is in the mapping
                    labels.append(label_mapping[int(label)])
                else:
                    raise KeyError(f"Label {label} not found in label_mapping.")
        else:
            labels = [example[label_field] for example in dataset]  # Use label_field to access the labels directly

        tokenized_texts = self.parallel_tokenization(texts)

        # Prepare the final dataset
        final_dataset = Dataset.from_dict({
            'input_ids': [tokenized['input_ids'].squeeze(0) for tokenized in tokenized_texts],
            'attention_mask': [tokenized['attention_mask'].squeeze(0) for tokenized in tokenized_texts],
            'labels': labels
        })

        return final_dataset

    def compute_cross_entropy(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = F.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def compute_huber_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.get("logits").squeeze(-1)
        loss = F.smooth_l1_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

    class CustomTrainer(Trainer):
        def __init__(self, *args, loss_function=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_function = loss_function

        def compute_loss(self, model, inputs, return_outputs=False):
            return self.loss_function(model, inputs, return_outputs)

    def compute_classification_metrics(self, model, eval_dataset):
        """
        Computes classification metrics such as F1 score for the model on the eval_dataset.
        """
        trainer = Trainer(model=model)
        predictions = trainer.predict(eval_dataset)
        logits = predictions.predictions
        preds = np.argmax(logits, axis=-1)
        labels = np.array(eval_dataset['labels'])

        f1 = f1_score(labels, preds, average='weighted')
        accuracy = accuracy_score(labels, preds)

        metrics = {
            'f1_score': f1,
            'accuracy': accuracy
        }
        return metrics

    def compute_regression_metrics(self, model, eval_dataset):
        """
        Computes regression metrics such as MSE, MAE for the model on the eval_dataset.
        """
        trainer = Trainer(model=model)
        predictions = trainer.predict(eval_dataset)
        logits = predictions.predictions.squeeze(-1)
        labels = np.array(eval_dataset['labels'])

        mse = mean_squared_error(labels, logits)
        mae = mean_absolute_error(labels, logits)
        r2 = r2_score(labels, logits)

        metrics = {
            'mean_squared_error': mse,
            'mean_absolute_error': mae,
            'r2_score': r2
        }
        return metrics

    def get_loss_and_hyperparams(self):
        """
        Returns the appropriate loss function, hyperparameters, and evaluation metric based on the loss function.
        """
        if self.dataset_config is None:
            raise ValueError("Dataset configuration not loaded. Ensure 'load_dataset_config' is called before 'get_loss_and_hyperparams'.")

        # Get the loss function name and get the function from self
        loss_function_name = self.dataset_config.get('loss_function')
        if loss_function_name:
            loss_function = getattr(self, loss_function_name)
        else:
            raise ValueError("Loss function not specified in the dataset configuration.")

        # Get the evaluation metric name and get the function from self
        evaluation_metric_name = self.dataset_config.get('evaluation_metric')
        if evaluation_metric_name:
            evaluation_metric = getattr(self, evaluation_metric_name)
        else:
            raise ValueError("Evaluation metric not specified in the dataset configuration.")

        # Set hyperparameters based on the loss function
        if loss_function_name == 'compute_huber_loss':
            hyperparams = {
                'learning_rate': 1e-5,
                'batch_size': 8,
                'epochs': 5,
                'weight_decay': 0.05
            }
        elif loss_function_name == 'compute_cross_entropy':
            hyperparams = {
                'learning_rate': 1e-5,
                'batch_size': 8,
                'epochs': 10,
                'weight_decay': 0.02
            }
        else:
            # Default hyperparameters
            hyperparams = {
                'learning_rate': 1e-5,
                'batch_size': 8,
                'epochs': 3,
                'weight_decay': 0.01
            }

        return loss_function, hyperparams, evaluation_metric

    def fine_tune(self, dataset=None, start_from_sentiment_context=None, save_fine_tune='yes', fine_tune_version_name='', fine_tune_quality=True):
        """
        Fine-tunes the ALBERT model on the provided dataset. Optionally starts from a specific fine-tuned version.
        """
        if dataset is None or 'train' not in dataset or 'test' not in dataset:
            raise ValueError("A properly structured dataset with 'train' and 'test' splits must be provided for fine-tuning.")

        # Get loss function, hyperparameters, and evaluation metric
        loss_function, hyperparams, evaluation_metric = self.get_loss_and_hyperparams()
        self.loss_function = loss_function

        # Ensure self.num_labels is set before initializing the model
        if self.num_labels is None:
            raise ValueError("Number of labels (num_labels) not set. Please ensure that prepare_finetuning_dataset has been called and num_labels is set.")

        # Set up the training arguments using hyperparameters
        training_args = TrainingArguments(
            output_dir='./results',
            learning_rate=hyperparams['learning_rate'],
            per_device_train_batch_size=hyperparams['batch_size'],
            num_train_epochs=hyperparams['epochs'],
            weight_decay=hyperparams.get('weight_decay', 0.0),
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch"
        )

        # Dynamic padding with DataCollator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Use the CustomTrainer
        trainer = self.CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=data_collator,
            loss_function=self.loss_function  # Pass the loss function here
        )

        # Start training
        trainer.train()
        print("Completed fine-tuning on the dataset.")

        if save_fine_tune == 'yes':
            # Save the model with metadata after fine-tuning
            sample_size = len(dataset['train'])
            hyperparameters = {
                'learning_rate': hyperparams['learning_rate'],
                'batch_size': hyperparams['batch_size'],
                'epochs': hyperparams['epochs'],
            }
            if 'weight_decay' in hyperparams:
                hyperparameters['weight_decay'] = hyperparams['weight_decay']

            model_type = 'albert-xlarge-v2' if self.model_size == 'xlarge' else 'albert-base-v2'

            self.save_model_with_metadata(
                model=self.model,
                sample_size=sample_size,
                hyperparameters=hyperparameters,
                dataset_name=self.dataset_name,
                model_type=model_type,
                fine_tune_version_name=fine_tune_version_name,
                fine_tune_quality=fine_tune_quality,
                eval_dataset=dataset['test'],
                evaluation_metric=evaluation_metric
            )

    def save_model_with_metadata(self, model, sample_size, hyperparameters, dataset_name, model_type, fine_tune_version_name='', fine_tune_quality=True, eval_dataset=None, evaluation_metric=None):
        """
        Saves the fine-tuned model along with associated metadata, including continuous_label or class_labels.
        """
        # Determine the version name
        if fine_tune_version_name:
            version_name = fine_tune_version_name
        else:
            version_name = f"version_{int(time.time())}"  # Generate a unique version name based on the current timestamp
        print(f"Auto-saving model with version name: {version_name}")

        save_directory = os.path.join('model_versions', version_name)
        os.makedirs(save_directory, exist_ok=True)

        model.save_pretrained(save_directory)

        # Compute quality metrics if requested
        if fine_tune_quality and eval_dataset is not None and evaluation_metric is not None:
            metrics = evaluation_metric(model, eval_dataset)
        else:
            metrics = {}

        # Create the metadata dictionary
        metadata = {
            'version_name': version_name,
            'sample_size': sample_size,
            'dataset': dataset_name,
            'model_type': model_type,
            'weights_filepath': os.path.join(save_directory, 'model.safetensors'),
            'output_mode': self.num_labels  # Use self.num_labels
        }

        # Add continuous_label or class_labels based on output_mode
        if self.num_labels == 1:
            # For continuous output mode
            metadata['continuous_label'] = "Sentiment"  # You can modify this to accept a custom value if needed
        else:
            # For categorical output mode
            metadata['class_labels'] = [f"Class_{i}" for i in range(self.num_labels)]  # Default class labels (can be modified)

        # Update metadata with hyperparameters and metrics
        metadata.update(hyperparameters)
        metadata.update(metrics)

        # Load existing metadata
        metadata_file_path = 'metadata.json'
        if os.path.exists(metadata_file_path):
            with open(metadata_file_path, 'r') as f:
                existing_metadata = json.load(f)
        else:
            existing_metadata = []

        # Append the new metadata entry
        existing_metadata.append(metadata)

        # Save all metadata to the centralized JSON file
        with open(metadata_file_path, 'w') as f:
            json.dump(existing_metadata, f, indent=4)

        print(f"Model and metadata for version '{version_name}' saved successfully.")

class ALBERTSentimentInferencer(ALBERTSentimentBaseProcessor):
    def run_inference(self, texts, sentiment_context=None, dataset_name=None, sentiment_mode='classic'):
        """
        Runs inference on a list of texts using the pre-initialized ALBERT model.
        Handles different sentiment modes: 'longform', 'snapshot', or 'classic'.
        """
        # Only load the dataset configuration if a valid dataset_name is provided
        if dataset_name:
            self.load_dataset_config(dataset_name)
        
        if self.model is None:
            raise ValueError("Model is not initialized. Ensure that the model has been initialized before calling `run_inference`.")

        cleaned_texts = [self.clean_text(text) for text in texts]

        total_chunks = self._calculate_total_chunks(cleaned_texts, sentiment_mode)

        document_predictions = []
        with tqdm(total=total_chunks, desc="Running Inference", unit="chunk") as pbar:
            for text in cleaned_texts:
                if sentiment_mode == 'longform':
                    tokenized_chunks = self._tokenize_and_chunk_text(text, sentiment_mode='true')
                    predictions = self._run_inference_on_chunks(tokenized_chunks)
                    pbar.update(len(tokenized_chunks))
                elif sentiment_mode == 'snapshot':
                    predictions = self.snapshot_sentiment(text)
                    pbar.update(2)
                else:  # Default classic mode
                    tokenized_chunk = self.tokenize_text(text)
                    predictions = self._run_inference_on_chunks([tokenized_chunk])
                    pbar.update(1)

                document_predictions.append(predictions)

        # Return predictions with information about the model type
        model_type = self.model_size
        fine_tuning_info = sentiment_context if sentiment_context else "pre-trained"
        print(f"Inference completed using {model_type} model on {dataset_name or 'your dataset'} dataset with {fine_tuning_info} weights.")

        return document_predictions


    def _initialize_with_fine_tuned_weights(self, sentiment_context):
        """
        Tries to load fine-tuned weights. If it fails, falls back to pre-trained weights.
        """
        try:
            self.initialize_model(sentiment_context=sentiment_context)
        except Exception as e:
            print(f"Failed to load fine-tuned weights for sentiment context '{sentiment_context}'. Defaulting to pre-trained weights. Error: {e}")
            # Load pre-trained model if fine-tuned weights fail
            self.initialize_model()

    def _calculate_total_chunks(self, cleaned_texts, sentiment_mode):
        """
        Calculates the total number of chunks or segments for the progress bar based on the sentiment mode.
        """
        total_chunks = 0
        for text in cleaned_texts:
            tokens = self.tokenizer.tokenize(text)
            if sentiment_mode == 'longform' and len(tokens) > self.max_seq_length:
                num_chunks = (len(tokens) + self.max_seq_length - 1) // self.max_seq_length
            elif sentiment_mode == 'snapshot':
                if len(tokens) > self.max_seq_length:
                    num_chunks = 2  # Process head and tail for long texts
                else:
                    num_chunks = max(4, 1)
            else:
                num_chunks = 1
            total_chunks += num_chunks
        return total_chunks

    def snapshot_sentiment(self, text):
        """
        Handles the 'snapshot' sentiment approach where the first half and last half
        of the text are sampled and averaged to compute the final sentiment.
        """
        tokens = self.tokenizer.tokenize(text)

        if len(tokens) > self.max_seq_length:
            half_seq_len = self.max_seq_length // 2
            # Take the first half and the last half of the tokens
            first_half = tokens[:half_seq_len]
            last_half = tokens[-half_seq_len:]
            sampled_text = [self.tokenizer.convert_tokens_to_string(first_half), self.tokenizer.convert_tokens_to_string(last_half)]
        else:
            # If the text is short enough, treat it as a single chunk
            sampled_text = [text]

        tokenized_chunks = [self.tokenize_text(chunk) for chunk in sampled_text]
        predictions = self._run_inference_on_chunks(tokenized_chunks)
        return predictions

    def _tokenize_and_chunk_text(self, text, sentiment_mode):
        """
        Tokenizes text and splits it into chunks if necessary using inherited tokenize_text method.
        """
        tokens = self.tokenizer.tokenize(text)
        tokenized_chunks = []

        if sentiment_mode == 'true' and len(tokens) > self.max_seq_length:
            chunks = [tokens[i:i + self.max_seq_length] for i in range(0, len(tokens), self.max_seq_length)]
            for chunk in chunks:
                tokenized_chunk = self.tokenize_text(self.tokenizer.convert_tokens_to_string(chunk))  # Reuse inherited tokenize_text
                tokenized_chunks.append(tokenized_chunk)
        else:
            tokenized_chunk = self.tokenize_text(text)  # Reuse inherited tokenize_text
            tokenized_chunks.append(tokenized_chunk)

        return tokenized_chunks

    def _run_inference_on_chunks(self, tokenized_chunks):
        """
        Runs inference on tokenized chunks and returns predictions.
        """

        # Prepare the dataset for inference
        inference_dataset = Dataset.from_dict({
            'input_ids': [tokenized['input_ids'].squeeze(0) for tokenized in tokenized_chunks],
            'attention_mask': [tokenized['attention_mask'].squeeze(0) for tokenized in tokenized_chunks],
        })

        # Initialize Trainer once
        trainer = Trainer(model=self.model, args=TrainingArguments(output_dir="./", disable_tqdm=True))

        # Run inference once for the entire dataset
        predictions = trainer.predict(inference_dataset)

        if predictions.predictions is not None:
            predictions_tensor = torch.tensor(predictions.predictions)
            return self._process_predictions(predictions_tensor)
        else:
            return None

    def _process_predictions(self, predictions_tensor):
        """
        Processes predictions depending on the model's output mode.
        """
        if self.output_mode == 1:
            # Continuous output, no softmax
            document_mean = predictions_tensor.mean(dim=0).detach().cpu().numpy().item()
            return document_mean
        else:
            # Categorical output, apply softmax to each chunk
            softmax_predictions = F.softmax(predictions_tensor, dim=-1)

            # Take the mean across all chunks for each class
            mean_probabilities = softmax_predictions.mean(dim=0)

            # No need to apply softmax again, just return the mean probabilities
            final_probability_distribution = mean_probabilities.detach().cpu().numpy().tolist()
            return final_probability_distribution

class SentimentCSVDataSaver:
    def __init__(self, dataset_handler, sentiment_context=None):
        """
        Initializes the SentimentCSVDataSaver with a reference to DatasetHandler and sentiment context.
        Retrieves and stores the output_mode, class labels, and continuous label based on the sentiment context.
        """
        self.dataset_handler = dataset_handler
        self.sentiment_context = sentiment_context
        self.output_mode, self.class_labels, self.continuous_label = self.get_metadata_info()

    def get_metadata_info(self):
        """
        Retrieves the output_mode, class labels, and continuous label from the metadata JSON based on the sentiment_context.
        """
        if self.sentiment_context is None:
            return self.dataset_handler.num_labels, None, None

        # Load metadata
        metadata_file_path = 'metadata.json'
        with open(metadata_file_path, 'r') as f:
            metadata_list = json.load(f)

        # Find the entry with the matching sentiment_context
        metadata = next((item for item in metadata_list if item['version_name'] == self.sentiment_context), None)
        if metadata is None:
            raise FileNotFoundError(f"No metadata found for sentiment context '{self.sentiment_context}'.")

        output_mode = metadata.get('output_mode')
        class_labels = metadata.get('class_labels', None)  # Get class labels if they exist
        continuous_label = metadata.get('continuous_label', 'Sentiment')  # Default to 'Sentiment' for continuous
        return output_mode, class_labels, continuous_label

    def save_results(self, predictions):
        """
        Saves sentiment analysis results to the CSV file based on the stored output_mode, class labels, and continuous label.
        """
        # Read the existing CSV file
        df = self.dataset_handler.read_csv()

        # Add analysis results based on the output_mode
        if self.output_mode == 1:
            # For continuous output, use "Continuous:" followed by the custom label from metadata
            continuous_header = f"continuous: {self.continuous_label}"
            df[continuous_header] = predictions
        else:
            # Dynamically generate column headers based on the number of classes
            num_labels = self.output_mode

            # Use custom class labels if provided, otherwise default to 'Class_X'
            labels = self.class_labels if self.class_labels else [f'Class_{i}' for i in range(num_labels)]

            probabilities = predictions
            for i, label in enumerate(labels):
                # Prefix with "Probability Distribution {i}:" followed by custom label
                column_header = f"Probability Distribution {i+1}: {label}"
                df[column_header] = [prob[i] for prob in probabilities]  # Create columns dynamically

        # Write back to CSV
        self.dataset_handler.write_csv(df)
