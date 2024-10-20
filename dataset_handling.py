import os
import random
import pandas as pd
import json
import random
from datasets import load_dataset, Dataset

class DatasetHandler:
    def __init__(self, csv_file='output.csv'):
        self.csv_file = csv_file

    def initialize_csv(self, our_datasets=None, your_dataset_toggle=False, sample_size=50, shuffle_data=True, split='test'):
        """
        Initializes the CSV file with uncleaned texts, time fields, and source fields.
        """
        # Initialize lists to collect data
        uncleaned_texts_list = []
        time_fields_list = []
        source_fields_list = []

        # Load data from your dataset first
        if your_dataset_toggle:
            # Load dataset configuration for your_dataset
            dataset_config = self.load_dataset_config('your_dataset')

            data_source = dataset_config.get('data_source', 'local_csv')
            csv_file_name = dataset_config.get('csv_file', 'your_dataset.csv')

            # Load local CSV file
            csv_file_path = os.path.join(os.path.dirname(__file__), csv_file_name)
            if not os.path.exists(csv_file_path):
                raise FileNotFoundError(f"Your dataset CSV file '{csv_file_path}' not found.")
            df = pd.read_csv(csv_file_path)

            # Do not perform sampling or shuffling on your_dataset

            # Get the text column name from dataset configuration
            text_field = dataset_config.get('text_field', 'Text')
            if text_field not in df.columns:
                raise KeyError(f"The text field '{text_field}' specified in dataset_configs.json does not exist in your dataset.")

            uncleaned_texts_list.extend(df[text_field].astype(str).tolist())

            # Time fields
            time_field = dataset_config.get('time_field', None)
            if time_field and time_field in df.columns:
                time_fields_list.extend(df[time_field].astype(str).tolist())
            else:
                time_fields_list.extend([''] * len(df))

            # Source fields
            source_field = dataset_config.get('source_field', None)
            default_source_value = dataset_config.get('default_source_value', 'your_dataset')

            if source_field and source_field in df.columns:
                source_fields_list.extend(df[source_field].astype(str).tolist())
            else:
                source_fields_list.extend([default_source_value] * len(df))

        # Load data from our datasets after
        if our_datasets:
            dataset_config = self.load_dataset_config(our_datasets)
            data_source = dataset_config.get('data_source', 'huggingface')

            if data_source == 'local_csv':
                # Load local CSV file
                csv_file_path = os.path.join(os.path.dirname(__file__), dataset_config.get('csv_file', ''))
                if not os.path.exists(csv_file_path):
                    raise FileNotFoundError(f"CSV file '{csv_file_path}' not found.")
                df = pd.read_csv(csv_file_path)
                dataset = Dataset.from_pandas(df)
            else:
                # Load the dataset from Hugging Face Datasets
                dataset = load_dataset(our_datasets, split=split, trust_remote_code=True)

            # Handle sampling and shuffling
            dataset = self.sample_dataset(dataset, sample_size, shuffle_data)

            text_field = dataset_config.get('text_field')
            time_field = dataset_config.get('time_field')
            source_field = dataset_config.get('source_field')
            default_source_value = dataset_config.get('default_source_value', our_datasets)

            for example in dataset:
                uncleaned_texts_list.append(example[text_field])
                time_fields_list.append(example.get(time_field, ''))

                if source_field and source_field in example:
                    source_fields_list.append(example[source_field])
                else:
                    source_fields_list.append(default_source_value)

        # Create the initial DataFrame
        data = {
            'Text': uncleaned_texts_list,
            'Source': source_fields_list,
            'Time': time_fields_list
        }
        initial_df = pd.DataFrame(data)

        # Save the initial CSV file
        initial_df.to_csv(self.csv_file, index=False)
        print(f"Initialized CSV file '{self.csv_file}' with data from the datasets.")

    def load_dataset_config(self, dataset_name):
        """
        Loads dataset-specific configurations from a JSON file.
        """
        with open('dataset_configs.json', 'r') as f:
            configs = json.load(f)
        if dataset_name in configs:
            return configs[dataset_name]
        else:
            raise ValueError(f"Dataset configuration for '{dataset_name}' not found.")

    def sample_dataset(self, dataset, sample_size, shuffle_data):
        # Print statement showing that the sampling process has started
        print(f"Sampling {sample_size} data points from dataset...")

        total_examples = len(dataset)
        
        # Ensure sample size does not exceed the total available examples
        if sample_size and sample_size > total_examples:
            sample_size = total_examples

        if shuffle_data:
            # Dynamically set the seed to ensure randomness in each run
            dynamic_seed = random.randint(1, 10000)  # Random seed for each run
            dataset = dataset.shuffle(seed=dynamic_seed)
        
        if sample_size:
            # Select only the 'sample_size' examples (after shuffling if applicable)
            dataset = dataset.select(range(sample_size))

        print(f"Sampled {sample_size} data points from dataset.")

        return dataset

    def read_csv(self):
        """
        Reads the CSV file into a DataFrame.
        """
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file '{self.csv_file}' not found.")
        df = pd.read_csv(self.csv_file)
        return df

    def write_csv(self, df):
        """
        Writes the DataFrame back to the CSV file.
        """
        df.to_csv(self.csv_file, index=False)
        print(f"Updated CSV file '{self.csv_file}' with new analysis results.")

    def get_texts_for_analysis(self):
        """
        Returns the texts from the CSV file for analysis.
        """
        df = self.read_csv()
        texts = df['Text'].astype(str).tolist()
        return texts

    def update_csv_with_results(self, results_dict):
        """
        Updates the CSV file with new analysis results.
        """
        df = self.read_csv()
        for column_name, results in results_dict.items():
            df[column_name] = results
        self.write_csv(df)
