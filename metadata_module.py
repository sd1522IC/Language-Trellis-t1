class MetadataCSVDataSaver:
    def __init__(self, dataset_handler):
        self.dataset_handler = dataset_handler

    def load_metadata_json(self):
        """
        Loads the metadata.json file.
        """
        import json
        try:
            with open('metadata.json', 'r') as f:
                metadata_list = json.load(f)
            return metadata_list
        except FileNotFoundError:
            print("metadata.json file not found.")
            return None

    def save_metadata(
        self,
        your_dataset_toggle,
        our_datasets,
        sample_size,
        shuffle_data,
        sentiment_context,
        LDA_analysis,
        LDA_num_topics,
        version_name
    ):
        """
        Saves metadata to the CSV file.
        """
        # Read the existing CSV file
        df = self.dataset_handler.read_csv()

        # Prepare metadata entries
        metadata_entries = []

        # Sentiment Metadata
        if sentiment_context:
            metadata_entries.append(f"sentiment context: {sentiment_context}")

        # Dataset Metadata
        if your_dataset_toggle:
            metadata_entries.append("your_dataset: your_dataset")

        if our_datasets:
            try:
                dataset_config = self.dataset_handler.load_dataset_config(our_datasets)
                dataset_reference = dataset_config.get('dataset reference', '')
                if dataset_reference:
                    metadata_entries.append(f"our_datasets: {dataset_reference}")
                else:
                    print(f"Dataset reference not found for '{our_datasets}' in dataset_configs.json.")
            except ValueError:
                print(f"Dataset configuration for '{our_datasets}' not found.")
        elif not your_dataset_toggle:
            print("No valid dataset specified.")

        # Data Sample Size
        metadata_entries.append(f"background dataset size = {sample_size}")

        # Shuffled Data
        if shuffle_data:
            metadata_entries.append("shuffled data: shuffled background data")
        else:
            metadata_entries.append("shuffled data: constant background data")

        # LDA Metadata
        if LDA_analysis:
            metadata_entries.append(f"number of LDA topics: {LDA_num_topics}")

        # Add 'societally linear' information from metadata.json
        metadata_list = self.load_metadata_json()
        if metadata_list:
            societally_linear_value = "unknown"
            if version_name:
                # Search for the entry with matching 'version_name'
                for entry in metadata_list:
                    if entry.get('version_name') == version_name:
                        societally_linear_value = entry.get('societally linear', 'unknown')
                        break
                if societally_linear_value in ['yes', 'no']:
                    metadata_entries.append(f"societally linear: {societally_linear_value}")
                else:
                    metadata_entries.append("societally linear: unknown")
            else:
                print("No valid version_name specified for metadata lookup.")
        else:
            print("Metadata JSON could not be loaded.")

        # Determine the next available column name
        existing_columns = df.columns.tolist()
        metadata_column_name = "Metadata"
        i = 1
        while metadata_column_name in existing_columns:
            metadata_column_name = f"Metadata_{i}"
            i += 1

        # Pad the metadata list to match the length of the DataFrame
        metadata_list_column = metadata_entries + [''] * (len(df) - len(metadata_entries))
        df[metadata_column_name] = metadata_list_column

        # Write back to CSV
        self.dataset_handler.write_csv(df)
        print(f"Metadata saved to column '{metadata_column_name}' in '{self.dataset_handler.csv_file}'.")
