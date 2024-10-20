print("Talking to analysis engines...")
from dataset_handling import DatasetHandler
from alb_s1 import ALBERTSentimentFineTuner, ALBERTSentimentInferencer, SentimentCSVDataSaver
from lda_module import LDATextProcessor, LDAProcessor, LDACSVDataSaver
from bertopic_module import BERTopicProcessor, BERTopicCSVDataSaver
from kcluster_module import TextProcessingAndDatabase, KclusterAnalysis, KclusterCSVDataSaver
from metadata_module import MetadataCSVDataSaver
from Data_Visualisation import BasePlot, ScientificTable, HistogramPlot, LineGraphPlot, LDA_Visualisation


if __name__ == "__main__":

    # Top Dog Toggles
    sentiment_context = 'Emotion Classification'
    our_datasets = '' 
    your_dataset_toggle = True

    # Dataset-related toggles
    sample_size = 1
    shuffle_data = True   
    split = 'train'

    # Sentiment Toggles
    sentiment_analysis = True
    sentiment_mode = 'classic' #use snapshot, longform or classic
    max_seq_length = 128
    model_size = 'base'

    # Fine-tune parameters
    fine_tune_or_inference = 'inference' 
    save_fine_tune = 'yes' 
    fine_tune_version_name = '' 
    fine_tune_quality = True 

    # LDA toggles
    LDA_analysis = True
    LDA_num_topics = 10

    # BERTopic Toggles
    BERTopic_analysis = True
    BERTopic_num_topics = 10

    # Kcluster Toggle
    Kcluster_analysis = False 

    #MetaData Toggle
    Metadata_analysis = True

    # Data Visualisation Toggle
    Data_Visualisation = True
    display_in_browser = True
    save_png = False
    generate_table = True
    num_table_entries = 10
    generate_histogram = True
    generate_line_graph = True
    t_sne_perplexity = 30
    t_sne_learning_rate = 200
    generate_LDA = True

    # Initialize DatasetHandler
    dataset_handler = DatasetHandler(csv_file='output.csv')
    if your_dataset_toggle and not our_datasets:
        print("No addtional background dataset was chosen.")
    else:
        # Initialize or read the CSV file
        dataset_handler.initialize_csv(
            our_datasets=our_datasets,
            your_dataset_toggle=your_dataset_toggle,
            sample_size=sample_size,
            shuffle_data=shuffle_data,
            split=split
        )

    if fine_tune_or_inference == 'fine_tune' and sentiment_analysis:
        # Initialize the fine-tuner
        fine_tuner = ALBERTSentimentFineTuner(
            max_seq_length=max_seq_length,
            model_size=model_size
        )

        # Prepare the dataset for fine-tuning
        fine_tune_dataset = {
            'train': fine_tuner.prepare_finetuning_dataset(
                dataset_name=our_datasets,
                split='train',
                sample_size=sample_size,
                shuffle_data=shuffle_data
            ),
            'test': fine_tuner.prepare_finetuning_dataset(
                dataset_name=our_datasets,
                split='test',
                sample_size=sample_size,
                shuffle_data=shuffle_data
            )
        }
        
        # Initialize the model after num_labels has been set in alb_s1
        fine_tuner.initialize_model(sentiment_context=sentiment_context)
        print("Fine_Tune mode selected")

        # Run the fine-tuning process using the prepared dataset
        fine_tuner.fine_tune(
            dataset=fine_tune_dataset,
            start_from_sentiment_context=sentiment_context,
            save_fine_tune=save_fine_tune,
            fine_tune_version_name=fine_tune_version_name,
            fine_tune_quality=fine_tune_quality
        )
        print("Fine-tuning completed.")

    else:
        # For each analysis
        if sentiment_analysis:
            inferencer = ALBERTSentimentInferencer(
                max_seq_length=max_seq_length,
                model_size=model_size
            )

            if your_dataset_toggle and not our_datasets:
                # Load the metadata.json file to get the output_mode for the given sentiment_context
                import json

                with open('metadata.json', 'r') as f:
                    metadata_list = json.load(f)

                # Find the entry with the matching sentiment_context
                metadata = next((item for item in metadata_list if item['version_name'] == sentiment_context), None)

                if metadata:
                    # Use the output_mode from metadata as num_labels
                    inferencer.num_labels = metadata['output_mode']
                    print(f"Running with output_mode={metadata['output_mode']} for sentiment_context='{sentiment_context}'.")
                else:
                    raise ValueError(f"Metadata for sentiment_context '{sentiment_context}' not found.")
            elif your_dataset_toggle and our_datasets:
                # Load dataset configuration to set num_labels
                inferencer.load_dataset_config(dataset_name=our_datasets)
            else:
                print("No valid dataset provided. Skipping dataset loading.")

            inferencer.initialize_model(sentiment_context=sentiment_context)

            texts_for_analysis = dataset_handler.get_texts_for_analysis()

            # Run inference, only passing dataset_name if our_datasets is provided
            predictions = inferencer.run_inference(
                texts=texts_for_analysis,
                sentiment_context=sentiment_context,
                dataset_name=our_datasets if our_datasets else None, 
                sentiment_mode=sentiment_mode
            )

            sentiment_csv_saver = SentimentCSVDataSaver(
                dataset_handler=dataset_handler,
                sentiment_context=sentiment_context
            )

            sentiment_csv_saver.save_results(
                predictions=predictions
            )

        if LDA_analysis:
            # Initialize the LDA processor with the toggle for number of topics
            lda_text_processor = LDATextProcessor()
            lda_processor = LDAProcessor(num_topics=LDA_num_topics)

            # Get texts for analysis
            texts_for_analysis = dataset_handler.get_texts_for_analysis()

            # Preprocess the texts
            lda_texts = lda_text_processor.preprocess_texts(texts_for_analysis)

            # Perform LDA and get dominant topics
            lda_model, corpus, dominant_topics = lda_processor.perform_lda(lda_texts)

            # Get topic matrix
            topic_matrix = lda_processor.get_topic_matrix(lda_model, corpus)

            # Perform PCA
            pca_result = lda_processor.perform_pca(topic_matrix)

            # Perform t-SNE
            tsne_result = lda_processor.perform_tsne(topic_matrix)

            # Convert results to lists
            pca_coordinates = pca_result.tolist()
            tsne_coordinates = tsne_result.tolist()

            # Save LDA results and dominant topics to CSV
            lda_csv_saver = LDACSVDataSaver(dataset_handler)
            lda_csv_saver.save_results(
                pca_coordinates=pca_coordinates,
                tsne_coordinates=tsne_coordinates,
                dominant_topics=dominant_topics
            )

        if BERTopic_analysis:
                    
            bertopic_processor = BERTopicProcessor(num_topics=BERTopic_num_topics)

            # Get texts for BERTopic analysis
            texts_for_bertopic = dataset_handler.get_texts_for_analysis()

            # Initialize the BERTopic model based on the size of the dataset
            bertopic_processor.initialize_model(len(texts_for_bertopic))

            # Run BERTopic analysis
            bertopic_results = bertopic_processor.perform_bertopic(texts=texts_for_bertopic)

            # Save visualizations as objects for later use
            BERT_distance = bertopic_processor.topic_model.visualize_topics()
            BERT_similarity_matrix = bertopic_processor.topic_model.visualize_heatmap()
            
            # Pass bertopic_processor to the CSV saver (optional)
            bertopic_csv_saver = BERTopicCSVDataSaver(dataset_handler=dataset_handler, bertopic_processor=bertopic_processor)
            bertopic_csv_saver.save_results(predictions=bertopic_results)

        if Kcluster_analysis:
            # Initialize the KCluster processor and TextProcessor
            text_processor = TextProcessingAndDatabase(dataset_handler)
            kcluster_processor = KclusterAnalysis(n_components=3)

            # Get texts for K-cluster analysis
            df, texts_for_kcluster = text_processor.process_texts()

            # Perform KMeans clustering and PCA
            if texts_for_kcluster:
                pca_results, labels = kcluster_processor.perform_analysis(texts_for_kcluster)  # Single call to perform_analysis

                # Save KCluster results to CSV
                kcluster_csv_saver = KclusterCSVDataSaver()
                kcluster_csv_saver.save_analysis_to_csv(pca_results, labels, dataset_handler)

    if Metadata_analysis:
        # Initialize the metadata saver
        metadata_saver = MetadataCSVDataSaver(dataset_handler=dataset_handler)

        # Determine the version_name for loading metadata.json
        if fine_tune_or_inference == 'fine_tune':
            version_name = fine_tune_version_name
        else:
            version_name = sentiment_context

        # Save metadata to CSV
        metadata_saver.save_metadata(
            your_dataset_toggle=your_dataset_toggle,
            our_datasets=our_datasets,
            sample_size=sample_size,
            shuffle_data=shuffle_data,
            sentiment_context=sentiment_context if sentiment_analysis else None,
            LDA_analysis=LDA_analysis,
            LDA_num_topics=LDA_num_topics if LDA_analysis else None,
            version_name=version_name
        )

    if Data_Visualisation:

        # Read the updated CSV into a DataFrame
        data = dataset_handler.read_csv()

        # Initialize figure counter
        fig_counter = 1

        # Create a BasePlot instance to identify columns
        base_plot = BasePlot(data)

        if generate_table:
            # Count the total number of 'probability distribution' columns
            probability_distribution_columns = []
            for col in base_plot.prediction_columns:
                prediction_type, number, prediction_word = base_plot.extract_prediction_info(col)
                if prediction_type == 'probability distribution':
                    probability_distribution_columns.append(col)
            total_probability_distributions = len(probability_distribution_columns)

            # Loop through prediction columns
            for column in base_plot.prediction_columns:
                prediction_type, number, prediction_word = base_plot.extract_prediction_info(column)

                # Skip 'probability distribution 2: {word}' columns only if there are exactly 2 distributions
                if prediction_type == 'probability distribution' and number == '2' and total_probability_distributions == 2:
                    # Skip producing this table
                    continue

                # Create a scientific table
                table = ScientificTable(
                    data=data,
                    column_name=column,
                    num_entries=num_table_entries,
                    fig_counter=fig_counter,
                    prediction_word=prediction_word
                )
                table.create_table()
                if display_in_browser:
                    table.display()
                if save_png:
                    filename = f'table_{fig_counter}.png'
                    table.save_png(filename)
                fig_counter += 1

        if generate_histogram:
            histogram = HistogramPlot(
                data=data,
                fig_counter=fig_counter,
                title_font_size=24,
                axis_label_font_size=16,
                axis_title_font_size=20,
                tick_font_size=14,
            )
            histogram.create_histogram()
            if hasattr(histogram, 'fig'):
                if display_in_browser:
                    histogram.display()
                if save_png:
                    filename = f'histogram_{fig_counter}.png'
                    histogram.save_png(filename)
                fig_counter += 1

        if generate_line_graph:
            line_graph = LineGraphPlot(
                data=data,
                fig_counter=fig_counter,
                perplexity=t_sne_perplexity,
                learning_rate=t_sne_learning_rate,
                title_font_size=24,
                axis_label_font_size=16,
                axis_title_font_size=20,
                tick_font_size=14,
            )
            line_graph.create_line_graph()
            if hasattr(line_graph, 'fig'):
                if display_in_browser:
                    line_graph.display()
                if save_png:
                    filename = f'line_graph_{fig_counter}.png'
                    line_graph.save_png(filename)
                fig_counter += 1

        if generate_LDA:
            lda_visualisation = LDA_Visualisation(data)
            lda_visualisation.create_lda_visualisation()
            lda_visualisation.display()

    print("All analyses completed. The results have been updated in 'output.csv'.")