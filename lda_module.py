import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm 

class LDATextProcessor:
    def __init__(self):
        # Ensure NLTK stopwords are downloaded
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            import nltk
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        # Ensure NLTK punkt tokenizer is downloaded
        try:
            word_tokenize("test")
        except LookupError:
            import nltk
            nltk.download('punkt')

    def clean_text(self, text):
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'\s+', ' ', text)     # Replace multiple spaces with a single space
        return text.strip()

    def tokenize_and_filter(self, text):
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
        return filtered_tokens

    def preprocess_texts(self, texts):
        processed_texts = []
        for text in texts:
            clean_text = self.clean_text(text)
            filtered_tokens = self.tokenize_and_filter(clean_text)
            processed_texts.append(filtered_tokens)
        return processed_texts

class LDAProcessor:
    def __init__(self, num_topics=5):
        self.num_topics = num_topics

    def perform_lda(self, texts):
        dictionary = corpora.Dictionary(texts)
        
        # Track the progress of corpus creation with a progress bar
        print("Performing LDA...")
        corpus = [dictionary.doc2bow(text) for text in tqdm(texts, desc="Creating Corpus", unit="document")]
        
        # Train the LDA model on the created corpus
        lda_model = models.LdaModel(corpus, num_topics=self.num_topics, random_state=42)
        
        dominant_topics = [max(lda_model[doc], key=lambda x: x[1])[0] for doc in corpus]
        
        return lda_model, corpus, dominant_topics

    def get_topic_matrix(self, lda_model, corpus):
        topic_distributions = [lda_model.get_document_topics(doc, minimum_probability=0) for doc in corpus]
        topic_matrix = np.array([[prob for _, prob in dist] for dist in topic_distributions])
        return topic_matrix

    def perform_pca(self, topic_matrix):
        pca = PCA(n_components=3, random_state=42)
        pca_result = pca.fit_transform(topic_matrix)
        print("Completed PCA on LDA topic distributions")
        return pca_result

    def perform_tsne(self, topic_matrix):
        tsne_model = TSNE(n_components=3, random_state=42, perplexity=min(30, len(topic_matrix) - 1))
        tsne_result = tsne_model.fit_transform(topic_matrix)
        print("Completed t-SNE on LDA topic distributions")
        return tsne_result

class LDACSVDataSaver:
    def __init__(self, dataset_handler):
        """
        Initializes the LDACSVDataSaver with a reference to DatasetHandler.
        """
        self.dataset_handler = dataset_handler

    def save_results(self, pca_coordinates, tsne_coordinates, dominant_topics=None):
        """
        Saves LDA analysis results to the CSV file.
        """
        # Read the existing CSV file
        df = self.dataset_handler.read_csv()

        # Add LDA analysis results
        df['LDA_PCA_Coordinates'] = [','.join(map(str, coords)) for coords in pca_coordinates]
        df['LDA_t_SNE_Coordinates'] = [','.join(map(str, coords)) for coords in tsne_coordinates]
        df['LDA_Topics'] = dominant_topics

        # Write back to CSV
        self.dataset_handler.write_csv(df)

