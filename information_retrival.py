import re
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class InformationRetrievalSystem:
    def __init__(self, folder_path=None):
        """Initialize the IR system with a folder containing text files."""
        self.folder_path = folder_path or os.getcwd()
        self.documents = []
        self.document_metadata = []  
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = None
        self.file_sources = []  
        
    def preprocess_text(self, text):
        """Clean and preprocess the text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()
    
    def count_statistics(self, text):
        """Calculate word count, character count, and other statistics."""
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        char_count_no_spaces = len(text.replace(' ', ''))
        unique_words = len(set(w.lower() for w in words))
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'char_count_no_spaces': char_count_no_spaces,
            'unique_words': unique_words,
            'avg_word_length': char_count_no_spaces / word_count if word_count > 0 else 0
        }
    
    def load_and_chunk_documents(self, chunk_size=500):
        """Load all txt documents from folder and split into chunks."""
        try:
            txt_files = list(Path(self.folder_path).glob('*.txt'))
            
            if not txt_files:
                print(f"Error: No .txt files found in '{self.folder_path}'")
                return False
            
            self.documents = []
            self.document_metadata = []
            self.file_sources = []
            
            for file_path in sorted(txt_files):
                print(f"\n📄 Processing file: {file_path.name}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    file_stats = self.count_statistics(content)
                    print(f"   ✓ Words: {file_stats['word_count']}, Characters: {file_stats['char_count']}")
                    
                    content = self.preprocess_text(content)
                    
                    sentences = re.split(r'[.!?]+', content)
                    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
                    
                    current_chunk = []
                    current_length = 0
                    
                    for sentence in sentences:
                        if current_length + len(sentence) > chunk_size and current_chunk:
                            chunk_text = ' '.join(current_chunk)
                            self.documents.append(chunk_text)
                            chunk_stats = self.count_statistics(chunk_text)
                            self.document_metadata.append(chunk_stats)
                            self.file_sources.append(file_path.name)
                            
                            current_chunk = [sentence]
                            current_length = len(sentence)
                        else:
                            current_chunk.append(sentence)
                            current_length += len(sentence)
                    
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        self.documents.append(chunk_text)
                        chunk_stats = self.count_statistics(chunk_text)
                        self.document_metadata.append(chunk_stats)
                        self.file_sources.append(file_path.name)
                
                except Exception as e:
                    print(f"   ✗ Error reading file: {e}")
                    continue
            
            print(f"\n✓ Loaded {len(self.documents)} document chunks from {len(txt_files)} file(s)")
            return True
            
        except Exception as e:
            print(f"Error loading documents: {e}")
            return False
    
    def build_index(self):
        """Build TF-IDF index from documents."""
        if not self.documents:
            print("No documents loaded. Call load_and_chunk_document() first.")
            return False
        
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            print(f"Index built with {self.tfidf_matrix.shape[0]} documents and {self.tfidf_matrix.shape[1]} features")
            return True
        except Exception as e:
            print(f"Error building index: {e}")
            return False
    
    def search(self, query, top_k=5):
        """Search for relevant documents given a query."""
        if self.tfidf_matrix is None:
            print("Index not built. Call build_index() first.")
            return []
        
        try:
            query_vec = self.vectorizer.transform([query])
            
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for i, idx in enumerate(top_indices):
                if similarities[idx] > 0:  
                    results.append({
                        'rank': i + 1,
                        'score': similarities[idx],
                        'document': self.documents[idx],
                        'file_source': self.file_sources[idx],
                        'preview': self.documents[idx][:200] + '...' if len(self.documents[idx]) > 200 else self.documents[idx],
                        'statistics': self.document_metadata[idx]
                    })
            
            return results
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def display_results(self, results):
        """Pretty print search results with detailed information."""
        if not results:
            print("No relevant results found.")
            return
        
        print(f"\n{'='*100}")
        print(f"Found {len(results)} relevant results:")
        print(f"{'='*100}\n")
        
        for result in results:
            stats = result['statistics']
            print(f"Rank {result['rank']} (Relevance Score: {result['score']:.4f})")
            print(f"Source File: {result['file_source']}")
            print(f"{'-'*100}")
            print(f"📊 Statistics:")
            print(f"   • Word Count: {stats['word_count']}")
            print(f"   • Character Count: {stats['char_count']} (with spaces), {stats['char_count_no_spaces']} (without spaces)")
            print(f"   • Unique Words: {stats['unique_words']}")
            print(f"   • Average Word Length: {stats['avg_word_length']:.2f} characters")
            print(f"\n📝 Content Preview:")
            print(result['preview'])
            print(f"\n{'='*100}\n")


if __name__ == "__main__":
    current_folder = os.path.dirname(os.path.abspath(__file__))
    
    ir_system = InformationRetrievalSystem(folder_path=current_folder)
    
    if ir_system.load_and_chunk_documents(chunk_size=400):
        if ir_system.build_index():
            
            print("\n" + "="*100)
            print("INFORMATION RETRIEVAL SYSTEM - MULTI-FILE SUPPORT")
            print("="*100)
            print("\nArticles loaded and indexed successfully!")
            print(f"Total searchable chunks: {len(ir_system.documents)}")
            print("\nSearch Tips:")
            print("   - Use single words: 'quantum', 'entanglement', 'Einstein'")
            print("   - Use phrases: 'spooky action', 'Bell theorem'")
            print("   - Use concepts: 'EPR paradox', 'quantum computing'")
            print("\nType 'exit' or 'quit' to leave")
            print("="*100 + "\n")
            
            while True:
                try:
                    query = input("Enter search query: ").strip()
                    
                    if query.lower() in ['exit', 'quit', 'q']:
                        print("\nThanks for using the IR system. Goodbye!")
                        break
                    
                    if not query:
                        print("Please enter a search term.\n")
                        continue
                    
                    if len(query.split()) > 10:
                        print("Query too long. Please use up to 10 words for best results.\n")
                        continue
                    
                    print(f"\nSearching for: '{query}'...")
                    results = ir_system.search(query, top_k=5)
                    
                    if results:
                        ir_system.display_results(results)
                    else:
                        print("No relevant results found. Try different keywords.\n")
                    
                    print("-" * 100 + "\n")
                    
                except KeyboardInterrupt:
                    print("\n\nInterrupted. Goodbye!")
                    break
                except Exception as e:
                    print(f"Error: {e}\n")