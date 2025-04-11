import re
import math
import json
import logging
import argparse
import sqlite3
from collections import Counter
from multiprocessing import Pool
from typing import List, Tuple, Dict
import csv
from io import StringIO, BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch
from flask import Flask, request, render_template, jsonify, send_file
from bs4 import BeautifulSoup
import bleach
from suffix_trees import STree

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt', quiet=True)

# Initialize Flask app
app = Flask(__name__)

class EnhancedOLMOTRACE:
    def __init__(self, db_path: str = ":memory:", tokenizer_name: str = "gpt2"):
        """
        Initialize with a SQLite database for the corpus.
        """
        self.db_path = db_path
        self.setup_database()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            logger.error(f"Failed to load tokenizer {tokenizer_name}: {e}. Falling back to NLTK.")
            self.tokenizer = None
        try:
            self.llm_judge_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.llm_judge_model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=4
            )
            self.fine_tune_llm_judge()
            self.llm_judge_model.eval()
        except Exception as e:
            logger.error(f"Failed to load or fine-tune DistilBERT: {e}. Falling back to heuristic.")
            self.llm_judge_model = None
        self.unigram_probs = self.compute_unigram_probs()
        self.suffix_tree = self.build_suffix_tree()
        logger.info("Initialized OLMOTRACE.")

    def setup_database(self):
        """
        Set up SQLite database for storing corpus.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('''CREATE TABLE IF NOT EXISTS corpus
                             (id INTEGER PRIMARY KEY AUTOINCREMENT, document TEXT)''')
                conn.commit()
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM corpus")
                if c.fetchone()[0] == 0:
                    default_corpus = [
                        "The Space Needle was built for the 1962 World Fair in Seattle.",
                        "I'm going on an adventure to explore new lands.",
                        "The binomial coefficient 10 choose 4 is 210.",
                        "The Space Needle is an iconic landmark in Seattle.",
                        "Adventure awaits those who seek it bravely."
                    ]
                    c.executemany("INSERT INTO corpus (document) VALUES (?)",
                                  [(doc,) for doc in default_corpus])
                    conn.commit()
            logger.info(f"Initialized SQLite database at {self.db_path}.")
        except Exception as e:
            logger.error(f"Error setting up database: {e}")

    def load_corpus_from_db(self) -> List[str]:
        """
        Load corpus from SQLite database.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute("SELECT document FROM corpus")
                corpus = [row[0] for row in c.fetchall()]
            return corpus
        except Exception as e:
            logger.error(f"Error loading corpus from DB: {e}")
            return []

    def save_corpus_to_db(self, corpus: List[str]):
        """
        Save corpus to SQLite database.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute("DELETE FROM corpus")
                c.executemany("INSERT INTO corpus (document) VALUES (?)",
                              [(doc,) for doc in corpus])
                conn.commit()
            logger.info(f"Saved {len(corpus)} documents to database.")
            self.unigram_probs = self.compute_unigram_probs()
            self.suffix_tree = self.build_suffix_tree()
        except Exception as e:
            logger.error(f"Error saving corpus to DB: {e}")

    def preprocess_document(self, text: str) -> str:
        """
        Clean document by removing HTML tags and normalizing whitespace.
        """
        try:
            soup = BeautifulSoup(text, 'html.parser')
            clean_text = soup.get_text()
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            clean_text = bleach.clean(clean_text, tags=[], strip=True)
            return clean_text
        except Exception as e:
            logger.error(f"Error preprocessing document: {e}")
            return text

    def compute_unigram_probs(self) -> Dict[str, float]:
        """
        Compute unigram log probabilities for tokens in the corpus.
        """
        corpus = self.load_corpus_from_db()
        all_tokens = []
        for doc in corpus:
            tokens = self.tokenize(doc)
            all_tokens.extend(tokens)
        
        token_counts = Counter(all_tokens)
        total_tokens = sum(token_counts.values())
        
        unigram_probs = {
            token: math.log(count / total_tokens) if count > 0 else float('-inf')
            for token, count in token_counts.items()
        }
        logger.info(f"Computed unigram probabilities for {len(unigram_probs)} unique tokens.")
        return unigram_probs

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using the loaded tokenizer or NLTK as fallback.
        """
        try:
            if self.tokenizer:
                tokens = self.tokenizer.tokenize(text)
                return [self.tokenizer.convert_tokens_to_string([t]).strip() for t in tokens]
            return word_tokenize(text.lower())
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            return text.lower().split()

    def build_suffix_tree(self) -> STree:
        """
        Build a suffix tree for the corpus.
        """
        try:
            corpus = self.load_corpus_from_db()
            combined_text = '$'.join(corpus)
            st = STree.STree(combined_text)
            logger.info("Built suffix tree for corpus.")
            return st
        except Exception as e:
            logger.error(f"Failed to build suffix tree: {e}")
            return None

    def fine_tune_llm_judge(self):
        """
        Fine-tune DistilBERT on a synthetic dataset for relevance scoring.
        """
        if not self.llm_judge_model:
            return
        
        data = [
            {"prompt": "Tell me about the Space Needle.", 
             "output": "The Space Needle was built for the 1962 World Fair.", 
             "snippet": "The Space Needle was built for the 1962 World Fair in Seattle.", 
             "label": 3},
            {"prompt": "Tell me about the Space Needle.", 
             "output": "The Space Needle was built for the 1962 World Fair.", 
             "snippet": "I'm going on an adventure to explore new lands.", 
             "label": 0},
            {"prompt": "What is an adventure?", 
             "output": "I'm going on an adventure.", 
             "snippet": "Adventure awaits those who seek it bravely.", 
             "label": 2},
            {"prompt": "Tell me about math.", 
             "output": "The binomial coefficient 10 choose 4 is 210.", 
             "snippet": "The Space Needle is an iconic landmark.", 
             "label": 0},
        ]
        
        texts = [f"{d['prompt']} [SEP] {d['output']} [SEP] {d['snippet']}" for d in data]
        labels = [d['label'] for d in data]
        encodings = self.llm_judge_tokenizer(texts, truncation=True, padding=True, max_length=512)
        dataset = Dataset.from_dict({
            "input_ids": encodings['input_ids'],
            "attention_mask": encodings['attention_mask'],
            "labels": labels
        })
        
        training_args = TrainingArguments(
            output_dir="./distilbert_finetune",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            logging_dir='./logs',
            logging_steps=10,
            save_steps=100,
            report_to="none"
        )
        
        trainer = Trainer(
            model=self.llm_judge_model,
            args=training_args,
            train_dataset=dataset
        )
        
        try:
            trainer.train()
            logger.info("Fine-tuned DistilBERT for relevance scoring.")
        except Exception as e:
            logger.error(f"Error fine-tuning DistilBERT: {e}")

    def find_maximal_spans(self, lm_output: str, token_level: bool = False) -> List[Tuple[str, int, int]]:
        """
        Find maximal spans or individual tokens in lm_output that appear in the corpus.
        """
        tokens = self.tokenize(lm_output)
        L = len(tokens)
        spans = []

        def check_span(start: int, end: int) -> Tuple[str, int, int]:
            span_tokens = tokens[start:end]
            span_text = ' '.join(span_tokens)
            corpus = self.load_corpus_from_db()
            if self.suffix_tree and self.suffix_tree.find(span_text) != -1:
                return (span_text, start, end)
            elif not self.suffix_tree:
                for doc in corpus:
                    if span_text in doc.lower():
                        return (span_text, start, end)
            return None

        if token_level:
            tasks = [(i, i + 1) for i in range(L)]
        else:
            tasks = [(i, j) for i in range(L) for j in range(i + 1, min(i + 20, L + 1))]
        
        with Pool() as pool:
            results = pool.starmap(check_span, tasks)
            spans = [r for r in results if r]

        maximal_spans = []
        for span, start, end in spans:
            is_maximal = True
            span_text = span.lower()
            if not token_level and re.search(r'[.\n]', span_text[:-1]):
                continue
            if not token_level and end < L:
                longer_span = ' '.join(tokens[start:end + 1])
                corpus = self.load_corpus_from_db()
                if self.suffix_tree and self.suffix_tree.find(longer_span) != -1:
                    is_maximal = False
                elif not self.suffix_tree and any(longer_span in doc.lower() for doc in corpus):
                    is_maximal = False
            if not token_level and start > 0:
                longer_span = ' '.join(tokens[start - 1:end])
                corpus = self.load_corpus_from_db()
                if self.suffix_tree and self.suffix_tree.find(longer_span) != -1:
                    is_maximal = False
                elif not self.suffix_tree and any(longer_span in doc.lower() for doc in corpus):
                    is_maximal = False
            if is_maximal:
                maximal_spans.append((span, start, end))

        logger.info(f"Found {len(maximal_spans)} maximal {'tokens' if token_level else 'spans'}.")
        return maximal_spans

    def merge_spans(self, spans: List[Tuple[str, int, int]], lm_output: str, token_level: bool = False) -> List[Tuple[str, int, int]]:
        """
        Merge overlapping spans (skipped for token-level tracing).
        """
        if not spans or token_level:
            return spans
        
        spans.sort(key=lambda x: x[1])
        merged = []
        current_span, current_start, current_end = spans[0]
        
        for span, start, end in spans[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
                current_span = ' '.join(self.tokenize(lm_output)[current_start:current_end])
            else:
                merged.append((current_span, current_start, current_end))
                current_span, current_start, current_end = span, start, end
        merged.append((current_span, current_start, current_end))
        
        logger.info(f"Merged to {len(merged)} spans.")
        return merged

    def compute_span_unigram_prob(self, span: str) -> float:
        """
        Compute the log unigram probability of a span.
        """
        tokens = self.tokenize(span)
        prob = sum(self.unigram_probs.get(t, float('-inf')) for t in tokens)
        return prob

    def filter_spans(self, spans: List[Tuple[str, int, int]], k_factor: float = 0.05, token_level: bool = False) -> List[Tuple[str, int, int]]:
        """
        Filter spans to keep top K spans/tokens with lowest unigram probability.
        """
        if not spans:
            return []
        
        span_probs = [(s, start, end, self.compute_span_unigram_prob(s)) for s, start, end in spans]
        span_probs.sort(key=lambda x: x[3])
        
        max_end = max(end for _, _, end, _ in span_probs)
        K = max(1, int(k_factor * max_end)) if not token_level else len(spans)
        filtered = [(s, start, end) for s, start, end, _ in span_probs[:K]]
        
        logger.info(f"Filtered to {len(filtered)} {'tokens' if token_level else 'spans'}.")
        return filtered

    def llm_judge(self, prompt: str, lm_output: str, snippet: str) -> int:
        """
        Evaluate document relevance using fine-tuned DistilBERT (0-3 scale).
        """
        try:
            if not self.llm_judge_model:
                prompt_tokens = set(self.tokenize(prompt.lower()))
                output_tokens = set(self.tokenize(lm_output.lower()))
                snippet_tokens = set(self.tokenize(snippet.lower()))
                prompt_overlap = len(prompt_tokens & snippet_tokens) / max(len(prompt_tokens), 1)
                output_overlap = len(output_tokens & snippet_tokens) / max(len(output_tokens), 1)
                score = (prompt_overlap + output_overlap) * 1.5
                if score > 0.75:
                    return 3
                elif score > 0.5:
                    return 2
                elif score > 0.25:
                    return 1
                return 0
            
            input_text = f"{prompt} [SEP] {lm_output} [SEP] {snippet}"
            inputs = self.llm_judge_tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            with torch.no_grad():
                outputs = self.llm_judge_model(**inputs)
                logits = outputs.logits
                score = torch.argmax(logits, dim=1).item()
            return min(score, 3)
        except Exception as e:
            logger.error(f"Error in LLM judge: {e}")
            return 0

    def retrieve_documents(self, spans: List[Tuple[str, int, int]], prompt: str, lm_output: str, max_docs: int = 10, snippet_length: int = 20, token_level: bool = False) -> List[Tuple[str, List[Tuple[int, str, float, int]]]]:
        """
        Retrieve documents with BM25 scores and LLM relevance.
        """
        try:
            query = (prompt + ' ' + lm_output).strip()
            corpus = self.load_corpus_from_db()
            tokenized_corpus = [self.tokenize(doc) for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            query_tokens = self.tokenize(query)
            scores = bm25.get_scores(query_tokens)
            
            results = []
            for span, start, end in spans:
                matching_docs = []
                span_tokens = self.tokenize(span)
                
                for doc_id, (doc, bm25_score) in enumerate(zip(corpus, scores)):
                    if span.lower() in doc.lower():
                        tokens = self.tokenize(doc)
                        for i in range(len(tokens) - len(span_tokens) + 1):
                            if tokens[i:i + len(span_tokens)] == span_tokens:
                                start_idx = max(0, i - snippet_length // 2)
                                end_idx = min(len(tokens), i + len(span_tokens) + snippet_length // 2)
                                snippet = ' '.join(tokens[start_idx:end_idx])
                                relevance = self.llm_judge(prompt, lm_output, snippet)
                                matching_docs.append((doc_id, snippet, bm25_score, relevance))
                                break
                    if len(matching_docs) >= max_docs:
                        break
                
                matching_docs.sort(key=lambda x: (x[2], x[3]), reverse=True)
                results.append((span, matching_docs))
            
            logger.info(f"Retrieved documents for {len(results)} spans.")
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return [(span, []) for span, _, _ in spans]

    def get_full_document(self, doc_id: int, span: str = None) -> str:
        """
        Retrieve the full document text by ID with optional span highlighting.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute("SELECT document FROM corpus WHERE id = ?", (doc_id + 1,))
                result = c.fetchone()
                if result and span:
                    doc = result[0]
                    return re.sub(f'({re.escape(span)})', r'<mark>\1</mark>', doc, flags=re.IGNORECASE)
                return result[0] if result else ""
        except Exception as e:
            logger.error(f"Error retrieving full document {doc_id}: {e}")
            return ""

    def search_corpus(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search the corpus for documents matching a query.
        """
        try:
            corpus = self.load_corpus_from_db()
            tokenized_corpus = [self.tokenize(doc) for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            query_tokens = self.tokenize(query)
            scores = bm25.get_scores(query_tokens)
            
            results = []
            for doc_id, (doc, score) in enumerate(zip(corpus, scores)):
                if score > 0:
                    results.append({"doc_id": doc_id, "document": doc, "bm25_score": score})
                if len(results) >= max_results:
                    break
            
            results.sort(key=lambda x: x["bm25_score"], reverse=True)
            return results[:max_results]
        except Exception as e:
            logger.error(f"Error searching corpus: {e}")
            return []

    def trace(self, lm_output: str, user_prompt: str = "", max_docs: int = 10, snippet_length: int = 20, token_level: bool = False) -> Tuple[List[Tuple[str, int, int]], List[Tuple[str, List[Tuple[int, str, float, int]]]]]:
        """
        Trace an LM output through the pipeline.
        """
        if not lm_output:
            logger.error("Empty LM output provided.")
            return [], []
        
        logger.info(f"Starting trace process (token_level={token_level}).")
        
        spans = self.find_maximal_spans(lm_output, token_level)
        spans = self.merge_spans(spans, lm_output, token_level)
        spans = self.filter_spans(spans, token_level=token_level)
        results = self.retrieve_documents(spans, user_prompt, lm_output, max_docs, snippet_length, token_level)
        
        logger.info("Trace completed.")
        return spans, results

    def save_results(self, spans: List[Tuple[str, int, int]], results: List[Tuple[str, List[Tuple[int, str, float, int]]]], filename: str):
        """
        Save tracing results to a JSON file.
        """
        try:
            output = {
                "spans": [{"text": s, "start": start, "end": end} for s, start, end in spans],
                "documents": [
                    {
                        "span": span,
                        "matches": [
                            {"doc_id": doc_id, "snippet": snippet, "bm25_score": score, "relevance": relevance}
                            for doc_id, snippet, score, relevance in docs
                        ]
                    }
                    for span, docs in results
                ]
            }
            with open(filename, 'w') as f:
                json.dump(output, f, indent=2)
            logger.info(f"Saved results to {filename}.")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

def load_corpus(corpus_file: str) -> List[str]:
    """
    Load a corpus from a text file.
    """
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded corpus with {len(corpus)} documents from {corpus_file}.")
        return corpus
    except Exception as e:
        logger.error(f"Failed to load corpus: {e}")
        return []

# Initialize tracer globally
tracer = EnhancedOLMOTRACE(db_path="corpus.db")

# Flask routes
@app.route('/')
def index():
    """
    Render the main page with input form.
    """
    return render_template('index.html')

@app.route('/trace', methods=['POST'])
def trace():
    """
    Handle trace request from the web interface.
    """
    try:
        data = request.form
        lm_output = data.get('lm_output', '').strip()
        user_prompt = data.get('user_prompt', '').strip()
        max_docs = int(data.get('max_docs', 10))
        snippet_length = int(data.get('snippet_length', 20))
        token_level = data.get('token_level', 'false').lower() == 'true'
        
        if not lm_output:
            return jsonify({"error": "LM output is required."}), 400
        
        spans, results = tracer.trace(lm_output, user_prompt, max_docs, snippet_length, token_level)
        
        highlighted_output = lm_output
        span_highlights = []
        for span, start, end in spans:
            span_highlights.append({"text": span, "start": start, "end": end})
        
        doc_data = []
        for span, docs in results:
            doc_list = [
                {
                    "doc_id": doc_id,
                    "snippet": snippet,
                    "bm25_score": round(score, 2),
                    "relevance": relevance,
                    "relevance_label": {0: "Low", 1: "Medium", 2: "Relevant", 3: "High"}[relevance],
                    "full_document": tracer.get_full_document(doc_id, span)
                }
                for doc_id, snippet, score, relevance in docs
            ]
            doc_data.append({"span": span, "documents": doc_list})
        
        return jsonify({
            "highlighted_output": highlighted_output,
            "spans": span_highlights,
            "documents": doc_data,
            "token_level": token_level
        })
    except Exception as e:
        logger.error(f"Error in trace route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    """
    Handle corpus search request.
    """
    try:
        data = request.form
        query = data.get('query', '').strip()
        max_results = int(data.get('max_results', 10))
        
        if not query:
            return jsonify({"error": "Query is required."}), 400
        
        results = tracer.search_corpus(query, max_results)
        return jsonify({
            "results": [
                {
                    "doc_id": r["doc_id"],
                    "document": r["document"],
                    "bm25_score": round(r["bm25_score"], 2)
                }
                for r in results
            ]
        })
    except Exception as e:
        logger.error(f"Error in search route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload_corpus', methods=['POST'])
def upload_corpus():
    """
    Handle corpus file upload.
    """
    global tracer
    try:
        if 'corpus_file' not in request.files:
            return jsonify({"error": "No file uploaded."}), 400
        
        file = request.files['corpus_file']
        if file.filename == '':
            return jsonify({"error": "No file selected."}), 400
        
        corpus = [tracer.preprocess_document(line.decode('utf-8').strip()) 
                  for line in file if line.strip()]
        if not corpus:
            return jsonify({"error": "Empty corpus."}), 400
        
        tracer.save_corpus_to_db(corpus)
        return jsonify({"message": f"Loaded corpus with {len(corpus)} documents."})
    except Exception as e:
        logger.error(f"Error uploading corpus: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/export_csv', methods=['POST'])
def export_csv():
    """
    Export tracing results as CSV.
    """
    try:
        data = request.get_json()
        spans = data.get('spans', [])
        documents = data.get('documents', [])
        
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["Span/Token", "Doc ID", "Snippet", "BM25 Score", "Relevance", "Full Document"])
        
        for span_data in documents:
            span = span_data["span"]
            for doc in span_data["documents"]:
                writer.writerow([
                    span,
                    doc["doc_id"],
                    doc["snippet"],
                    doc["bm25_score"],
                    doc["relevance_label"] + f" ({doc['relevance']})",
                    doc["full_document"]
                ])
        
        output.seek(0)
        return send_file(
            BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='trace_results.csv'
        )
    except Exception as e:
        logger.error(f"Error exporting CSV: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/export_pdf', methods=['POST'])
def export_pdf():
    """
    Export tracing results as PDF.
    """
    try:
        data = request.get_json()
        spans = data.get('spans', [])
        documents = data.get('documents', [])
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph("OLMOTRACE Results", styles['Title']))
        story.append(Spacer(1, 12))
        
        for span_data in documents:
            span = span_data["span"]
            story.append(Paragraph(f"Span/Token: {span}", styles['Heading2']))
            for doc in span_data["documents"]:
                text = (f"Doc {doc['doc_id']}: {doc['snippet']}<br/>"
                        f"BM25 Score: {doc['bm25_score']}, Relevance: {doc['relevance_label']} ({doc['relevance']})<br/>"
                        f"Full Document: {doc['full_document']}")
                story.append(Paragraph(text, styles['BodyText']))
                story.append(Spacer(1, 6))
            story.append(Spacer(1, 12))
        
        doc.build(story)
        buffer.seek(0)
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='trace_results.pdf'
        )
    except Exception as e:
        logger.error(f"Error exporting PDF: {e}")
        return jsonify({"error": str(e)}), 500

# HTML template
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OLMOTRACE Simulator</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 900px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #333; }
        textarea, input[type="text"] { width: 100%; margin: 10px 0; padding: 10px; border: 1px solid #ccc; border-radius: 4px; }
        input[type="number"] { width: 100px; margin: 10px 0; padding: 5px; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
        button:hover { background: #0056b3; }
        .checkbox { margin: 10px 0; }
        .dropzone { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 10px 0; cursor: pointer; }
        .dropzone.dragover { border-color: #007bff; background: #e9f7ff; }
        .output, .search-results { margin-top: 20px; }
        .span-highlight { background: #ffeb3b; padding: 2px; cursor: pointer; transition: background 0.2s; }
        .span-highlight:hover { background: #ffd700; }
        .document { border: 1px solid #ccc; padding: 15px; margin: 10px 0; border-radius: 4px; background: #fff; }
        .relevance-high { border-left: 5px solid #4caf50; }
        .relevance-relevant { border-left: 5px solid #2196f3; }
        .relevance-medium { border-left: 5px solid #ff9800; }
        .relevance-low { border-left: 5px solid #f44336; }
        .error { color: #d32f2f; font-weight: bold; }
        #loading { display: none; text-align: center; padding: 20px; }
        .full-doc { display: none; margin-top: 10px; padding: 10px; background: #f9f9f9; border: 1px solid #ddd; }
        mark { background: #ffeb3b; }
    </style>
</head>
<body>
    <div class="container">
        <h1>OLMOTRACE Simulator</h1>
        <form id="trace-form">
            <label for="user_prompt">User Prompt:</label><br>
            <textarea id="user_prompt" name="user_prompt" placeholder="Enter user prompt (optional)"></textarea><br>
            <label for="lm_output">LM Output:</label><br>
            <textarea id="lm_output" name="lm_output" placeholder="Enter language model output" required></textarea><br>
            <label for="max_docs">Max Documents per Span:</label><br>
            <input type="number" id="max_docs" name="max_docs" value="10" min="1"><br>
            <label for="snippet_length">Snippet Length (tokens):</label><br>
            <input type="number" id="snippet_length" name="snippet_length" value="20" min="10"><br>
            <label class="checkbox">
                <input type="checkbox" id="token_level" name="token_level" value="true">
                Trace Individual Tokens
            </label><br>
            <button type="submit">Trace</button>
        </form>
        <form id="search-form">
            <label for="search_query">Search Corpus:</label><br>
            <input type="text" id="search_query" name="query" placeholder="Enter search query"><br>
            <label for="max_results">Max Results:</label><br>
            <input type="number" id="max_results" name="max_results" value="10" min="1"><br>
            <button type="submit">Search</button>
        </form>
        <form id="corpus-form">
            <label for="corpus_file">Upload Corpus (one document per line):</label><br>
            <div class="dropzone" id="dropzone">Drag and drop a file here or click to upload</div>
            <input type="file" id="corpus_file" name="corpus_file" accept=".txt" style="display: none;"><br>
            <button type="submit">Upload Corpus</button>
        </form>
        <div id="loading">Processing...</div>
        <div id="error" class="error"></div>
        <div id="output" class="output" style="display: none;">
            <h2>Tracing Results</h2>
            <div id="highlighted-output"></div>
            <div id="export-buttons">
                <button onclick="exportCSV()">Export as CSV</button>
                <button onclick="exportPDF()">Export as PDF</button>
            </div>
            <div id="documents"></div>
        </div>
        <div id="search-results" class="search-results" style="display: none;">
            <h2>Search Results</h2>
            <div id="search-output"></div>
        </div>
    </div>
    <script>
        let latestTraceData = null;

        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('corpus_file');
        dropzone.addEventListener('click', () => fileInput.click());
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });
        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            fileInput.files = e.dataTransfer.files;
        });

        document.getElementById('trace-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            document.getElementById('loading').style.display = 'block';
            document.getElementById('error').textContent = '';
            document.getElementById('output').style.display = 'none';
            document.getElementById('search-results').style.display = 'none';
            
            try {
                const response = await fetch('/trace', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('error').textContent = data.error;
                } else {
                    latestTraceData = data;
                    let outputHtml = data.highlighted_output;
                    data.spans.forEach(span => {
                        const regex = new RegExp(`\\b${span.text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'g');
                        outputHtml = outputHtml.replace(regex, `<span class="span-highlight" data-span="${span.text}">${span.text}</span>`);
                    });
                    document.getElementById('highlighted-output').innerHTML = `<p><strong>Highlighted ${data.token_level ? 'Tokens' : 'Spans'}:</strong> ${outputHtml}</p>`;
                    
                    let docHtml = '';
                    data.documents.forEach(doc => {
                        docHtml += `<h3>${data.token_level ? 'Token' : 'Span'}: "${doc.span}"</h3>`;
                        if (doc.documents.length === 0) {
                            docHtml += '<p>No matching documents found.</p>';
                        } else {
                            doc.documents.forEach(d => {
                                docHtml += `
                                    <div class="document relevance-${d.relevance_label.toLowerCase()}">
                                        <p><strong>Doc ${d.doc_id}</strong>: ${d.snippet}</p>
                                        <p>BM25 Score: ${d.bm25_score}, Relevance: ${d.relevance_label} (${d.relevance})</p>
                                        <button onclick="toggleFullDoc(${d.doc_id}, this, '${d.full_document}')">View Full Document</button>
                                        <div class="full-doc" id="full-doc-${d.doc_id}">${d.full_document}</div>
                                    </div>`;
                            });
                        }
                    });
                    document.getElementById('documents').innerHTML = docHtml;
                    document.getElementById('output').style.display = 'block';
                    
                    document.querySelectorAll('.span-highlight').forEach(span => {
                        span.addEventListener('click', () => {
                            const spanText = span.dataset.span;
                            const docs = data.documents.find(d => d.span === spanText);
                            if (docs) {
                                let filteredHtml = `<h3>${data.token_level ? 'Token' : 'Span'}: "${spanText}"</h3>`;
                                if (docs.documents.length === 0) {
                                    filteredHtml += '<p>No matching documents found.</p>';
                                } else {
                                    docs.documents.forEach(d => {
                                        filteredHtml += `
                                            <div class="document relevance-${d.relevance_label.toLowerCase()}">
                                                <p><strong>Doc ${d.doc_id}</strong>: ${d.snippet}</p>
                                                <p>BM25 Score: ${d.bm25_score}, Relevance: ${d.relevance_label} (${d.relevance})</p>
                                                <button onclick="toggleFullDoc(${d.doc_id}, this, '${d.full_document}')">View Full Document</button>
                                                <div class="full-doc" id="full-doc-${d.doc_id}">${d.full_document}</div>
                                            </div>`;
                                    });
                                }
                                document.getElementById('documents').innerHTML = filteredHtml;
                            }
                        });
                    });
                }
            } catch (err) {
                document.getElementById('error').textContent = 'Error processing request.';
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });

        document.getElementById('search-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            document.getElementById('loading').style.display = 'block';
            document.getElementById('error').textContent = '';
            document.getElementById('output').style.display = 'none';
            document.getElementById('search-results').style.display = 'none';
            
            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('error').textContent = data.error;
                } else {
                    let searchHtml = '';
                    if (data.results.length === 0) {
                        searchHtml = '<p>No matching documents found.</p>';
                    } else {
                        data.results.forEach(r => {
                            searchHtml += `
                                <div class="document">
                                    <p><strong>Doc ${r.doc_id}</strong>: ${r.document}</p>
                                    <p>BM25 Score: ${r.bm25_score}</p>
                                </div>`;
                        });
                    }
                    document.getElementById('search-output').innerHTML = searchHtml;
                    document.getElementById('search-results').style.display = 'block';
                }
            } catch (err) {
                document.getElementById('error').textContent = 'Error processing search.';
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });

        document.getElementById('corpus-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData();
            formData.append('corpus_file', document.getElementById('corpus_file').files[0]);
            document.getElementById('loading').style.display = 'block';
            document.getElementById('error').textContent = '';
            
            try {
                const response = await fetch('/upload_corpus', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('error').textContent = data.error;
                } else {
                    alert(data.message);
                }
            } catch (err) {
                document.getElementById('error').textContent = 'Error uploading corpus.';
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });

        function toggleFullDoc(docId, button, fullDocHtml) {
            const fullDoc = document.getElementById(`full-doc-${docId}`);
            if (fullDoc.style.display === 'block') {
                fullDoc.style.display = 'none';
                button.textContent = 'View Full Document';
            } else {
                fullDoc.innerHTML = fullDocHtml;
                fullDoc.style.display = 'block';
                button.textContent = 'Hide Full Document';
            }
        }

        async function exportCSV() {
            if (!latestTraceData) {
                alert('No results to export.');
                return;
            }
            try {
                const response = await fetch('/export_csv', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(latestTraceData)
                });
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'trace_results.csv';
                a.click();
                window.URL.revokeObjectURL(url);
            } catch (err) {
                document.getElementById('error').textContent = 'Error exporting CSV.';
            }
        }

        async function exportPDF() {
            if (!latestTraceData) {
                alert('No results to export.');
                return;
            }
            try {
                const response = await fetch('/export_pdf', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(latestTraceData)
                });
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'trace_results.pdf';
                a.click();
                window.URL.revokeObjectURL(url);
            } catch (err) {
                document.getElementById('error').textContent = 'Error exporting PDF.';
            }
        }
    </script>
</body>
</html>
"""

# Save the template
import os
os.makedirs('templates', exist_ok=True)
with open('templates/index.html', 'w') as f:
    f.write(INDEX_HTML)

def main():
    parser = argparse.ArgumentParser(description="Enhanced OLMOTRACE with Flask UI")
    parser.add_argument("--corpus-file", type=str, help="Path to initial corpus file.")
    parser.add_argument("--port", type=int, default=5000, help="Port for Flask server.")
    parser.add_argument("--db-path", type=str, default="corpus.db", help="Path to SQLite database.")
    args = parser.parse_args()

    # Initialize tracer
    global tracer
    tracer = EnhancedOLMOTRACE(db_path=args.db_path)
    
    # Load initial corpus if provided
    if args.corpus_file:
        corpus = [tracer.preprocess_document(doc) for doc in load_corpus(args.corpus_file)]
        if corpus:
            tracer.save_corpus_to_db(corpus)
    
    # Start Flask server
    logger.info(f"Starting Flask server on port {args.port}")
    app.run(debug=False, port=args.port)

if __name__ == "__main__":
    main()