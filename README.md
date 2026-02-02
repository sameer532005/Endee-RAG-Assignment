# Intelligent Document Question Answering System  
*(Using Retrieval-Augmented Generation – RAG)*

## Project Overview
In today’s digital world, large volumes of textual data are generated in the form of documents, reports, and notes. Manually searching through these documents is inefficient and time-consuming.

This project implements an Intelligent Document Question Answering System that allows users to ask natural language questions and retrieve relevant answers from documents using semantic similarity instead of keyword matching. The system follows the Retrieval-Augmented Generation (RAG) approach.

## Problem Statement
Traditional document search systems rely on keyword matching, which often fails to capture the semantic meaning of queries. This leads to inaccurate or incomplete results.

The goal of this project is to design a semantic document retrieval system that understands user intent and retrieves the most relevant information efficiently.

## System Design / Technical Approach
The system works in two phases:

### Document Ingestion
- Reads documents from a text file
- Converts text into vector embeddings using a pre-trained transformer model
- Stores embeddings and text locally using Endee-style vector storage

### Query Processing
- Converts user queries into embeddings
- Calculates cosine similarity with stored embeddings
- Retrieves the most relevant document as the answer

## Use of Endee
Endee is used as a local vector storage mechanism in this project. It stores:
- Document embeddings
- Corresponding document text

Endee enables fast similarity-based retrieval and demonstrates how vector databases are used in real-world RAG systems.

## Technologies Used
- Python
- SentenceTransformers
- NumPy
- Pre-trained model: all-MiniLM-L6-v2
- Endee (local vector storage simulation)

## Project Structure
```
endee-rag-document-qa/
│
├── data/
│   ├── documents.txt
│   ├── embeddings.npy
│   └── texts.npy
│
├── src/
│   ├── ingest.py
│   └── query.py
│
├── requirements.txt
└── README.md
```

## Setup Instructions
1. Extract or clone the project
2. (Optional) Create a virtual environment
3. Install dependencies using:
   pip install -r requirements.txt

## Execution Instructions
1. Add content to data/documents.txt
2. Run ingestion:
   python src/ingest.py
3. Run query:
   python src/query.py
4. Enter a question to retrieve relevant content

## Future Enhancements
- Integration with large language models
- PDF and Word document support
- Top-K document retrieval
- Web interface using Flask or Streamlit

## Conclusion
This project demonstrates a simple and effective implementation of a semantic document question answering system using RAG principles. It highlights the importance of embeddings and vector similarity in modern information retrieval systems.
