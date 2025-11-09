This project investigates the challenge of generating faithful, accurate, and salient natural language descriptions from multimodal data sources, specifically structured data and images. 
Multimodal data-to-text generation combines information from multiple input sources such as images, structured data graphs, or tables to produce meaningful textual descriptions. 
Recent advancements in LMMs (Large Multimodal Models) have enabled more fluent generation but still struggle with two main issues:
1. Hallucination: adding details not grounded in the input.
2. Lack of saliency: failing to highlight important or relevant visual features.

Real-world applications such as real-estate listings, product descriptions, journalism, and advertisement require text that is both faithful and attractive. 
This project studies the FaithD2T framework, which integrates a vision critic model and post‑hoc text correction to significantly improve faithfulness and saliency. 
The critic model acts as a verifier, ensuring the generated text corresponds closely to visual content while removing unimportant or fabricated details.

Methodology-

1. Data Preparation
The datasets contain structured data (knowledge graphs or attribute-value tables), images, and ground-truth advertisement text. 
Since ground-truth text contains hallucinated information, feature extraction and correction are required.

2. Initial Text Generation
Large multimodal models (MiniGPT‑4 or LLaVA) generate initial text using:
• Linearized structured data input
• Image-based feature extraction

3. Feature Extraction Using LLM
GPT‑3.5 extracts all features from the generated text sentence-by-sentence.

4. Vision Critic Model (BLIP‑2 with LoRA)
A fine-tuned BLIP‑2 critic model performs:
• Feature classification (salient, non-salient, hallucinated)
• Salient feature suggestion (missing features)

5. Post‑hoc Correction Pipeline
GPT‑3.5 revises the initial text by:
• Removing hallucinated and non-salient features
• Appending missing salient features

6. Evaluation
Experiments measure improvements in faithfulness and saliency using:
• BLEU, ROUGE, METEOR, BERTScore for text similarity
• CLIPScore for text-image alignment
