# Item–Item Anime Recommender System

## Overview
This project implements an **item–item content-based recommendation system** for anime titles using genre metadata and rating information.  
The system addresses the **cold-start problem** by relying solely on item attributes, without requiring user interaction data.

## Approach
- Genre information encoded using **TF-IDF**
- Ratings normalized using **Min–Max scaling**
- Features combined into a unified representation
- **Cosine similarity** used to compute item–item similarity
- Self-similarity explicitly removed
- Top-N similar anime returned per query item

## Dataset
- ~12,300 anime titles
- Attributes include genre, rating, popularity, and type
- >98% data retained after preprocessing

## How to Run
```bash
pip install -r requirements.txt
