# Sekai Story Recommendation System

## Local Setup

1. **Prerequisites**  
   - Python 3.9+  
   - Docker Desktop (for Redis)  
   - OpenAI API key in `.env` (e.g. `OPENAI_API_KEY=sk-...`)

2. **Clone & Install**  
   ```bash
   git clone https://github.com/evoked121/recommend-task.git
   cd recommend-system

   python3 -m venv venv
   source venv/bin/activate

   pip install -r requirements.txt
   
   docker pull redis:latest
   docker run -d --name local-redis -p 6379:6379 redis:latest

3. **Run**
   ```bash
   python main.py