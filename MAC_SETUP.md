# Setup on Mac

## Required Files

Copy these files to your Mac:

```
📁 Your Mac Folder/
├── multi_agent_app.py          # Gradio web interface
├── multi_agent_synthesizer.py  # Multi-agent system logic
├── materials_agent.py           # Neo4j database interface
├── requirements_agent.txt       # Python dependencies
└── start_mac.sh                 # Mac startup script
```

## Prerequisites on Mac

1. **Python 3.8+** (check with `python3 --version`)
2. **Neo4j Database** running (you already have this ✓)
3. **OpenAI API Key** (from https://platform.openai.com/api-keys)

## Setup Steps

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements_agent.txt
```

### 2. Set Environment Variables

Create a `.env` file in the same folder:

```bash
# Neo4j Connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# OpenAI API Key
OPENAI_API_KEY=sk-your-openai-api-key-here
```

Or export them directly:

```bash
export NEO4J_PASSWORD="your_password"
export OPENAI_API_KEY="sk-your-key-here"
```

### 3. Run the App

```bash
# Option 1: Use the start script
chmod +x start_mac.sh
./start_mac.sh

# Option 2: Run directly
python3 multi_agent_app.py
```

### 4. Access the Web Interface

Open your browser to: **http://localhost:7860**

## Troubleshooting

**Port already in use?**
```bash
# Kill any existing process
lsof -ti:7860 | xargs kill -9
```

**Neo4j connection failed?**
- Make sure Neo4j is running on your Mac
- Check the bolt port (default: 7687)
- Verify your password in Neo4j Desktop

**Missing dependencies?**
```bash
pip install --upgrade -r requirements_agent.txt
```

## Quick Test

After starting, test with:
- `CeMnO₃` - Perovskite
- `LiMn2O4` - Spinel
- `TiO2` - Simple oxide

The system will query your Neo4j database and generate synthesis recommendations!

