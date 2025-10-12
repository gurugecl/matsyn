# Materials Synthesis AI - Mac Setup

## 📦 Files You Need

Transfer these files from Linux to your Mac:

### Core Application Files
- `multi_agent_app.py` - Web interface
- `multi_agent_synthesizer.py` - Agent logic  
- `materials_agent.py` - Database interface

### Configuration Files
- `requirements_agent.txt` - Python dependencies
- `start_mac.sh` - Startup script
- `example.env` - Environment template

### Documentation
- `MAC_SETUP.md` - This file
- `MULTI_AGENT_GUIDE.md` - System documentation (optional)

## 🚀 Quick Start

### Step 1: Copy Files

```bash
# On your Mac, create a folder
mkdir ~/MaterialsSynthesisAI
cd ~/MaterialsSynthesisAI

# Copy the 6 required files here
```

### Step 2: Install Python Packages

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements_agent.txt
```

### Step 3: Configure Environment

```bash
# Copy the template
cp example.env .env

# Edit with your values
nano .env
```

Fill in:
- `NEO4J_PASSWORD=your_actual_password`
- `OPENAI_API_KEY=sk-your-actual-key`

### Step 4: Start the App

```bash
chmod +x start_mac.sh
./start_mac.sh
```

### Step 5: Open Browser

Go to: **http://localhost:7860**

## ✅ System Requirements

- **macOS** 10.14+
- **Python** 3.8 or higher
- **Neo4j Desktop** running locally
- **OpenAI API Key** (free tier works)
- **~200 MB** disk space for dependencies

## 🔧 Common Issues

### "Python not found"
```bash
# Install Python via Homebrew
brew install python3
```

### "Neo4j connection failed"
1. Open Neo4j Desktop
2. Start your database
3. Check it's running on `bolt://localhost:7687`
4. Verify your password

### "Port 7860 already in use"
```bash
# Kill the existing process
lsof -ti:7860 | xargs kill -9
./start_mac.sh
```

### "Module not found"
```bash
# Reinstall dependencies
pip install --upgrade -r requirements_agent.txt
```

## 📊 What This Does

This system uses **3 AI agents** to generate materials synthesis recommendations:

1. **Agent 1 (GPT-4o-mini)**: Analyzes your target material structure
2. **Agent 2 (Neo4j)**: Queries database for similar synthesis procedures
3. **Agent 3 (GPT-4o-mini)**: Generates recommendations with stoichiometry

**Database**: 35,675 synthesis procedures from scientific literature

**Output**: Complete synthesis procedure with:
- Proposed precursors and stoichiometry
- Temperature and time conditions  
- Step-by-step procedure
- DOI citations to real papers
- Downloadable text report

## 💡 Example Usage

Try these materials:
- `CeMnO₃` - Perovskite
- `LiMn₂O₄` - Spinel
- `TiO₂` - Simple oxide
- `Ba₀.₅Sr₀.₅TiO₃` - Doped perovskite

## 🆘 Need Help?

Check the terminal output for detailed error messages and suggestions.

