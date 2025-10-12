#!/bin/bash

echo "================================================================"
echo "   Materials Synthesis AI System (MatSyn) - Mac Launcher"
echo "================================================================"
echo ""

# Check if .env file exists
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check Neo4j password
if [ -z "$NEO4J_PASSWORD" ]; then
    echo "⚠️  NEO4J_PASSWORD not set."
    read -sp "Enter your Neo4j password: " NEO4J_PASS
    echo ""
    export NEO4J_PASSWORD="$NEO4J_PASS"
fi

# Check OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY not set!"
    echo ""
    echo "Please set your OpenAI API key:"
    echo "  export OPENAI_API_KEY='sk-your-key-here'"
    echo ""
    exit 1
fi

echo "✅ Neo4j Password: Set"
echo "✅ OpenAI API Key: Set"
echo ""

# Test Neo4j connection
echo "Testing Neo4j connection..."
python3 -c "
from neo4j import GraphDatabase
import os
try:
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD')
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        session.run('RETURN 1')
    driver.close()
    print('✅ Neo4j: Connected')
except Exception as e:
    print(f'❌ Neo4j: {e}')
    print('')
    print('Solutions:')
    print('1. Check Neo4j Desktop is running')
    print('2. Verify password is correct')
    print('3. Check bolt://localhost:7687 is accessible')
    exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
echo "Starting MatSyn Multi-Agent App..."
echo "Access at: http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop"
echo "================================================================"
echo ""

python3 multi_agent_app.py

