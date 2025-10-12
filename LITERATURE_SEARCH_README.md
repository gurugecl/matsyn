# Literature Search Integration - MatSyn

## Overview

MatSyn now integrates with the **FutureHouse API** to automatically search scientific literature when a synthesis procedure is not found in the local Neo4j database. This feature enables the system to continuously expand its knowledge base with the latest research.

## How It Works

### Workflow

1. **User requests synthesis** for a material (e.g., `LiCr1.5Mn0.5O4`)
2. **Agent 1 (GPT-4o-mini)** analyzes the material and suggests similar compounds
3. **Agent 2 (Database Query)** searches the Neo4j knowledge graph:
   - First tries exact match
   - Then searches for similar materials
   - **NEW:** If no results found, searches scientific literature via FutureHouse API
4. **Literature Search Agent** (if no database results):
   - Queries FutureHouse CROW API for synthesis procedures
   - Parses the literature response using GPT-4o-mini
   - Extracts structured synthesis data (precursors, operations, conditions, DOI)
   - **Automatically adds the synthesis to Neo4j database**
   - Returns the newly added synthesis for recommendation
5. **Agent 3 (Synthesis Planner)** generates recommendations based on all available data

### What Gets Added to Database

When a synthesis is found in literature, the following information is extracted and stored:

```json
{
  "doi": "paper DOI or 'literature_search'",
  "paragraph_string": "description of synthesis",
  "reaction_string": "precursors -> target_material",
  "target": {
    "material_formula": "target formula",
    "material_name": ""
  },
  "precursors": [
    {
      "material_formula": "precursor formula",
      "material_name": ""
    }
  ],
  "operations": [
    {
      "type": "HeatingOperation | MixingOperation | etc.",
      "conditions": {
        "temperature": {"values": [900], "units": "°C"},
        "time": {"values": [2], "units": "h"},
        "atmosphere": ["air"]
      },
      "string": "operation description"
    }
  ],
  "type": "synthesis type (solid-state, sol-gel, etc.)"
}
```

## Setup

### 1. Install Requirements

The `futurehouse-client` package is now included in `requirements_agent.txt`:

```bash
pip install futurehouse-client>=0.1.0
```

Or reinstall all requirements:

```bash
pip install -r requirements_agent.txt
```

### 2. Get FutureHouse API Key

1. Visit [FutureHouse Platform](https://futurehouse.ai/)
2. Create an account or log in
3. Go to your profile page
4. Generate an API key

### 3. Configure Environment

Add your FutureHouse API key to `.env`:

```bash
FUTUREHOUSE_API_KEY=your_futurehouse_api_key_here
```

Or export it in your shell:

```bash
export FUTUREHOUSE_API_KEY="your_key_here"
```

**Note:** Literature search is **optional**. The system works without it, but won't be able to search literature for missing materials.

## Features

### Automatic Knowledge Base Expansion

- **Persistent Storage**: Synthesis procedures found in literature are permanently added to your Neo4j database
- **Future Queries**: Once added, the synthesis is available for all future queries without re-searching
- **Growing Database**: Your knowledge base automatically expands as users query new materials

### Smart Literature Search

- **Only When Needed**: Literature search only triggers when database has no results
- **High-Quality Sources**: FutureHouse API searches peer-reviewed journals
- **Structured Extraction**: Uses GPT-4o-mini to parse and structure the literature data
- **Citation Tracking**: DOIs are preserved for reference

### Seamless Integration

- **No User Intervention**: Literature search happens automatically in the background
- **Same Interface**: Results from literature look identical to database results
- **Priority System**: Literature results are integrated with priority-based recommendations

## Modified Files

### 1. `materials_agent.py`
- Added `add_synthesis_from_literature()` method
- Handles inserting new synthesis data into Neo4j graph

### 2. `multi_agent_synthesizer.py`
- Added `LiteratureSearchAgent` class for FutureHouse integration
- Modified `DatabaseQueryAgent` to call literature search when needed
- Updated `MultiAgentSynthesizer` to initialize literature agent
- Added automatic database insertion after successful literature search

### 3. `multi_agent_app.py`
- Added `FUTUREHOUSE_API_KEY` environment variable loading
- Updated synthesizer initialization to include FutureHouse key

### 4. `example.env`
- Added `FUTUREHOUSE_API_KEY` configuration option

### 5. `requirements_agent.txt`
- Added `futurehouse-client>=0.1.0` dependency

## Usage Example

### Scenario: Material Not in Database

```python
# User queries: "NaMnF2"
# Database search: No results found
# System automatically:
# 1. Searches scientific literature via FutureHouse
# 2. Finds synthesis paper with detailed procedure
# 3. Extracts structured data
# 4. Adds to Neo4j database
# 5. Returns recommendation based on literature
```

### Terminal Output

```
[Agent 2] Querying knowledge graph...

  Querying exact match: NaMnF2 (normalized: NaMnF2)
    ⚠️ No exact match, trying pattern search with 'NaMnF2'...
    → Found 0 candidate materials
    
    📚 No database results found. Searching scientific literature...
  
  📚 Searching scientific literature for NaMnF2...
    → Querying FutureHouse API...
    ✓ Literature search completed
    ✓ Parsed synthesis data for NaMnF2
    ✓ Added synthesis for NaMnF2 (ID: a3f8d2c9e1b4)
    → Re-querying database for newly added synthesis...
    ✓ Added synthesis from literature to database
```

## Benefits

1. **Self-Improving System**: Database grows automatically with each query
2. **Latest Research**: Access to cutting-edge synthesis methods from recent publications
3. **Reduced Manual Effort**: No need to manually add synthesis data
4. **Always-On Discovery**: Continuously discovers new synthesis routes
5. **Citation Preservation**: Maintains links to original research papers

## Limitations

- Requires FutureHouse API key (optional feature)
- API rate limits may apply depending on your FutureHouse plan
- Literature search adds 10-30 seconds to query time when triggered
- Quality depends on FutureHouse's ability to find relevant papers

## Future Enhancements

Potential improvements:
- Cache literature search results to avoid duplicate API calls
- Add user confirmation before adding to database
- Support for batch literature imports
- Integration with additional literature databases
- Quality scoring for literature-derived syntheses

## Troubleshooting

### FutureHouse Client Not Available

```
⚠️ FutureHouse client not installed. Literature search will be disabled.
```

**Solution**: `pip install futurehouse-client`

### FutureHouse API Not Configured

```
⚠️ FutureHouse API not configured - literature search disabled
```

**Solution**: Add `FUTUREHOUSE_API_KEY` to your `.env` file

### Literature Search Fails

```
✗ Literature search error: [error message]
```

**Solutions**:
- Check API key is valid
- Verify network connectivity
- Check FutureHouse API status
- Review API rate limits

## References

- [FutureHouse Platform](https://futurehouse.ai/)
- [FutureHouse API Documentation](https://futurehouse.gitbook.io/futurehouse-cookbook/futurehouse-client)
- [PaperQA2 (CROW Agent)](https://github.com/Future-House/paper-qa)

---

**MatSyn - Materials Synthesis AI System**  
Integrated Literature Search | Self-Expanding Knowledge Base

