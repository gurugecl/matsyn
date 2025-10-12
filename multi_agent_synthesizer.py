"""
Multi-Agent Materials Synthesis System
Uses multiple LLM agents with prioritized knowledge graph retrieval
"""

import os
import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from materials_agent import MaterialsSynthesisAgent
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# FutureHouse client import
try:
    from futurehouse_client import FutureHouseClient, JobNames
    FUTUREHOUSE_AVAILABLE = True
except ImportError:
    FUTUREHOUSE_AVAILABLE = False
    print("FutureHouse client not installed. Literature search will be disabled.")


@dataclass
class SynthesisProcedure:
    """Structured synthesis procedure from database"""
    material: str
    reaction_string: str
    synthesis_type: str
    precursors: List[str]
    precursor_amounts: Dict
    operations: List[Dict]
    doi: str
    priority: int


class MaterialAnalysisAgent:
    """Agent 1: Analyzes material and suggests similar compounds"""
    
    ANALYSIS_PROMPT = """I want to synthesize {formula}. I have a database of synthetic routes for well known materials. To help design a query for this database:

1) What is your best estimate of the structure type of this material (Spinel, layered oxide, perovskite, etc.)
2) What element types are in the formula (alkali, early transition metal, halogen, etc.)

Based on the answers to these questions, please list the formulas of the most similar well known materials you can think of. Group them in priority order by similarity.

Format your response as:

**Structure Type:** [your prediction]

**Element Types:** [list of types]

**Priority 1 (Most Similar):**
- Material 1
- Material 2
- Material 3

**Priority 2 (Similar):**
- Material 4
- Material 5

**Priority 3 (Somewhat Similar):**
- Material 6
- Material 7

Be concise and list only formulas in the priority groups."""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def analyze(self, formula: str) -> Dict[str, any]:
        """Analyze material and return structure + similar materials"""
        prompt = self.ANALYSIS_PROMPT.format(formula=formula)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Parse response
        text = response.content
        
        # Extract structure type
        structure_match = re.search(r'\*\*Structure Type:\*\*\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        structure = structure_match.group(1).strip() if structure_match else "Unknown"
        
        # Extract element types
        elements_match = re.search(r'\*\*Element Types:\*\*\s*(.+?)(?:\n\n|\*\*)', text, re.DOTALL | re.IGNORECASE)
        elements = elements_match.group(1).strip() if elements_match else "Unknown"
        
        # Extract priority groups
        priorities = {}
        for i in range(1, 4):
            pattern = rf'\*\*Priority {i}[^:]*:\*\*\s*((?:[-•]\s*.+?\n)+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                materials = re.findall(r'[-•]\s*(.+)', match.group(1))
                priorities[i] = [m.strip() for m in materials if m.strip()]
        
        return {
            'structure_type': structure,
            'element_types': elements,
            'priorities': priorities,
            'full_response': text
        }


class LiteratureSearchAgent:
    """Agent for searching scientific literature using FutureHouse API"""
    
    def __init__(self, futurehouse_api_key: str = None, llm: ChatOpenAI = None):
        self.api_key = futurehouse_api_key or os.getenv("FUTUREHOUSE_API_KEY")
        self.llm = llm
        self.client = None
        
        print(f"    DEBUG LiteratureSearchAgent: FUTUREHOUSE_AVAILABLE={FUTUREHOUSE_AVAILABLE}")
        print(f"    DEBUG LiteratureSearchAgent: api_key={'SET (len=' + str(len(self.api_key)) + ')' if self.api_key else 'NOT SET'}")
        
        if FUTUREHOUSE_AVAILABLE and self.api_key:
            try:
                print(f"    DEBUG: Creating FutureHouseClient...")
                self.client = FutureHouseClient(api_key=self.api_key)
                print(f"    FutureHouse client initialized successfully")
            except Exception as e:
                print(f"    FutureHouse initialization failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            if not FUTUREHOUSE_AVAILABLE:
                print(f"    futurehouse-client package not installed")
            if not self.api_key:
                print(f"    No FUTUREHOUSE_API_KEY provided")
    
    def search_literature(self, formula: str) -> Optional[Dict]:
        """Search literature for synthesis of a material"""
        if not self.client:
            print(f"    FutureHouse client not available for literature search")
            return None
        
        try:
            print(f"\n  Searching scientific literature for {formula}...")
            
            query = f"""Search the academic literature for a detailed synthesis procedure of {formula}.
            
Focus on peer-reviewed journals with high-quality synthesis methods. If you find a paper with 
a detailed synthetic method, extract:
1. The target material formula
2. All precursor materials and their formulas
3. The synthesis type (e.g., solid-state, sol-gel, hydrothermal, etc.)
4. Step-by-step operations (heating, mixing, calcination, etc.)
5. Reaction conditions (temperature, time, atmosphere)
6. The DOI or citation of the paper

Format the response as a structured synthesis procedure."""
            
            # Use CROW for fast literature search
            task_data = {
                "name": JobNames.CROW,
                "query": query
            }
            
            print(f"    Querying FutureHouse API...")
            response = self.client.run_tasks_until_done(task_data)
            
            if hasattr(response, 'answer') and response.answer:
                print(f"    Literature search completed")
                return self._parse_literature_response(formula, response.answer, response)
            else:
                print(f"    No answer from literature search")
                return None
                
        except Exception as e:
            print(f"    - Literature search error: {e}")
            return None
    
    def _parse_literature_response(self, formula: str, answer: str, response) -> Optional[Dict]:
        """Parse FutureHouse response and convert to synthesis data format"""
        try:
            # Use LLM to structure the response into our format
            if not self.llm:
                return None
            
            parsing_prompt = f"""Given the following literature search result about synthesizing {formula}, 
extract and structure the synthesis information into JSON format.

Literature Search Result:
{answer}

Extract the following information and format as JSON:
{{
  "doi": "extracted DOI or 'literature_search'",
  "paragraph_string": "brief description of the synthesis",
  "reaction_string": "precursors -> {formula}",
  "target": {{
    "material_formula": "{formula}",
    "material_name": ""
  }},
  "precursors": [
    {{
      "material_formula": "precursor formula",
      "material_name": ""
    }}
  ],
  "operations": [
    {{
      "type": "operation type (e.g., HeatingOperation, MixingOperation)",
      "conditions": {{
        "temperature": {{"values": [temp], "units": "°C"}},
        "time": {{"values": [time], "units": "h"}},
        "atmosphere": ["air or other"]
      }},
      "string": "brief description"
    }}
  ],
  "type": "synthesis type (e.g., solid-state, sol-gel, hydrothermal)"
}}

Only output valid JSON. If information is missing, use null or empty arrays."""
            
            llm_response = self.llm.invoke([HumanMessage(content=parsing_prompt)])
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response.content, re.DOTALL)
            if json_match:
                synthesis_data = json.loads(json_match.group())
                print(f"    Parsed synthesis data for {formula}")
                return synthesis_data
            else:
                print(f"    Could not parse synthesis data from LLM response")
                return None
                
        except Exception as e:
            print(f"    Error parsing literature response: {e}")
            return None


class DatabaseQueryAgent:
    """Agent 2: Queries Neo4j knowledge graph"""
    
    def __init__(self, neo4j_agent: MaterialsSynthesisAgent, literature_agent: Optional[LiteratureSearchAgent] = None):
        self.agent = neo4j_agent
        self.literature_agent = literature_agent
    
    def normalize_formula(self, formula: str) -> str:
        """Convert Unicode subscripts to regular numbers"""
        subscript_map = {
            '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
            '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
        }
        for sub, num in subscript_map.items():
            formula = formula.replace(sub, num)
        return formula
    
    def query_material(self, formula: str) -> List[Dict]:
        """Query database for a specific material (flexible search)"""
        results = []
        
        # Normalize formula (convert Unicode subscripts to numbers)
        normalized_formula = self.normalize_formula(formula)
        
        try:
            # Try exact match first
            routes = self.agent.find_synthesis_routes(normalized_formula, limit=5)
            print(f"    Found {len(routes)} exact matches for {formula}")
            
            for route in routes:
                results.append({
                    'material': route.get('reaction', '').split('->')[-1].strip() or formula,
                    'reaction_string': route['reaction'],
                    'synthesis_type': route['type'],
                    'precursors': route['precursors'],
                    'operations': route.get('operations', []),
                    'paragraph': route.get('paragraph', ''),
                    'recipe_id': route.get('recipe_id', ''),
                    'doi': route['doi']
                })
        except Exception as e:
            print(f"    Exact match failed for {formula}: {e}")
        
        # If no exact match, try pattern search
        if not results:
            try:
                # Extract elements from formula for broader search
                import re
                elements = re.findall(r'([A-Z][a-z]?)', normalized_formula)
                search_term = normalized_formula[:6] if len(normalized_formula) >= 6 else normalized_formula
                
                print(f"    No exact match, trying pattern search with '{search_term}'...")
                
                # Search for materials containing these elements
                materials = self.agent.search_materials(search_term, limit=30)  # Use first part of formula
                print(f"    Found {len(materials)} candidate materials")
                
                # Try to find synthesis routes for similar materials
                for material in materials[:5]:  # Try more materials
                    try:
                        routes = self.agent.find_synthesis_routes(material, limit=2)
                        if routes:
                            print(f"      {material}: {len(routes)} route(s)")
                        for route in routes:
                            results.append({
                                'material': material,
                                'reaction_string': route['reaction'],
                                'synthesis_type': route['type'],
                                'precursors': route['precursors'],
                                'operations': route.get('operations', []),
                                'paragraph': route.get('paragraph', ''),
                                'recipe_id': route.get('recipe_id', ''),
                                'doi': route['doi']
                            })
                        
                        if len(results) >= 3:  # Get at least a few examples
                            break
                    except Exception as e2:
                        print(f"      {material}: {e2}")
                        continue
            except Exception as e:
                print(f"    Pattern search error: {e}")
        
        # If still no results, try literature search
        print(f"    DEBUG: results={len(results)}, literature_agent={self.literature_agent is not None}")
        if not results and self.literature_agent:
            print(f"    No database results found. Searching scientific literature...")
            try:
                lit_data = self.literature_agent.search_literature(normalized_formula)
                
                if lit_data:
                    # Add to database
                    success = self.agent.add_synthesis_from_literature(lit_data)
                    
                    if success:
                        # Query the newly added synthesis
                        print(f"    Re-querying database for newly added synthesis...")
                        routes = self.agent.find_synthesis_routes(normalized_formula, limit=1)
                        
                        if routes:
                            for route in routes:
                                results.append({
                                    'material': normalized_formula,
                                    'reaction_string': route['reaction'],
                                    'synthesis_type': route['type'],
                                    'precursors': route['precursors'],
                                    'operations': route.get('operations', []),
                                    'paragraph': route.get('paragraph', ''),
                                    'recipe_id': route.get('recipe_id', ''),
                                    'doi': route['doi'],
                                    'from_literature': True
                                })
                            print(f"    Added synthesis from literature to database")
                    else:
                        print(f"    Could not add literature synthesis to database")
                else:
                    print(f"    No synthesis found in literature")
                    
            except Exception as e:
                print(f"    Literature search error: {e}")
        
        return results[:5]  # Limit to 5 procedures per material
    
    def query_by_priority(self, target: str, priorities: Dict[int, List[str]]) -> Dict[int, List[Dict]]:
        """Query database for target + priority materials"""
        results = {}
        
        # Normalize target formula
        normalized_target = self.normalize_formula(target)
        
        # Priority 0: Exact match
        print(f"\n  Querying exact match: {target} (normalized: {normalized_target})")
        exact = self.query_material(target)
        if exact:
            results[0] = exact
            print(f"    Found {len(exact)} routes")
        
        # Priority 1, 2, 3: Similar materials
        for priority, materials in sorted(priorities.items()):
            print(f"\n  Querying Priority {priority} materials...")
            priority_results = []
            for material in materials[:3]:  # Limit to top 3 per priority
                print(f"    - {material}")
                routes = self.query_material(material)
                priority_results.extend(routes)
            
            if priority_results:
                results[priority] = priority_results
                print(f"    Found {len(priority_results)} total routes")
        
        return results


class SynthesisRecommendationAgent:
    """Agent 3: Synthesizes recommendations from database results"""
    
    RECOMMENDATION_PROMPT = """I'm trying to synthesize {target_formula}.

Here is a collection of actual material syntheses from scientific literature for {similarity_level} compounds:

{synthesis_data}

TASK: Based on these database examples, propose a synthesis procedure for {target_formula}.

Your response must have TWO sections:

SECTION 1: PROPOSED SYNTHESIS FOR {target_formula}
Based on the most relevant examples above, propose:
1. **Recommended Method:** Which synthesis method from the examples is most appropriate
2. **Proposed Stoichiometry:** Calculate the molar ratios of precursors needed for {target_formula}
   - List each precursor with its chemical formula and molar amount
   - Show the balanced equation: precursors → {target_formula}
3. **Proposed Conditions:** Based on the temperature/time ranges from similar examples
   - Temperature: [specify value/range in °C]
   - Time: [specify value/range in hours]
   - Atmosphere: [if specified in examples]
   - Other conditions: [mixing media, etc.]
4. **Proposed Procedure:** Step-by-step based on operations from examples
   - List each operation (mixing, heating, cooling, etc.)
   - Include conditions for each step from the database

SECTION 2: DATABASE EVIDENCE
5. **Key Citations:** DOIs of the most relevant procedures used
6. **Reasoning:** 
   - Which example is most similar and why
   - How you adapted the conditions/precursors for {target_formula}
   - What assumptions were made

RULES:
- Stoichiometry must be chemically balanced for {target_formula}
- Proposed conditions must be based on ranges/values from the database examples
- Clearly distinguish between "proposed for target" vs "from database example"
- If adapting from examples, explain the adaptation logic
"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def format_synthesis_data(self, procedures: List[Dict]) -> str:
        """Format database results for LLM"""
        import json
        text = ""
        for i, proc in enumerate(procedures, 1):
            text += f"\n### Example {i}: {proc['material']}\n"
            text += f"**Synthesis Type:** {proc['synthesis_type']}\n"
            text += f"**Reaction:** {proc['reaction_string']}\n"
            text += f"**Precursors:** {', '.join(proc['precursors'])}\n"
            
            # Add operations if available
            if proc.get('operations'):
                text += f"**Operations:**\n"
                for op in proc['operations']:
                    if op.get('type'):
                        text += f"  - {op['type']}"
                        if op.get('string'):
                            text += f": {op['string']}"
                        if op.get('conditions'):
                            try:
                                conditions = json.loads(op['conditions']) if isinstance(op['conditions'], str) else op['conditions']
                                cond_parts = []
                                for key, val in conditions.items():
                                    if val and str(val).strip():
                                        cond_parts.append(f"{key}={val}")
                                if cond_parts:
                                    text += f" ({', '.join(cond_parts)})"
                            except:
                                pass
                        text += "\n"
            
            # Add paragraph if available
            if proc.get('paragraph'):
                text += f"**Original Text:** {proc['paragraph'][:500]}...\n"
            
            text += f"**DOI:** {proc['doi']}\n"
        
        return text
    
    def recommend(self, target: str, priority: int, procedures: List[Dict]) -> str:
        """Generate synthesis recommendation for a priority group"""
        
        if priority == 0:
            similarity = "exact match"
        elif priority == 1:
            similarity = "highly similar"
        elif priority == 2:
            similarity = "similar"
        else:
            similarity = "related"
        
        synthesis_data = self.format_synthesis_data(procedures)
        
        prompt = self.RECOMMENDATION_PROMPT.format(
            target_formula=target,
            similarity_level=similarity,
            synthesis_data=synthesis_data
        )
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content


class MultiAgentSynthesizer:
    """Orchestrates the multi-agent synthesis recommendation system"""
    
    def __init__(self, neo4j_agent: MaterialsSynthesisAgent, api_key: str = None, model: str = "gpt-4o-mini", 
                 futurehouse_api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.1,
            openai_api_key=self.api_key
        )
        
        # Initialize literature search agent (optional)
        futurehouse_key = futurehouse_api_key or os.getenv("FUTUREHOUSE_API_KEY")
        print(f"  DEBUG: FUTUREHOUSE_AVAILABLE={FUTUREHOUSE_AVAILABLE}, futurehouse_key={'SET' if futurehouse_key else 'NOT SET'}")
        if futurehouse_key and FUTUREHOUSE_AVAILABLE:
            print("  Initializing FutureHouse literature search...")
            self.literature_agent = LiteratureSearchAgent(futurehouse_key, self.llm)
            print(f"  Literature agent initialized: {self.literature_agent.client is not None}")
        else:
            if not FUTUREHOUSE_AVAILABLE:
                print("  FutureHouse client not installed - literature search disabled")
            elif not futurehouse_key:
                print("  FutureHouse API key not set - literature search disabled")
            self.literature_agent = None
        
        # Initialize agents
        self.analysis_agent = MaterialAnalysisAgent(self.llm)
        self.query_agent = DatabaseQueryAgent(neo4j_agent, self.literature_agent)
        self.recommendation_agent = SynthesisRecommendationAgent(self.llm)
    
    def synthesize(self, formula: str) -> Dict:
        """Complete synthesis recommendation workflow"""
        
        print(f"\n{'='*70}")
        print(f"Multi-Agent Synthesis Recommendation for {formula}")
        print(f"{'='*70}")
        
        # Step 1: Analyze material
        print("\n[Agent 1] Analyzing material and identifying similar compounds...")
        analysis = self.analysis_agent.analyze(formula)
        
        print(f"\n  Structure Type: {analysis['structure_type']}")
        print(f"  Element Types: {analysis['element_types']}")
        print(f"  Priorities identified: {list(analysis['priorities'].keys())}")
        
        # Step 2: Query database
        print("\n[Agent 2] Querying knowledge graph...")
        database_results = self.query_agent.query_by_priority(
            formula,
            analysis['priorities']
        )
        
        if not database_results:
            print("\n  No synthesis procedures found in database")
            return {
                'analysis': analysis,
                'database_results': {},
                'recommendations': {}
            }
        
        # Step 3: Generate recommendations for each priority
        print("\n[Agent 3] Generating synthesis recommendations...")
        recommendations = {}
        
        for priority in sorted(database_results.keys()):
            procedures = database_results[priority]
            print(f"\n  Processing Priority {priority} ({len(procedures)} procedures)...")
            
            recommendation = self.recommendation_agent.recommend(
                formula,
                priority,
                procedures
            )
            
            recommendations[priority] = {
                'procedures': procedures,
                'recommendation': recommendation
            }
        
        return {
            'analysis': analysis,
            'database_results': database_results,
            'recommendations': recommendations
        }
    
    def format_report(self, result: Dict, target_formula: str) -> str:
        """Format complete synthesis report"""
        
        report = f"# Synthesis Recommendation for {target_formula}\n\n"
        
        # Analysis
        report += f"## Initial Analysis\n\n"
        report += f"**Predicted Structure:** {result['analysis']['structure_type']}\n\n"
        report += f"**Element Classification:** {result['analysis']['element_types']}\n\n"
        
        # Show priority groups
        report += f"## Similar Materials Identified\n\n"
        for priority, materials in sorted(result['analysis']['priorities'].items()):
            report += f"**Priority {priority}:** {', '.join(materials[:3])}\n\n"
        
        # Recommendations by priority
        report += f"## Synthesis Recommendations\n\n"
        
        priorities = sorted(result['recommendations'].keys())
        
        for priority in priorities:
            rec_data = result['recommendations'][priority]
            
            if priority == 0:
                report += f"### Based on Exact Match\n\n"
            elif priority == 1:
                report += f"### Priority 1: Most Similar Materials\n\n"
            elif priority == 2:
                report += f"### Priority 2: Similar Materials\n\n"
            else:
                report += f"### Priority {priority}: Related Materials\n\n"
            
            report += rec_data['recommendation'] + "\n\n"
            report += "---\n\n"
        
        # Database stats
        total_procedures = sum(len(r['procedures']) for r in result['recommendations'].values())
        report += f"\n*Analysis based on {total_procedures} synthesis procedures from literature database.*\n"
        
        return report


def main():
    """Demo the multi-agent system"""
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "ai4science")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set!")
        print("Please run: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Initialize
    neo4j_agent = MaterialsSynthesisAgent(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    synthesizer = MultiAgentSynthesizer(neo4j_agent, OPENAI_API_KEY)
    
    # Test
    formula = "LiCr1.5Mn0.5O4"
    result = synthesizer.synthesize(formula)
    
    # Print report
    print("\n" + "="*70)
    print("FINAL REPORT")
    print("="*70 + "\n")
    
    report = synthesizer.format_report(result, formula)
    print(report)
    
    neo4j_agent.close()


if __name__ == "__main__":
    main()

