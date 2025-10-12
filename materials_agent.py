"""
Materials Synthesis Neo4j Agent
Query the materials synthesis knowledge graph using natural language
"""

import os
from typing import Any, Dict, List
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "ai4science")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class MaterialsSynthesisAgent:
    """Agent for querying materials synthesis database"""
    
    def __init__(self, uri: str, user: str, password: str, llm_api_key: str = None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.llm_api_key = llm_api_key
        
        # Common query templates
        self.query_templates = {
            "find_synthesis": """
                MATCH (target:Material {formula: $formula})<-[:PRODUCES]-(recipe:SynthesisRecipe)
                MATCH (precursor:Material)-[:PRECURSOR_OF]->(recipe)
                OPTIONAL MATCH (recipe)-[rel:HAS_OPERATION]->(op:Operation)
                WITH recipe, target, 
                     collect(DISTINCT precursor.formula) as precursors,
                     collect(DISTINCT {type: op.type, conditions: op.conditions, string: op.string, sequence: rel.sequence}) as operations
                RETURN recipe.reaction_string as reaction, 
                       recipe.type as type,
                       recipe.recipe_id as recipe_id,
                       recipe.paragraph_string as paragraph,
                       precursors,
                       operations,
                       recipe.doi as doi
                ORDER BY recipe.recipe_id
                LIMIT $limit
            """,
            "find_products": """
                MATCH (precursor:Material {formula: $formula})-[:PRECURSOR_OF]->(recipe:SynthesisRecipe)
                      -[:PRODUCES]->(product:Material)
                RETURN product.formula as product, 
                       recipe.reaction_string as reaction,
                       recipe.type as type
                LIMIT $limit
            """,
            "find_pathways": """
                MATCH path = (start:Material {formula: $start})-[:CAN_SYNTHESIZE*1..{hops}]->(end:Material)
                WHERE end.formula <> start.formula
                RETURN [node in nodes(path) | node.formula] as pathway
                LIMIT $limit
            """,
            "find_by_type": """
                MATCH (recipe:SynthesisRecipe {type: $type})-[:PRODUCES]->(product:Material)
                RETURN product.formula as product,
                       recipe.reaction_string as reaction,
                       recipe.doi as doi
                LIMIT $limit
            """,
            "get_operations": """
                MATCH (recipe:SynthesisRecipe {recipe_id: $recipe_id})-[r:HAS_OPERATION]->(op:Operation)
                RETURN op.type as operation, 
                       op.string as description,
                       op.conditions as conditions
                ORDER BY r.sequence
            """,
        }
    
    def close(self):
        self.driver.close()
    
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict]:
        """Execute a Cypher query and return results"""
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]
    
    def find_synthesis_routes(self, material: str, limit: int = 5) -> List[Dict]:
        """Find synthesis routes to create a specific material"""
        return self.execute_query(
            self.query_templates["find_synthesis"],
            {"formula": material, "limit": limit}
        )
    
    def find_products_from(self, precursor: str, limit: int = 10) -> List[Dict]:
        """Find what can be made from a specific precursor"""
        return self.execute_query(
            self.query_templates["find_products"],
            {"formula": precursor, "limit": limit}
        )
    
    def find_synthesis_pathways(self, start_material: str, hops: int = 2, limit: int = 5) -> List[Dict]:
        """Find multi-step synthesis pathways"""
        query = self.query_templates["find_pathways"].format(hops=hops)
        return self.execute_query(query, {"start": start_material, "limit": limit})
    
    def find_by_synthesis_type(self, synthesis_type: str, limit: int = 10) -> List[Dict]:
        """Find recipes by synthesis type (e.g., 'precipitation', 'hydrothermal')"""
        return self.execute_query(
            self.query_templates["find_by_type"],
            {"type": synthesis_type, "limit": limit}
        )
    
    def get_recipe_operations(self, recipe_id: str) -> List[Dict]:
        """Get detailed operations for a specific recipe"""
        return self.execute_query(
            self.query_templates["get_operations"],
            {"recipe_id": recipe_id}
        )
    
    def search_materials(self, pattern: str, limit: int = 10) -> List[str]:
        """Search for materials by formula pattern"""
        query = """
            MATCH (m:Material)
            WHERE m.formula CONTAINS $pattern
            RETURN DISTINCT m.formula as formula
            LIMIT $limit
        """
        results = self.execute_query(query, {"pattern": pattern, "limit": limit})
        return [r["formula"] for r in results]
    
    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics"""
        stats = {}
        
        with self.driver.session() as session:
            stats["materials"] = session.run("MATCH (m:Material) RETURN count(m)").single()[0]
            stats["recipes"] = session.run("MATCH (r:SynthesisRecipe) RETURN count(r)").single()[0]
            stats["operations"] = session.run("MATCH (o:Operation) RETURN count(o)").single()[0]
            stats["solvents"] = session.run("MATCH (s:Solvent) RETURN count(s)").single()[0]
        
        return stats
    
    def natural_language_query(self, question: str) -> str:
        """
        Convert natural language to Cypher query and execute
        (Requires OpenAI API key)
        """
        if not self.llm_api_key:
            return "Error: OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
        
        try:
            from langchain_openai import ChatOpenAI
            from langchain.chains import GraphCypherQAChain
            from langchain_community.graphs import Neo4jGraph
            
            graph = Neo4jGraph(
                url=NEO4J_URI,
                username=NEO4J_USER,
                password=NEO4J_PASSWORD
            )
            
            llm = ChatOpenAI(temperature=0, model="gpt-4")
            
            chain = GraphCypherQAChain.from_llm(
                llm=llm,
                graph=graph,
                verbose=True,
                return_intermediate_steps=True
            )
            
            result = chain.invoke({"query": question})
            return result
            
        except ImportError:
            return "Error: LangChain not installed. Run: pip install -r requirements_agent.txt"
        except Exception as e:
            return f"Error: {str(e)}"


def main():
    """Demo the agent"""
    print("=" * 70)
    print("Materials Synthesis Knowledge Graph Agent")
    print("=" * 70)
    
    agent = MaterialsSynthesisAgent(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY)
    
    try:
        # Show statistics
        print("\nDatabase Statistics:")
        stats = agent.get_statistics()
        for key, value in stats.items():
            print(f"  {key.capitalize()}: {value:,}")
        
        # Example queries
        print("\n" + "=" * 70)
        print("Example 1: Find synthesis routes for La4Ti9O24")
        print("=" * 70)
        routes = agent.find_synthesis_routes("La4Ti9O24", limit=3)
        for i, route in enumerate(routes, 1):
            print(f"\n{i}. {route['reaction']}")
            print(f"   Type: {route['type']}")
            print(f"   Precursors: {', '.join(route['precursors'])}")
            print(f"   DOI: {route['doi']}")
        
        print("\n" + "=" * 70)
        print("Example 2: What can be made from TiCl4?")
        print("=" * 70)
        products = agent.find_products_from("TiCl4", limit=5)
        for i, prod in enumerate(products, 1):
            print(f"{i}. {prod['product']} via {prod['type']}")
        
        print("\n" + "=" * 70)
        print("Example 3: Search for titanium-containing materials")
        print("=" * 70)
        materials = agent.search_materials("Ti", limit=10)
        print(", ".join(materials))
        
        print("\n" + "=" * 70)
        print("Example 4: Find precipitation reactions")
        print("=" * 70)
        precip = agent.find_by_synthesis_type("precipitation", limit=3)
        for i, recipe in enumerate(precip, 1):
            print(f"{i}. {recipe['reaction']}")
        
    finally:
        agent.close()
    
    print("\n" + "=" * 70)
    print("Agent demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

