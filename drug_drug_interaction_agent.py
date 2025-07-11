#!/usr/bin/env python3
"""
Drug-Drug Interaction Agent - Clean Version with JSON Support
"""

import os
import json
import requests
from typing import Dict, List, TypedDict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from urllib.parse import quote
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime

# Optional judgeval imports
try:
    from judgeval.tracer import Tracer
    from judgeval.scorers import AnswerRelevancyScorer, InstructionAdherenceScorer
    from judgeval.judgment_client import JudgmentClient
    from judgeval.integrations.langgraph import JudgevalCallbackHandler
    JUDGEVAL_AVAILABLE = True
except ImportError:
    print("Note: judgeval package not available. Evaluation features will be disabled.")
    Tracer = None
    AnswerRelevancyScorer = None
    InstructionAdherenceScorer = None
    JudgmentClient = None
    JudgevalCallbackHandler = None
    JUDGEVAL_AVAILABLE = False

# Initialize clients with error handling
try:
    client = JudgmentClient() if JUDGEVAL_AVAILABLE else None
except Exception as e:
    print(f"Warning: JudgmentClient initialization failed: {e}")
    client = None

try:
    chat_model = ChatOpenAI(model="gpt-4", temperature=0)
except Exception as e:
    print(f"Warning: ChatOpenAI initialization failed: {e}")
    chat_model = None

try:
    judgment = Tracer(
        api_key=os.getenv("JUDGMENT_API_KEY"),
        project_name="drug-drug-interaction-bot",
        enable_monitoring=True,
        deep_tracing=False
    ) if JUDGEVAL_AVAILABLE else None
except Exception as e:
    print(f"Warning: Tracer initialization failed: {e}")
    judgment = None

# API Configuration
FDA_API_BASE = "https://api.fda.gov/drug"
RXNORM_API_BASE = "https://rxnav.nlm.nih.gov/REST"

# Create a conditional decorator
def observe_conditional(span_type: str, name: str = None):
    def decorator(func):
        if judgment is not None:
            if name:
                return judgment.observe(span_type=span_type, name=name)(func)
            else:
                return judgment.observe(span_type=span_type)(func)
        else:
            return func
    return decorator

# State definition for the workflow
class DrugInteractionState(TypedDict):
    user_input: str
    extracted_drugs: List[str]
    drug1: str
    drug2: str
    validated_drugs: Dict[str, Any]
    drug1_info: Dict[str, Any]
    drug2_info: Dict[str, Any]
    interaction_data: Dict[str, Any]
    summary: str
    final_result: str
    error_message: str
    evaluation: Dict[str, Any]

class DrugInteractionTools:
    """Drug interaction analysis tools with JSON support"""
    
    def __init__(self):
        self.initialized = False
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DrugInteractionAgent/1.0'
        })
        self.chroma_client = None
        self.collection = None
        
        # Initialize embedding function
        try:
            if os.getenv("OPENAI_API_KEY"):
                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name="text-embedding-ada-002"
                )
            else:
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        except Exception as e:
            print(f"Warning: Could not initialize embedding function: {e}")
            self.embedding_function = None
    
    def initialize(self) -> None:
        """Initialize the tools"""
        if not self.initialized:
            print("Initializing drug interaction tools...")
            self.setup_chromadb()
            self.initialized = True
    
    def setup_chromadb(self) -> None:
        """Setup ChromaDB with data from JSON file or sample data"""
        try:
            if self.embedding_function is None:
                print("Warning: No embedding function available, skipping ChromaDB setup")
                return
                
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            
            try:
                self.collection = self.chroma_client.get_collection(
                    name="drug_interactions",
                    embedding_function=self.embedding_function
                )
                print(f"✓ Using existing collection with {self.collection.count()} documents")
            except Exception:
                print("Collection doesn't exist, creating new one...")
                try:
                    self.collection = self.chroma_client.create_collection(
                        name="drug_interactions",
                        embedding_function=self.embedding_function
                    )
                    print("✓ Created new collection")
                    self.load_drug_interactions_data()
                except Exception as create_error:
                    print(f"✗ Failed to create collection: {create_error}")
                    self.chroma_client = None
                    self.collection = None
                    return
                
        except Exception as e:
            print(f"✗ ChromaDB setup failed: {e}")
            self.chroma_client = None
            self.collection = None
    
    def load_drug_interactions_data(self) -> None:
        """Load drug interactions from JSON file, create if not exists"""
        try:
            if os.path.exists('drug_interactions_data.json'):
                print("Loading data from drug_interactions_data.json...")
                with open('drug_interactions_data.json', 'r') as f:
                    data = json.load(f)
                
                interactions = data.get('drug_interactions', [])
                if interactions:
                    for i, interaction in enumerate(interactions):
                        document = f"Drug Interaction: {interaction['drug1']} and {interaction['drug2']}, Severity: {interaction['severity']}, Description: {interaction['description']}, Mechanism: {interaction['mechanism']}, Recommendation: {interaction['recommendation']}"
                        
                        try:
                            self.collection.add(
                                documents=[document],
                                metadatas=[{
                                    'drug1': interaction['drug1'],
                                    'drug2': interaction['drug2'],
                                    'severity': interaction['severity'],
                                    'source': interaction.get('source', 'json_file')
                                }],
                                ids=[f"interaction_{interaction.get('id', i)}"]
                            )
                        except Exception as add_error:
                            if "already exists" not in str(add_error):
                                print(f"Warning: Failed to add interaction {interaction['drug1']} + {interaction['drug2']}: {add_error}")
                    
                    print(f"✓ Loaded {len(interactions)} interactions from JSON file")
                else:
                    print("No interactions found in JSON file, creating sample data")
                    self.create_sample_data_and_json()
            else:
                print("drug_interactions_data.json not found, creating sample data...")
                self.create_sample_data_and_json()
                
        except Exception as e:
            print(f"Error loading JSON data: {e}")
            print("Falling back to sample data...")
            self.load_sample_data()
    
    def create_sample_data_and_json(self) -> None:
        """Create sample data and save to JSON file"""
        sample_data = {
            "drug_interactions": [
                {
                    "id": 1,
                    "drug1": "warfarin",
                    "drug2": "aspirin",
                    "severity": "MAJOR",
                    "description": "Increased bleeding risk due to additive anticoagulant effects",
                    "mechanism": "Warfarin inhibits vitamin K-dependent clotting factors while aspirin prevents platelet aggregation",
                    "recommendation": "Avoid concurrent use or monitor INR closely",
                    "source": "clinical_database"
                },
                {
                    "id": 2,
                    "drug1": "metformin",
                    "drug2": "alcohol",
                    "severity": "MODERATE",
                    "description": "Increased risk of lactic acidosis",
                    "mechanism": "Both can cause lactic acidosis through different pathways",
                    "recommendation": "Limit alcohol intake, monitor lactate levels",
                    "source": "clinical_database"
                },
                {
                    "id": 3,
                    "drug1": "simvastatin",
                    "drug2": "grapefruit",
                    "severity": "MAJOR",
                    "description": "Increased statin concentration leading to myopathy risk",
                    "mechanism": "Grapefruit inhibits CYP3A4 enzyme",
                    "recommendation": "Avoid grapefruit juice during statin therapy",
                    "source": "clinical_database"
                },
                {
                    "id": 4,
                    "drug1": "digoxin",
                    "drug2": "furosemide",
                    "severity": "MODERATE",
                    "description": "Increased digoxin levels due to potassium depletion",
                    "mechanism": "Furosemide causes potassium loss, increasing digoxin toxicity risk",
                    "recommendation": "Monitor digoxin levels and potassium, consider dose adjustment",
                    "source": "clinical_database"
                },
                {
                    "id": 5,
                    "drug1": "lisinopril",
                    "drug2": "ibuprofen",
                    "severity": "MODERATE",
                    "description": "Reduced antihypertensive effect and potential kidney damage",
                    "mechanism": "NSAIDs can reduce ACE inhibitor effectiveness and cause nephrotoxicity",
                    "recommendation": "Monitor blood pressure and kidney function, consider alternative pain relief",
                    "source": "clinical_database"
                }
            ]
        }
        
        try:
            with open('drug_interactions_data.json', 'w') as f:
                json.dump(sample_data, f, indent=2)
            print("✓ Created drug_interactions_data.json with sample data")
            
            # Now load the data into ChromaDB
            self.load_drug_interactions_data()
            
        except Exception as e:
            print(f"Error creating JSON file: {e}")
            print("Using in-memory sample data only...")
            self.load_sample_data()
    
    def load_sample_data(self) -> None:
        """Load sample drug interaction data directly (fallback)"""
        if self.collection is None:
            print("No ChromaDB collection available, skipping data loading")
            return
            
        sample_interactions = [
            {
                "drug1": "warfarin", "drug2": "aspirin",
                "severity": "MAJOR",
                "description": "Increased bleeding risk due to additive anticoagulant effects",
                "mechanism": "Warfarin inhibits vitamin K-dependent clotting factors while aspirin prevents platelet aggregation",
                "recommendation": "Avoid concurrent use or monitor INR closely"
            },
            {
                "drug1": "metformin", "drug2": "alcohol",
                "severity": "MODERATE",
                "description": "Increased risk of lactic acidosis",
                "mechanism": "Both can cause lactic acidosis through different pathways",
                "recommendation": "Limit alcohol intake, monitor lactate levels"
            },
            {
                "drug1": "simvastatin", "drug2": "grapefruit",
                "severity": "MAJOR",
                "description": "Increased statin concentration leading to myopathy risk",
                "mechanism": "Grapefruit inhibits CYP3A4 enzyme",
                "recommendation": "Avoid grapefruit juice during statin therapy"
            }
        ]
        
        try:
            for interaction in sample_interactions:
                document = f"Drug1: {interaction['drug1']}, Drug2: {interaction['drug2']}, Severity: {interaction['severity']}, Description: {interaction['description']}, Mechanism: {interaction['mechanism']}, Recommendation: {interaction['recommendation']}"
                
                self.collection.add(
                    documents=[document],
                    metadatas=[interaction],
                    ids=[f"{interaction['drug1']}_{interaction['drug2']}"]
                )
            
            print(f"✓ Loaded {len(sample_interactions)} sample interactions")
        except Exception as e:
            print(f"Error loading sample data: {e}")
            self.collection = None
    
    @observe_conditional("tool", name="RAG vector search")
    def search_drug_interactions_rag(self, drug1: str, drug2: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Search for drug interactions using RAG"""
        if self.collection is None:
            return []
        
        try:
            query = f"interaction between {drug1} and {drug2}"
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            formatted_results = []
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'drug1': results['metadatas'][0][i]['drug1'],
                    'drug2': results['metadatas'][0][i]['drug2'],
                    'severity': results['metadatas'][0][i]['severity'],
                    'document': doc,
                    'similarity_score': 1 - results['distances'][0][i]
                })
            
            return formatted_results
        except Exception as e:
            print(f"RAG search failed: {e}")
            return []
    
    @observe_conditional("tool", name="Extract drug names")
    def extract_drug_names(self, user_input: str) -> List[str]:
        """Extract drug names from user input"""
        print(f"Extracting drugs from: {user_input}")
        
        try:
            if chat_model is None:
                words = user_input.lower().split()
                drugs = [word for word in words if len(word) > 3][:2]
                return drugs
            
            response = chat_model.invoke([
                SystemMessage(content="Extract drug names from the input. Return only the drug names separated by commas, or 'NO_DRUGS_FOUND' if no drugs found."),
                HumanMessage(content=f"Extract drug names from: {user_input}")
            ])
            
            extracted_text = response.content.strip()
            if extracted_text == 'NO_DRUGS_FOUND':
                return []
            
            drugs = [drug.strip() for drug in extracted_text.split(',')]
            print(f"Extracted: {drugs}")
            return drugs
        except Exception as e:
            print(f"Drug extraction failed: {e}")
            return []
    
    @observe_conditional("tool", name="RxNorm drug validation")
    def validate_drug_with_rxnorm(self, drug_name: str) -> Dict[str, Any]:
        """Validate drug name using RxNorm API"""
        print(f"Validating drug: {drug_name}")
        
        try:
            search_url = f"{RXNORM_API_BASE}/drugs.json?name={quote(drug_name)}"
            response = self.session.get(search_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'drugGroup' in data and 'conceptGroup' in data['drugGroup']:
                    concepts = data['drugGroup']['conceptGroup']
                    if concepts:
                        for concept_group in concepts:
                            if 'conceptProperties' in concept_group:
                                for concept in concept_group['conceptProperties']:
                                    print(f"✓ Validated: {concept.get('name', drug_name)}")
                                    return {
                                        'rxcui': concept.get('rxcui', ''),
                                        'name': concept.get('name', drug_name),
                                        'synonym': concept.get('synonym', ''),
                                        'tty': concept.get('tty', ''),
                                        'valid': True,
                                        'source': 'rxnorm'
                                    }
            
            print(f"✗ Not found in RxNorm: {drug_name}")
            return {
                'name': drug_name,
                'valid': False,
                'error': 'Drug not found in RxNorm',
                'source': 'rxnorm'
            }
        except Exception as e:
            print(f"✗ Validation error for {drug_name}: {e}")
            return {
                'name': drug_name,
                'valid': False,
                'error': str(e),
                'source': 'rxnorm'
            }
    
    @observe_conditional("tool", name="FDA drug information")
    def get_drug_info_from_fda(self, drug_name: str) -> Dict[str, Any]:
        """Get drug information from FDA API"""
        print(f"Getting FDA info for: {drug_name}")
        
        try:
            url = f"{FDA_API_BASE}/label.json"
            params = {
                "search": f"openfda.brand_name:{drug_name} OR openfda.generic_name:{drug_name}",
                "limit": 1
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    result = data['results'][0]
                    openfda = result.get('openfda', {})
                    
                    brand_name = openfda.get('brand_name', [drug_name])[0] if openfda.get('brand_name') else drug_name
                    generic_name = openfda.get('generic_name', [drug_name])[0] if openfda.get('generic_name') else drug_name
                    manufacturer = openfda.get('manufacturer_name', ['Unknown'])[0] if openfda.get('manufacturer_name') else 'Unknown'
                    
                    print(f"✓ Found FDA info: {brand_name} ({generic_name})")
                    return {
                        'brand_name': brand_name,
                        'generic_name': generic_name,
                        'manufacturer': manufacturer,
                        'route': openfda.get('route', ['Unknown'])[0] if openfda.get('route') else 'Unknown',
                        'substance_name': openfda.get('substance_name', []),
                        'pharmacologic_class': openfda.get('pharm_class_epc', []),
                        'valid': True,
                        'source': 'fda'
                    }
            
            print(f"✗ Not found in FDA: {drug_name}")
            return {
                'name': drug_name,
                'valid': False,
                'error': 'Drug not found in FDA database',
                'source': 'fda'
            }
        except Exception as e:
            print(f"✗ FDA info error for {drug_name}: {e}")
            return {
                'name': drug_name,
                'valid': False,
                'error': str(e),
                'source': 'fda'
            }
    
    @observe_conditional("tool", name="Analyze drug interaction")
    def analyze_drug_interaction(self, drug1: str, drug2: str) -> Dict[str, Any]:
        """Analyze drug interaction using RAG and LLM"""
        print(f"Analyzing interaction: {drug1} + {drug2}")
        
        rag_results = self.search_drug_interactions_rag(drug1, drug2)
        
        if rag_results and rag_results[0]['similarity_score'] > 0.7:
            best_match = rag_results[0]
            print(f"Found RAG match: {best_match['severity']} severity")
            return {
                'drug1': drug1,
                'drug2': drug2,
                'severity': best_match['severity'],
                'description': best_match['document'].split('Description:')[1].split('Mechanism:')[0].strip() if 'Description:' in best_match['document'] else 'See analysis',
                'mechanism': best_match['document'].split('Mechanism:')[1].split('Recommendation:')[0].strip() if 'Mechanism:' in best_match['document'] else 'See analysis',
                'recommendation': best_match['document'].split('Recommendation:')[1].strip() if 'Recommendation:' in best_match['document'] else 'Consult healthcare provider',
                'source': 'database'
            }
        
        try:
            if chat_model is None:
                return {
                    'drug1': drug1,
                    'drug2': drug2,
                    'severity': 'UNKNOWN',
                    'description': f'Cannot analyze {drug1} and {drug2} interaction - AI model not available',
                    'mechanism': 'Analysis not available',
                    'recommendation': 'Consult healthcare provider',
                    'source': 'fallback'
                }
            
            prompt = f"""
            Analyze the potential drug-drug interaction between {drug1} and {drug2}.
            
            Please provide:
            1. Severity level (MAJOR, MODERATE, MINOR, or NONE)
            2. Description of the interaction
            3. Mechanism of interaction
            4. Clinical recommendations
            
            Format your response as JSON with these exact keys:
            - severity
            - description
            - mechanism
            - recommendation
            """
            
            response = chat_model.invoke([
                SystemMessage(content="You are a clinical pharmacist expert. Analyze drug interactions and provide detailed information."),
                HumanMessage(content=prompt)
            ])
            
            try:
                interaction_data = json.loads(response.content)
                interaction_data['drug1'] = drug1
                interaction_data['drug2'] = drug2
                interaction_data['source'] = 'llm_analysis'
                return interaction_data
            except json.JSONDecodeError:
                return {
                    'drug1': drug1,
                    'drug2': drug2,
                    'severity': 'UNKNOWN',
                    'description': response.content[:200] + '...',
                    'mechanism': 'Analysis provided by LLM',
                    'recommendation': 'Consult healthcare provider',
                    'source': 'llm_analysis'
                }
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return {
                'drug1': drug1,
                'drug2': drug2,
                'severity': 'UNKNOWN',
                'description': f'Error analyzing interaction: {str(e)}',
                'mechanism': 'Error occurred',
                'recommendation': 'Consult healthcare provider',
                'source': 'error'
            }
    
    @observe_conditional("tool", name="Generate summary")
    def generate_summary(self, state_data: Dict[str, Any]) -> str:
        """Generate concise summary of drug interaction"""
        print("Generating summary...")
        
        try:
            drug1 = state_data.get('drug1', 'Unknown')
            drug2 = state_data.get('drug2', 'Unknown')
            interaction_data = state_data.get('interaction_data', {})
            
            severity = interaction_data.get('severity', 'UNKNOWN')
            description = interaction_data.get('description', 'Unable to determine interaction details')
            recommendation = interaction_data.get('recommendation', 'Consult healthcare provider')
            
            if chat_model is None:
                return f"The interaction between {drug1} and {drug2} shows {severity} severity. {description} {recommendation}"
            
            prompt = f"""
            Create a concise paragraph summarizing the drug interaction between {drug1} and {drug2}.
            
            Interaction Data: {json.dumps(interaction_data, indent=2)}
            
            Provide a single focused paragraph that includes:
            - Severity level
            - Key mechanism
            - Main clinical recommendation
            - Brief monitoring notes if needed
            
            Keep it professional and concise.
            """
            
            response = chat_model.invoke([
                SystemMessage(content="You are a clinical pharmacist providing concise drug interaction summaries."),
                HumanMessage(content=prompt)
            ])
            
            return response.content
            
        except Exception as e:
            print(f"Summary generation failed: {e}")
            return f"Error generating summary: {str(e)}"
    
    def evaluate_interaction_analysis(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate using Answer Relevancy and Instruction Adherence scorers"""
        print("Evaluating with relevancy and adherence scorers...")
        
        try:
            user_input = state_data.get('user_input', '')
            summary = state_data.get('summary', '')
            
            if judgment is not None and summary and JUDGEVAL_AVAILABLE:
                try:
                    judgment.async_evaluate(
                        input=user_input,
                        actual_output=summary,
                        scorers=[
                            AnswerRelevancyScorer(threshold=0.7),
                            InstructionAdherenceScorer(threshold=0.7)
                        ],
                        model="gpt-4o"
                    )
                    print("✓ Evaluation submitted successfully")
                    return {
                        'evaluation_submitted': True,
                        'scorers_used': ['AnswerRelevancyScorer', 'InstructionAdherenceScorer'],
                        'timestamp': datetime.now().isoformat()
                    }
                except Exception as e:
                    print(f"✗ Evaluation failed: {e}")
                    return {
                        'evaluation_submitted': False,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
            else:
                print("✗ Cannot evaluate - judgment platform or summary not available")
                return {
                    'evaluation_submitted': False,
                    'error': 'Judgment platform or summary not available',
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            print(f"Evaluation error: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Initialize tools
tools = DrugInteractionTools()

# Workflow nodes
@observe_conditional("workflow_orchestration", name="Input processing")
def input_processing_node(state: DrugInteractionState) -> DrugInteractionState:
    """Process user input and extract drug names"""
    print("\n=== Processing Input ===")
    
    if not tools.initialized:
        tools.initialize()
    
    user_input = state.get('user_input', '')
    extracted_drugs = tools.extract_drug_names(user_input)
    
    if len(extracted_drugs) < 2:
        return {
            **state,
            'error_message': 'Please provide two drug names for interaction analysis'
        }
    
    return {
        **state,
        'extracted_drugs': extracted_drugs,
        'drug1': extracted_drugs[0],
        'drug2': extracted_drugs[1]
    }

@observe_conditional("workflow_orchestration", name="Drug validation")
def drug_validation_node(state: DrugInteractionState) -> DrugInteractionState:
    """Validate drug names using RxNorm"""
    print("\n=== Validating Drugs ===")
    
    drug1 = state.get('drug1', '')
    drug2 = state.get('drug2', '')
    
    drug1_validation = tools.validate_drug_with_rxnorm(drug1)
    drug2_validation = tools.validate_drug_with_rxnorm(drug2)
    
    validated_drugs = {
        'drug1_validation': drug1_validation,
        'drug2_validation': drug2_validation,
        'both_valid': drug1_validation.get('valid', False) and drug2_validation.get('valid', False)
    }
    
    return {
        **state,
        'validated_drugs': validated_drugs
    }

@observe_conditional("workflow_orchestration", name="Drug info retrieval")
def drug_info_retrieval_node(state: DrugInteractionState) -> DrugInteractionState:
    """Retrieve drug information from FDA"""
    print("\n=== Getting Drug Info ===")
    
    drug1 = state.get('drug1', '')
    drug2 = state.get('drug2', '')
    
    drug1_info = tools.get_drug_info_from_fda(drug1)
    drug2_info = tools.get_drug_info_from_fda(drug2)
    
    return {
        **state,
        'drug1_info': drug1_info,
        'drug2_info': drug2_info
    }

@observe_conditional("workflow_orchestration", name="Interaction analysis")
def interaction_analysis_node(state: DrugInteractionState) -> DrugInteractionState:
    """Analyze drug interactions"""
    print("\n=== Analyzing Interaction ===")
    
    drug1 = state.get('drug1', '')
    drug2 = state.get('drug2', '')
    
    interaction_data = tools.analyze_drug_interaction(drug1, drug2)
    
    return {
        **state,
        'interaction_data': interaction_data
    }

@observe_conditional("workflow_orchestration", name="Summary generation")
def summary_generation_node(state: DrugInteractionState) -> DrugInteractionState:
    """Generate summary"""
    print("\n=== Generating Summary ===")
    
    summary = tools.generate_summary(state)
    
    return {
        **state,
        'summary': summary,
        'final_result': summary
    }

@observe_conditional("workflow_orchestration", name="Evaluation")
def evaluation_node(state: DrugInteractionState) -> DrugInteractionState:
    """Evaluate the analysis"""
    print("\n=== Evaluating Analysis ===")
    
    evaluation_result = tools.evaluate_interaction_analysis(state)
    
    return {
        **state,
        'evaluation': evaluation_result
    }

def error_node(state: DrugInteractionState) -> DrugInteractionState:
    """Handle errors"""
    error_message = state.get('error_message', 'Unknown error occurred')
    return {
        **state,
        'final_result': f"Error: {error_message}"
    }

def should_continue_after_input(state: DrugInteractionState) -> str:
    """Determine next step after input processing"""
    if state.get('error_message'):
        return 'error'
    return 'drug_validation'

def create_workflow() -> StateGraph:
    """Create the drug interaction workflow"""
    
    workflow = StateGraph(DrugInteractionState)
    
    workflow.add_node("input_processing", input_processing_node)
    workflow.add_node("drug_validation", drug_validation_node)
    workflow.add_node("drug_info_retrieval", drug_info_retrieval_node)
    workflow.add_node("interaction_analysis", interaction_analysis_node)
    workflow.add_node("summary_generation", summary_generation_node)
    workflow.add_node("evaluation", evaluation_node)
    workflow.add_node("error", error_node)
    
    workflow.set_entry_point("input_processing")
    workflow.add_conditional_edges(
        "input_processing",
        should_continue_after_input,
        {"drug_validation": "drug_validation", "error": "error"}
    )
    workflow.add_edge("drug_validation", "drug_info_retrieval")
    workflow.add_edge("drug_info_retrieval", "interaction_analysis")
    workflow.add_edge("interaction_analysis", "summary_generation")
    workflow.add_edge("summary_generation", "evaluation")
    workflow.add_edge("evaluation", END)
    workflow.add_edge("error", END)
    
    return workflow.compile()

@observe_conditional("agent_execution", name="drug interaction analysis")
def drug_interaction_analysis(user_input: str) -> str:
    """Main function to analyze drug interactions"""
    print(f"\n=== Drug Interaction Analysis ===")
    print(f"Query: {user_input}")
    
    workflow = create_workflow()
    
    initial_state = {
        'user_input': user_input,
        'extracted_drugs': [],
        'drug1': '',
        'drug2': '',
        'validated_drugs': {},
        'drug1_info': {},
        'drug2_info': {},
        'interaction_data': {},
        'summary': '',
        'final_result': '',
        'error_message': '',
        'evaluation': {}
    }
    
    try:
        final_state = workflow.invoke(initial_state)
        
        if final_state.get('interaction_data'):
            interaction = final_state['interaction_data']
            print(f"\n=== Results ===")
            print(f"Drugs: {interaction.get('drug1', 'N/A')} + {interaction.get('drug2', 'N/A')}")
            print(f"Severity: {interaction.get('severity', 'N/A')}")
            print(f"Source: {interaction.get('source', 'N/A')}")
            
            validated = final_state.get('validated_drugs', {})
            if validated:
                drug1_valid = validated.get('drug1_validation', {}).get('valid', False)
                drug2_valid = validated.get('drug2_validation', {}).get('valid', False)
                print(f"Validation: Drug1 {'✓' if drug1_valid else '✗'} | Drug2 {'✓' if drug2_valid else '✗'}")
            
            evaluation = final_state.get('evaluation', {})
            if evaluation.get('evaluation_submitted'):
                print(f"Evaluation: ✓ Submitted")
            elif evaluation.get('error'):
                print(f"Evaluation: ✗ {evaluation.get('error')}")
        
        return final_state.get('final_result', 'No result generated')
    except Exception as e:
        print(f"Workflow error: {e}")
        return f"Error: {str(e)}"

def drug_interaction_analysis_with_handler(handler, user_input: str) -> str:
    """Run analysis with judgeval handler"""
    print(f"\n=== Drug Interaction Analysis ===")
    print(f"Query: {user_input}")
    
    workflow = create_workflow()
    
    initial_state = {
        'user_input': user_input,
        'extracted_drugs': [],
        'drug1': '',
        'drug2': '',
        'validated_drugs': {},
        'drug1_info': {},
        'drug2_info': {},
        'interaction_data': {},
        'summary': '',
        'final_result': '',
        'error_message': '',
        'evaluation': {}
    }
    
    try:
        config_with_callbacks = {"callbacks": [handler]}
        final_state = workflow.invoke(initial_state, config=config_with_callbacks)
        
        if final_state.get('interaction_data'):
            interaction = final_state['interaction_data']
            print(f"\n=== Results ===")
            print(f"Drugs: {interaction.get('drug1', 'N/A')} + {interaction.get('drug2', 'N/A')}")
            print(f"Severity: {interaction.get('severity', 'N/A')}")
            print(f"Source: {interaction.get('source', 'N/A')}")
            
            validated = final_state.get('validated_drugs', {})
            if validated:
                drug1_valid = validated.get('drug1_validation', {}).get('valid', False)
                drug2_valid = validated.get('drug2_validation', {}).get('valid', False)
                print(f"Validation: Drug1 {'✓' if drug1_valid else '✗'} | Drug2 {'✓' if drug2_valid else '✗'}")
            
            evaluation = final_state.get('evaluation', {})
            if evaluation.get('evaluation_submitted'):
                print(f"Evaluation: ✓ Submitted")
            elif evaluation.get('error'):
                print(f"Evaluation: ✗ {evaluation.get('error')}")
        
        return final_state.get('final_result', 'No result generated')
    except Exception as e:
        print(f"Workflow error: {e}")
        return f"Error: {str(e)}"

def interactive_cli():
    """Interactive command line interface"""
    print("=== Drug-Drug Interaction Analysis ===")
    print("Enter drug names or questions about interactions")
    
    try:
        user_input = input("\n> ")
        
        if not user_input.strip():
            print("Please enter drug names to analyze.")
            return
        
        if judgment is not None and JUDGEVAL_AVAILABLE:
            handler = JudgevalCallbackHandler(judgment)
            result = drug_interaction_analysis_with_handler(handler, user_input)
        else:
            result = drug_interaction_analysis(user_input)
        
        print(f"\n{result}")
        print("\nAnalysis complete!")
        
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    interactive_cli() 