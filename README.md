# Drug-Drug Interaction Agent

A simple AI agent that analyzes potential drug interactions using LangGraph workflow, RAG with ChromaDB, and external APIs (RxNorm, FDA). Includes judgeval integration for monitoring and evaluation.

## Quick Setup

1. **Install dependencies:**
```bash
#create virtual environment or simply run
pip install -r requirements.txt
```

2. **Set up environment variables:**
```bash
#  Export environment variables
export OPENAI_API_KEY=your_openai_api_key
export JUDGMENT_API_KEY=your_judgeval_api_key  
export JUDGMENT_ORG_ID=your_judgeval_org_key

```

3. **Run the agent:**
```bash
python drug_drug_interaction_agent.py
```

## Testing the Agent

### Basic Usage
```bash
python drug_drug_interaction_agent.py
```

When prompted, enter drug names like:
- `"aspirin and warfarin"`
- `"ibuprofen and metformin"`
- `"analyze interaction between lisinopril and hydrochlorothiazide"`

### Expected Output
The agent will:
1. Extract drug names from your input
2. Validate drugs using RxNorm API
3. Get drug information from FDA API
4. Search local database for interactions
5. Generate AI-powered analysis
6. Provide summary with severity and recommendations

## Judgeval Dashboard Monitoring

When judgeval is configured, you'll see these traces in your dashboard:

### Main Trace
- **"drug interaction analysis"** - Complete workflow execution

### Tool Operations
- **"Extract drug names"** - LLM-powered drug name extraction
- **"RxNorm drug validation"** - Drug validation via RxNorm API
- **"FDA drug information"** - Drug info retrieval from FDA API
- **"RAG vector search"** - ChromaDB similarity search
- **"Analyze drug interaction"** - AI-powered interaction analysis
- **"Generate summary"** - Final summary generation

### Workflow Steps
- **"Input processing"** - Initial input handling
- **"Drug validation"** - Validation orchestration
- **"Drug info retrieval"** - Information gathering
- **"Interaction analysis"** - Core analysis step
- **"Summary generation"** - Final output creation
- **"Evaluation"** - Result evaluation

## Sample Interaction

```
> aspirin and warfarin

=== Drug Interaction Analysis ===
Query: aspirin and warfarin
Extracting drugs from: aspirin and warfarin
Extracted: ['aspirin', 'warfarin']
Validating drug: aspirin
✓ Validated: aspirin
Validating drug: warfarin
✓ Validated: warfarin
Getting FDA info for: aspirin
✓ Found FDA info: Aspirin (aspirin)
Getting FDA info for: warfarin
✓ Found FDA info: Coumadin (warfarin)
Analyzing interaction: aspirin + warfarin
Found RAG match: MAJOR severity

=== Results ===
Drugs: aspirin + warfarin
Severity: MAJOR
Source: database
Validation: Drug1 ✓ | Drug2 ✓
Evaluation: ✓ Submitted

**MAJOR INTERACTION DETECTED**
Drugs: aspirin + warfarin
Severity: MAJOR
Risk: Increased bleeding risk
Recommendation: Monitor closely, consider alternatives
```

## What Gets Traced

- **API Calls**: RxNorm and FDA API requests
- **LLM Usage**: Drug extraction and interaction analysis
- **Vector Search**: ChromaDB similarity searches
- **Workflow Execution**: Complete LangGraph workflow
- **Tool Performance**: Individual tool execution times
- **Error Handling**: Any failures or exceptions

## Files Structure

```
drug_drug_interaction_agent.py  # Main agent
requirements.txt                # Dependencies
README.md                      # This file
.env                           # Environment variables
chroma_db/                     # ChromaDB storage
├── chroma.sqlite3       # Vector database
drug_interactions.json     # Sample data
example.py               # to understand about judgeval
test_rag.py              # test chroma_db and sample data accessibility
```

## Troubleshooting

**No judgeval monitoring?**
- Check your environment variables
- Verify judgeval package is installed

**API errors?**
- Ensure `OPENAI_API_KEY` is set
- Check internet connection for RxNorm/FDA APIs

**No drug database?**
- Agent will create sample data automatically
- Check `chroma_db/` directory is created

## Dependencies

- **langchain-openai**: LLM integration
- **langgraph**: Workflow orchestration  
- **chromadb**: Vector database
- **requests**: API calls
- **judgeval**: Monitoring and evaluation

Run `pip install -r requirements.txt` to install all dependencies. 
Built with curiosity using judgeval for comprehensive AI agent monitoring and evaluation!