# Drug-Drug Interaction Analysis Agent

A sophisticated AI agent that analyzes potential drug-drug interactions using multiple data sources and provides clinical recommendations. This agent integrates with **judgeval** for comprehensive tracing and evaluation.

## 🎯 **Overview**

This agent combines:
- **RxNorm API** for drug validation
- **FDA API** for drug information retrieval
- **RAG (Retrieval-Augmented Generation)** with ChromaDB for interaction database search
- **LLM analysis** for complex interaction assessment
- **judgeval integration** for tracing and evaluation

## 🏗️ **Architecture**

Built using **LangGraph** workflow orchestration with the following pipeline:

```
Input Processing → Drug Validation → FDA Info Retrieval → Interaction Analysis → Summary Generation → Evaluation
```

## 📊 **judgeval Integration**

### **Tracing Coverage**
The agent uses multiple trace decorators to provide comprehensive monitoring:

- `@observe_conditional("vector_search")` - RAG database searches
- `@observe_conditional("drug_extraction")` - LLM drug name extraction  
- `@observe_conditional("drug_validation")` - RxNorm API validation
- `@observe_conditional("drug_info_retrieval")` - FDA API calls
- `@observe_conditional("interaction_analysis")` - RAG + LLM analysis
- `@observe_conditional("summary_generation")` - LLM summary creation
- `@observe_conditional("workflow_orchestration")` - Workflow node execution
- `@observe_conditional("agent_execution")` - Main agent execution

### **Evaluation Scorers**
Integrated scorers for quality assessment:
- **Answer Relevancy Scorer** (threshold: 0.7)
- **Instruction Adherence Scorer** (threshold: 0.7)

## 🚀 **Installation**

### **Prerequisites**
- Python 3.8+
- OpenAI API key
- judgeval API key (optional, for evaluation features)

### **Setup**
1. **Clone or download the agent**:
   ```bash
   git clone <repository-url>
   cd drug-interaction-agent
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export JUDGMENT_API_KEY="your-judgeval-api-key"  # Optional
   ```

### **Dependencies**
```txt
langchain-openai>=0.1.0
langgraph>=0.1.0
chromadb>=0.4.0
requests>=2.28.0
judgeval>=0.1.0
```

## 💊 **Usage**

### **Interactive CLI**
```bash
python drug_drug_interaction_agent.py
```

**Example interaction**:
```
=== Drug-Drug Interaction Analysis ===
Enter drug names or questions about interactions

> warfarin and aspirin

=== Processing Input ===
Extracting drugs from: warfarin and aspirin
Extracted: ['warfarin', 'aspirin']

=== Validating Drugs ===
Validating drug: warfarin
✓ Validated: warfarin
Validating drug: aspirin
✓ Validated: aspirin

=== Getting Drug Info ===
Getting FDA info for: warfarin
✓ Found FDA info: Coumadin (warfarin)
Getting FDA info for: aspirin
✓ Found FDA info: Aspirin (aspirin)

=== Analyzing Interaction ===
Analyzing interaction: warfarin + aspirin
Found RAG match: MAJOR severity

=== Generating Summary ===
Generating summary...

=== Evaluating Analysis ===
Evaluating with relevancy and adherence scorers...
✓ Evaluation submitted successfully

=== Results ===
Drugs: warfarin + aspirin
Severity: MAJOR
Source: database
Validation: Drug1 ✓ | Drug2 ✓
Evaluation: ✓ Submitted

The concurrent use of warfarin and aspirin presents a MAJOR drug interaction...

Analysis complete!
```

### **Programmatic Usage**
```python
from drug_drug_interaction_agent import drug_interaction_analysis

# Analyze drug interaction
result = drug_interaction_analysis("What happens when I take warfarin with aspirin?")
print(result)
```

## 🔧 **Features**

### **Core Capabilities**
- ✅ **Multi-source validation** (RxNorm + FDA APIs)
- ✅ **RAG-powered search** with ChromaDB vector database
- ✅ **LLM fallback analysis** for unknown interactions
- ✅ **JSON data support** with automatic sample data creation
- ✅ **Comprehensive error handling** with graceful fallbacks
- ✅ **Clean CLI interface** with status indicators

### **Data Sources**
- **RxNorm API**: Drug validation and standardization
- **FDA API**: Official drug information and labeling
- **Local JSON database**: Curated drug interaction data
- **ChromaDB**: Vector similarity search for interactions
- **GPT-4**: LLM analysis for complex cases

### **judgeval Integration**
- **Multi-level tracing**: Individual operation traces
- **Automatic evaluation**: Answer relevancy and instruction adherence
- **Error monitoring**: Comprehensive error tracking
- **Performance metrics**: Response time and success rates

## 📁 **Project Structure**

```
drug-interaction-agent/
├── drug_drug_interaction_agent.py    # Main agent code
├── drug_interactions_data.json       # Interaction database (auto-created)
├── chroma_db/                        # ChromaDB vector store
├── requirements.txt                  # Python dependencies
└── README.md                        # This file
```

## 📈 **judgeval Dashboard**

After running the agent, visit your judgeval dashboard to view:

1. **Trace Timeline**: Complete workflow execution steps
2. **LLM Calls**: Token usage and response quality
3. **API Calls**: External service performance
4. **Evaluation Results**: Scoring metrics and insights
5. **Error Analysis**: Failed operations and recovery

## 🔍 **Example Traces**

The agent creates detailed traces for each operation:

```
Drug Interaction Analysis
├── Input Processing (drug_extraction)
│   └── LLM Call: Extract drug names
├── Drug Validation (drug_validation)  
│   ├── RxNorm API: Validate warfarin
│   └── RxNorm API: Validate aspirin
├── FDA Info Retrieval (drug_info_retrieval)
│   ├── FDA API: Get warfarin info
│   └── FDA API: Get aspirin info
├── Interaction Analysis (interaction_analysis)
│   ├── Vector Search: ChromaDB query
│   └── LLM Call: Analyze interaction
├── Summary Generation (summary_generation)
│   └── LLM Call: Generate summary
└── Evaluation
    ├── Answer Relevancy Score: 0.85
    └── Instruction Adherence Score: 0.92
```

## 🚨 **Safety Notice**

⚠️ **This agent is for educational and research purposes only. Always consult healthcare professionals for medical decisions.**

## 🛠️ **Development Notes**

### **Extending the Agent**
- Add new data sources by implementing additional API clients
- Enhance interaction analysis with more sophisticated algorithms
- Integrate additional evaluation metrics
- Add support for more drug databases

### **Configuration**
The agent supports various configuration options:
- Custom API endpoints
- Adjustable similarity thresholds
- Configurable evaluation criteria
- Custom data sources

## 🐛 **judgeval SDK Feedback**

During development, I identified several areas for improvement in the judgeval SDK:

### **Issues Identified**
1. **Documentation Gaps**:
   - Missing examples for LangGraph integration patterns
   - Unclear conditional decorator usage documentation
   - Limited guidance on multi-level tracing strategies

2. **Feature Requests**:
   - Better support for optional tracing (graceful degradation)
   - Batch evaluation APIs for multiple test cases
   - Custom scorer creation documentation

3. **SDK Improvements**:
   - More granular control over trace collection
   - Better error handling for SDK initialization failures
   - Async tracing support for better performance

### **Suggested Improvements**
- [ ] Add cookbook example for conditional tracing patterns
- [ ] Include LangGraph-specific integration guide
- [ ] Provide templates for common agent architectures
- [ ] Add performance optimization guidelines

## 🤝 **Contributing**

Feel free to contribute by:
1. Adding new drug interaction data sources
2. Improving the interaction analysis algorithms
3. Enhancing the evaluation metrics
4. Reporting bugs or suggesting features

## 📄 **License**

This project is licensed under the MIT License.

## 🔗 **Links**

- [judgeval Documentation](https://docs.judgeval.com)
- [judgeval GitHub](https://github.com/judgeval/judgeval)
- [RxNorm API Documentation](https://lhncbc.nlm.nih.gov/RxNav/APIs/)
- [FDA API Documentation](https://open.fda.gov/apis/)
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)

---

**Built with ❤️ using judgeval for comprehensive AI agent monitoring and evaluation** 