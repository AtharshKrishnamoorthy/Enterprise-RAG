# Deep eval metrics 

import os
from deepeval import evaluate 
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.llms.gemini_model import GeminiModel
from ingestion import IngestionPipeline
from retrieval import RetrieverPipeline
from dotenv import load_dotenv , find_dotenv 


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Creating a workflow class for the evaluation 
class EvaluationPipeline:
    def __init__(self, eval_llm: str, query: str, answer: str, retrieval_context: str, expected_output: str):
        self.eval_llm = "gemini-2.0-flash-lite"
        self.query = query
        self.retrieval_context = [retrieval_context]
        self.expected_output = expected_output
        self.answer = answer
        
        
    def get_eval_llm(self):
        eval_llm = GeminiModel(
            model_name = self.eval_llm,
            api_key = GOOGLE_API_KEY,
        )
        return eval_llm
        
    def answer_relevancy(self):
        llm = self.get_eval_llm()
        metric = AnswerRelevancyMetric(
        threshold = 0.8,
        model = llm,
        strict_mode = True,
        include_reason = True
        )
        
        test_case = LLMTestCase(
            input = self.query,
            actual_output = self.answer
        )
        
        metric.measure(test_case)
        return metric.score , metric.reason
    
    def faithfulness(self):
        llm = self.get_eval_llm()
        metric = FaithfulnessMetric(
        threshold = 0.8,
        model = llm,
        strict_mode = True,
        include_reason = True
        )
        
        test_case = LLMTestCase(
            input = self.query,
            actual_output = self.answer,
            retrieval_context = self.retrieval_context
        )
        
        metric.measure(test_case)
        return metric.score , metric.reason
    
    
    def context_precision(self):
        llm = self.get_eval_llm()
        metric = ContextualPrecisionMetric(
        threshold = 0.8,
        model = llm,
        strict_mode = True,
        include_reason = True
        )
        
        test_case = LLMTestCase(
            input = self.query,
            actual_output = self.answer,
            retrieval_context = self.retrieval_context,
            expected_output = self.expected_output
        )
        
        metric.measure(test_case)
        return metric.score , metric.reason
        
    def context_recall(self):
        llm = self.get_eval_llm()
        metric = ContextualRecallMetric(
        threshold = 0.8,
        model = llm,
        strict_mode = True,
        include_reason = True
        )
        
        test_case = LLMTestCase(
            input = self.query,
            actual_output = self.answer,
            retrieval_context = self.retrieval_context,
            expected_output = self.expected_output
        )
        
        metric.measure(test_case)
        return metric.score , metric.reason
        
    def context_relevance(self):
        llm = self.get_eval_llm()
        metric = ContextualRelevancyMetric(
        threshold = 0.8,
        model = llm,
        strict_mode = True,
        include_reason = True
        )
        
        test_case = LLMTestCase(
            input = self.query,
            actual_output = self.answer,
            retrieval_context = self.retrieval_context,
           
        )
        
        metric.measure(test_case)
        return metric.score , metric.reason
        
    
    # Complied version of the evaluation 
    
    def compiled_eval(self):
        
        # Neat compiled version of the eval 
        results = {}
        
        # Run all individual metric functions
        results['answer_relevancy'] = {
            'score': self.answer_relevancy()[0],
            'reason': self.answer_relevancy()[1]
        }
        
        results['faithfulness'] = {
            'score': self.faithfulness()[0],
            'reason': self.faithfulness()[1]
        }
        
        results['contextual_precision'] = {
            'score': self.context_precision()[0],
            'reason': self.context_precision()[1]
        }
        
        results['contextual_recall'] = {
            'score': self.context_recall()[0],
            'reason': self.context_recall()[1]
        }
        
        results['contextual_relevancy'] = {
            'score': self.context_relevance()[0],
            'reason': self.context_relevance()[1]
        }
        
        return results
        
        
if __name__ == "__main__":
    # Main logic code
    # Example usage
    query = "What if these shoes don't fit?"
    answer = "We offer a 30-day full refund at no extra cost."
    retrieval_context = "All customers are eligible for a 30 day full refund at no extra cost."
    expected_output = "We provide a 30-day return policy with full refund."
    
    # Create evaluation pipeline instance
    eval_pipeline = EvaluationPipeline(
        eval_llm="gemini-2.0-flash-exp",
        query=query,
        answer=answer,
        retrieval_context=retrieval_context,
        expected_output=expected_output
    )
    
    # Run compiled evaluation
    results = eval_pipeline.compiled_eval()
    
    # Print results
    print("Evaluation Results:")
    print("=" * 50)
    for metric_name, result in results.items():
        print(f"{metric_name.replace('_', ' ').title()}:")
        print(f"  Score: {result['score']}")
        print(f"  Reason: {result['reason']}")
        print("-" * 30)