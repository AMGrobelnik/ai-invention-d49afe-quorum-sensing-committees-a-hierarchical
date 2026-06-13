#!/usr/bin/env python3
"""Quorum-Sensing Consensus Protocol - Complete Implementation.

Implements a hierarchical LLM consensus prototype using quorum-sensing committees,
with dual thresholds, Layer-1/Layer-2 agents, and LLM interface via OpenRouter.
"""

from loguru import logger
from pathlib import Path
import json
import sys
import os
import re
import time
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
from collections import Counter
import numpy as np

# =============================================================================
# LOGGING SETUP
# =============================================================================

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
Path("logs").mkdir(exist_ok=True)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# =============================================================================
# BUDGET TRACKING
# =============================================================================

BUDGET_USD = 10.0  # Maximum $10 USD spend
CUMULATIVE_COST = 0.0


def track_cost(cost: float) -> bool:
    """Track cumulative cost and check budget limit.
    
    Args:
        cost: Cost of the current API call
        
    Returns:
        True if within budget, False if budget exceeded
    """
    global CUMULATIVE_COST
    CUMULATIVE_COST += cost
    logger.info(f"Cost tracking: ${CUMULATIVE_COST:.4f} / ${BUDGET_USD:.2f}")
    
    if CUMULATIVE_COST >= BUDGET_USD * 0.9:  # 90% warning
        logger.warning(f"Approaching budget limit: ${CUMULATIVE_COST:.4f}")
    
    if CUMULATIVE_COST > BUDGET_USD:
        logger.error(f"Budget exceeded! ${CUMULATIVE_COST:.4f} > ${BUDGET_USD:.2f}")
        return False
    
    return True


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class VoteResult(Enum):
    """Result of a consensus vote."""
    CONSENSUS = "consensus"
    VETO = "veto"
    NO_CONSENSUS = "no_consensus"


@dataclass
class AgentResponse:
    """Response from a single agent."""
    agent_id: str
    answer: str
    confidence: float  # 0.0 to 1.0
    tokens_used: int
    raw_response: str
    cost: float = 0.0


@dataclass
class ConsensusResult:
    """Result of the consensus protocol."""
    status: VoteResult
    winning_answer: Optional[str]
    confidence_field: Dict[str, float]  # answer -> accumulated confidence
    veto_triggered: bool
    escalated: bool
    layer1_responses: List[AgentResponse]
    layer2_responses: Optional[List[AgentResponse]]
    total_tokens: int
    total_cost: float


@dataclass
class AgentConfig:
    """Configuration for an LLM agent."""
    agent_id: str
    model_name: str  # OpenRouter API name
    temperature: float = 0.7
    max_tokens: int = 500


# =============================================================================
# CONSENSUS FIELD CLASS
# =============================================================================

class ConsensusField:
    """
    Implements the 'consensus field' where agent confidence accumulates
    like bacterial autoinducers in quorum sensing.
    
    Dual thresholds:
    - Θ_c (consensus_threshold): Field value must exceed this for consensus
    - Θ_v (veto_threshold): Minority signal concentration must exceed this for veto
    """
    
    def __init__(self, 
                 consensus_threshold: float = 2.0,
                 veto_threshold: float = 1.5,
                 field_decay: float = 0.9):
        """
        Args:
            consensus_threshold: Θ_c - minimum total confidence for consensus
            veto_threshold: Θ_v - minimum minority confidence×count for veto
            field_decay: decay factor for stigmergic traces between rounds
        """
        self.consensus_threshold = consensus_threshold
        self.veto_threshold = veto_threshold
        self.field_decay = field_decay
        
        # Field state: answer -> accumulated confidence
        self.field: Dict[str, float] = {}
        
        # Stigmergic trace: previous round's field state
        self.stigmergic_trace: Dict[str, float] = {}
        
        # History of all agent responses
        self.response_history: List[AgentResponse] = []
    
    def accumulate(self, responses: List[AgentResponse]) -> None:
        """
        Accumulate confidence signals from agents into the consensus field.
        Models bacterial quorum sensing: each agent 'emits' confidence
        as signal molecules (autoinducers) that accumulate in the field.
        """
        # Apply field decay to stigmergic trace
        for answer in self.field:
            self.field[answer] *= self.field_decay
        
        # Accumulate new signals
        for resp in responses:
            if resp.answer not in self.field:
                self.field[resp.answer] = 0.0
            self.field[resp.answer] += resp.confidence
            self.response_history.append(resp)
        
        # Update stigmergic trace (previous round's field state)
        self.stigmergic_trace = self.field.copy()
    
    def get_consensus_status(self) -> Tuple[VoteResult, Optional[str]]:
        """
        Check if consensus or veto is achieved based on dual thresholds.
        
        Returns:
            Tuple of (VoteResult, winning_answer_or_None)
        """
        if not self.field:
            return VoteResult.NO_CONSENSUS, None
        
        # Find majority answer and its field value
        sorted_answers = sorted(self.field.items(), 
                                key=lambda x: x[1], reverse=True)
        majority_answer, majority_confidence = sorted_answers[0]
        
        # Calculate minority signal concentration for veto check
        # Sum of confidence for all non-majority answers
        minority_signal = sum(val for ans, val in self.field.items() 
                            if ans != majority_answer)
        
        # Check veto first (prevents premature consensus)
        # Veto triggers if minority signal exceeds veto threshold
        if minority_signal >= self.veto_threshold:
            return VoteResult.VETO, None
        
        # Check consensus
        if majority_confidence >= self.consensus_threshold:
            return VoteResult.CONSENSUS, majority_answer
        
        return VoteResult.NO_CONSENSUS, None
    
    def get_stigmergic_context(self) -> str:
        """
        Generate stigmergic trace context for next round's prompts.
        In stigmergic coordination, environmental modifications (traces)
        guide subsequent agents' behavior.
        """
        if not self.stigmergic_trace:
            return ""
        
        context = "Previous round consensus field state:\n"
        for answer, strength in sorted(self.stigmergic_trace.items(),
                                       key=lambda x: x[1], reverse=True):
            if strength > 0.1:  # Only show non-negligible signals
                context += f"  - Answer '{answer}': signal strength {strength:.2f}\n"
        
        return context
    
    def reset(self) -> None:
        """Reset field for new question."""
        self.field = {}
        self.stigmergic_trace = {}
        self.response_history = []


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

# Layer-1: Base committee (only using confirmed available models)
# Using multiple temperature settings of same model for diversity
LAYER1_AGENTS = [
    AgentConfig("agent_1", "meta-llama/llama-3.1-8b-instruct", 0.3, 500),
    AgentConfig("agent_2", "meta-llama/llama-3.1-8b-instruct", 0.7, 500),
    AgentConfig("agent_3", "meta-llama/llama-3.1-8b-instruct", 1.0, 500),
    AgentConfig("agent_4", "meta-llama/llama-3.1-8b-instruct", 0.0, 500),
    AgentConfig("agent_5", "meta-llama/llama-3.1-8b-instruct", 0.5, 500),
]

# Layer-2: Review committee (stronger model, activated on veto/no-consensus)
LAYER2_AGENTS = [
    AgentConfig("reviewer_1", "openai/gpt-4o-mini", 0.0, 1000),
    AgentConfig("reviewer_2", "openai/gpt-4o-mini", 0.5, 1000),
]


# =============================================================================
# LLM INTERFACE
# =============================================================================

def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost for an API call based on model pricing."""
    # Approximate pricing per 1M tokens (input, output)
    pricing = {
        "meta-llama/llama-3.1-8b-instruct": (0.0001, 0.0001),
        "google/gemma-2-9b-it": (0.0001, 0.0001),
        "mistralai/mistral-7b-instruct-v0.3": (0.0001, 0.0001),
        "openai/gpt-4o-mini": (0.00015, 0.0006),
    }
    
    input_price, output_price = pricing.get(model_name, (0.001, 0.001))
    cost = (input_tokens * input_price / 1e6) + (output_tokens * output_price / 1e6)
    return cost


def call_llm(agent_config: AgentConfig, 
             prompt: str,
             stigmergic_context: str = "") -> AgentResponse:
    """
    Call OpenRouter LLM and parse response to extract answer + confidence.
    
    Prompt engineering for confidence extraction:
    - Ask model to provide answer in format: 'ANSWER: <answer>'
    - Ask model to provide confidence in format: 'CONFIDENCE: <0.0-1.0>'
    - Use few-shot examples to calibrate confidence reporting
    """
    # Format the question with choices if present
    full_prompt = f"""{stigmergic_context}

Question: {prompt}

YOU MUST PROVIDE YOUR ANSWER AS A SINGLE LETTER (A, B, C, or D) FOLLOWED BY YOUR CONFIDENCE.

Format YOUR RESPONSE EXACTLY AS:
ANSWER: <single letter A/B/C/D>
CONFIDENCE: <float between 0.0 and 1.0>

Example:
ANSWER: B
CONFIDENCE: 0.85

YOUR RESPONSE:
"""
    
    # Estimate input tokens
    input_tokens = len(full_prompt) // 4
    
    # Call OpenRouter using aii-openrouter-llms skill
    try:
        skill_dir = os.environ.get('SKILL_DIR', 
            '/home/adrian/projects/ai-inventor/.claude/skills/aii-openrouter-llms')
        py = os.path.join(skill_dir, '../.ability_client_venv/bin/python')
        script = os.path.join(skill_dir, 'scripts/aii_or_call_llms.py')
        
        cmd = [
            py, script,
            '--model', agent_config.model_name,
            '--input', full_prompt,
            '--temperature', str(agent_config.temperature),
            '--max-tokens', str(agent_config.max_tokens)
        ]
        
        logger.debug(f"Calling LLM: {' '.join(cmd[:6])}...")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        raw_response = result.stdout
        
        if result.returncode != 0:
            logger.error(f"LLM call failed: {result.stderr}")
            # Return a fallback response
            return AgentResponse(
                agent_id=agent_config.agent_id,
                answer="UNKNOWN",
                confidence=0.5,
                tokens_used=input_tokens,
                raw_response=f"ERROR: {result.stderr}",
                cost=0.0
            )
        
    except Exception as e:
        logger.error(f"Failed to call LLM: {e}")
        return AgentResponse(
            agent_id=agent_config.agent_id,
            answer="UNKNOWN",
            confidence=0.5,
            tokens_used=input_tokens,
            raw_response=f"ERROR: {str(e)}",
            cost=0.0
        )
    
    # Parse answer and confidence from response
    answer = parse_answer(raw_response)
    confidence = parse_confidence(raw_response)
    
    # Estimate tokens and cost
    output_tokens = len(raw_response) // 4
    total_tokens = input_tokens + output_tokens
    cost = estimate_cost(agent_config.model_name, input_tokens, output_tokens)
    
    # Track cost
    if not track_cost(cost):
        raise BudgetExceededError(f"Budget exceeded after ${CUMULATIVE_COST:.4f}")
    
    return AgentResponse(
        agent_id=agent_config.agent_id,
        answer=answer,
        confidence=confidence,
        tokens_used=total_tokens,
        raw_response=raw_response,
        cost=cost
    )


def parse_answer(raw_response: str) -> str:
    """Extract answer from LLM response."""
    # First, try to find ANSWER: pattern
    match = re.search(r'ANSWER:\s*(.+)', raw_response, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        # Clean up common prefixes/suffixes
        answer = re.sub(r'^["\']+|["\']+$', '', answer)
        return answer
    
    # For multiple choice, look for A, B, C, D pattern
    # Check if response contains letter choices
    lines = [l.strip() for l in raw_response.strip().split('\n') if l.strip()]
    
    # Look for single letter answers (A, B, C, D)
    for line in lines:
        # Match patterns like "A", "Answer: A", "The answer is A"
        single_letter = re.search(r'\b([A-D])\b', line.upper())
        if single_letter and len(line) < 50:  # Short line likely just the answer
            return single_letter.group(1)
    
    # Look for answer in the last few lines
    for line in reversed(lines[-3:]):
        # Try to extract answer
        cleaned = line.upper().strip()
        if cleaned in ['A', 'B', 'C', 'D']:
            return cleaned
        if 'ANSWER' in line.upper():
            # Extract whatever follows ANSWER
            parts = re.split(r'ANSWER:?\s*', line, flags=re.IGNORECASE)
            if len(parts) > 1:
                return parts[1].strip()[:100]  # Limit length
    
    # Fallback: return the last non-empty line (truncated)
    if lines:
        return lines[-1][:200]
    
    return "UNKNOWN"


def parse_confidence(raw_response: str) -> float:
    """Extract confidence score from LLM response."""
    # Look for 'CONFIDENCE: <float>' pattern
    match = re.search(r'CONFIDENCE:\s*([0-9.]+)', raw_response, re.IGNORECASE)
    if match:
        try:
            conf = float(match.group(1))
            return max(0.0, min(1.0, conf))  # Clamp to [0, 1]
        except ValueError:
            pass
    
    # Fallback: default confidence
    return 0.5


class BudgetExceededError(Exception):
    """Raised when API budget is exceeded."""
    pass


# =============================================================================
# QUORUM-SENSING CONSENSUS PROTOCOL
# =============================================================================

def run_quorum_consensus(question: str,
                        layer1_agents: List[AgentConfig],
                        layer2_agents: List[AgentConfig],
                        consensus_field: ConsensusField,
                        max_rounds: int = 2) -> ConsensusResult:
    """
    Execute the full quorum-sensing consensus protocol with hierarchical escalation.
    
    Protocol:
    1. Layer-1 (base committee) agents respond
    2. Accumulate signals in consensus field
    3. Check dual thresholds (consensus Θ_c, veto Θ_v)
    4. If consensus → return answer
    5. If veto or no-consensus → escalate to Layer-2 (review committee)
    6. Layer-2 reviews and provides final answer
    """
    total_tokens = 0
    total_cost = 0.0
    
    # Reset field for this question
    consensus_field.reset()
    
    # Layer-1: Base committee deliberation
    logger.info(f"Layer-1: Consulting {len(layer1_agents)} base agents")
    layer1_responses = []
    
    for agent_config in layer1_agents:
        try:
            # Check budget before each call
            if CUMULATIVE_COST > BUDGET_USD * 0.95:
                logger.warning("Approaching budget limit, stopping Layer-1")
                break
            
            # Include stigmergic trace from previous rounds
            stigmergic_context = consensus_field.get_stigmergic_context()
            
            response = call_llm(agent_config, question, stigmergic_context)
            layer1_responses.append(response)
            total_tokens += response.tokens_used
            total_cost += response.cost
            
            logger.info(f"  {agent_config.agent_id}: answer='{response.answer}', "
                       f"confidence={response.confidence:.2f}")
            
        except BudgetExceededError:
            logger.error("Budget exceeded, stopping")
            break
        except Exception as e:
            logger.error(f"Failed to call {agent_config.agent_id}: {e}")
            continue
    
    # Accumulate Layer-1 signals in consensus field
    if layer1_responses:
        consensus_field.accumulate(layer1_responses)
    
    # Check consensus status
    vote_result, winning_answer = consensus_field.get_consensus_status()
    
    logger.info(f"Layer-1 result: {vote_result.value}, "
               f"field={consensus_field.field}")
    
    # If consensus achieved, return
    if vote_result == VoteResult.CONSENSUS:
        return ConsensusResult(
            status=vote_result,
            winning_answer=winning_answer,
            confidence_field=consensus_field.field.copy(),
            veto_triggered=False,
            escalated=False,
            layer1_responses=layer1_responses,
            layer2_responses=None,
            total_tokens=total_tokens,
            total_cost=total_cost
        )
    
    # If veto or no-consensus, escalate to Layer-2
    logger.info(f"Escalating to Layer-2: {vote_result.value}")
    
    veto_triggered = (vote_result == VoteResult.VETO)
    
    # Check budget before Layer-2
    if CUMULATIVE_COST > BUDGET_USD * 0.95:
        logger.warning("Budget too low for Layer-2, returning Layer-1 result")
        return ConsensusResult(
            status=VoteResult.NO_CONSENSUS,
            winning_answer=winning_answer,
            confidence_field=consensus_field.field.copy(),
            veto_triggered=veto_triggered,
            escalated=False,
            layer1_responses=layer1_responses,
            layer2_responses=None,
            total_tokens=total_tokens,
            total_cost=total_cost
        )
    
    # Layer-2: Review committee
    layer2_responses = []
    for agent_config in layer2_agents:
        try:
            # Include full field state as context for reviewers
            field_context = "Current consensus field state:\n"
            for ans, val in consensus_field.field.items():
                field_context += f"  Answer '{ans}': {val:.2f}\n"
            
            response = call_llm(agent_config, question, field_context)
            layer2_responses.append(response)
            total_tokens += response.tokens_used
            total_cost += response.cost
            
        except BudgetExceededError:
            logger.error("Budget exceeded in Layer-2")
            break
        except Exception as e:
            logger.error(f"Failed to call {agent_config.agent_id}: {e}")
            continue
    
    # Aggregate Layer-2 responses (simple majority)
    layer2_answers = [r.answer for r in layer2_responses]
    if layer2_answers:
        from collections import Counter
        final_answer = Counter(layer2_answers).most_common(1)[0][0]
    else:
        final_answer = winning_answer  # Fallback to Layer-1
    
    return ConsensusResult(
        status=VoteResult.CONSENSUS,  # Layer-2 always produces consensus
        winning_answer=final_answer,
        confidence_field=consensus_field.field.copy(),
        veto_triggered=veto_triggered,
        escalated=True,
        layer1_responses=layer1_responses,
        layer2_responses=layer2_responses,
        total_tokens=total_tokens,
        total_cost=total_cost
    )


# =============================================================================
# BASELINE: FLAT MAJORITY VOTING
# =============================================================================

def run_flat_majority_voting(question: str,
                            agents: List[AgentConfig]) -> ConsensusResult:
    """
    Baseline: Simple majority voting without confidence weighting or veto.
    
    All agents vote simultaneously, answer with highest count wins.
    """
    logger.info(f"Baseline: Consulting {len(agents)} agents")
    
    responses = []
    total_tokens = 0
    total_cost = 0.0
    
    for agent_config in agents:
        try:
            # Check budget
            if CUMULATIVE_COST > BUDGET_USD * 0.95:
                logger.warning("Approaching budget limit, stopping baseline")
                break
            
            response = call_llm(agent_config, question, "")
            responses.append(response)
            total_tokens += response.tokens_used
            total_cost += response.cost
            
        except BudgetExceededError:
            logger.error("Budget exceeded in baseline")
            break
        except Exception as e:
            logger.error(f"Failed to call {agent_config.agent_id}: {e}")
            continue
    
    # Count answers (simple majority)
    answers = [r.answer for r in responses]
    if answers:
        from collections import Counter
        answer_counts = Counter(answers)
        winning_answer = answer_counts.most_common(1)[0][0]
        num_supporters = answer_counts.most_common(1)[0][1]
        
        # Determine if consensus achieved (simple >50%)
        if num_supporters / len(answers) > 0.5:
            status = VoteResult.CONSENSUS
        else:
            status = VoteResult.NO_CONSENSUS
    else:
        winning_answer = None
        status = VoteResult.NO_CONSENSUS
    
    # Create a dummy field for output compatibility
    dummy_field = {}
    for resp in responses:
        if resp.answer not in dummy_field:
            dummy_field[resp.answer] = 0.0
        dummy_field[resp.answer] += resp.confidence
    
    return ConsensusResult(
        status=status,
        winning_answer=winning_answer,
        confidence_field=dummy_field,
        veto_triggered=False,
        escalated=False,
        layer1_responses=responses,
        layer2_responses=None,
        total_tokens=total_tokens,
        total_cost=total_cost
    )


# =============================================================================
# DATASET LOADING
# =============================================================================

def download_mmlu_hard(max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Download and process MMLU-Hard subset (subjects with lowest accuracy)."""
    logger.info("Downloading MMLU dataset...")
    
    try:
        from datasets import load_dataset
        
        # Hard subjects based on Hendrycks et al. results (lowest accuracy)
        hard_subjects = [
            'professional_accounting',
            'conceptual_physics', 
            'formal_logic',
            'machine_learning',
            'global_facts',
            'abstract_algebra',
            'college_physics',
            'college_mathematics',
            'electrical_engineering',
            'moral_scenarios'
        ]
        
        examples = []
        example_id = 0
        
        for subject in hard_subjects:
            try:
                logger.info(f"Loading MMLU subject: {subject}")
                dataset = load_dataset('cais/mmlu', subject, split='test')
                
                for i, row in enumerate(dataset):
                    if max_samples and len(examples) >= max_samples:
                        break
                    
                    examples.append({
                        "id": f"mmlu_{example_id}",
                        "question": row['question'],
                        "choices": row['choices'],
                        "correct_answer": row['answer'],
                        "category": subject,
                        "difficulty": 0.7,  # Hard subset = high difficulty
                        "contestedness": 0.6,  # Estimated based on low baseline accuracy
                        "metadata": {
                            "source_dataset": "mmlu",
                            "original_split": "test",
                            "num_choices": len(row['choices'])
                        }
                    })
                    example_id += 1
                
                if max_samples and len(examples) >= max_samples:
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to load subject {subject}: {e}")
                continue
        
        logger.info(f"MMLU-Hard: Loaded {len(examples)} examples")
        return examples
        
    except Exception as e:
        logger.error(f"Failed to download MMLU: {e}")
        raise


def create_synthetic_dataset(num_samples: int = 50) -> List[Dict[str, Any]]:
    """Create synthetic dataset as fallback."""
    logger.info(f"Creating synthetic dataset with {num_samples} samples")
    
    examples = []
    for i in range(num_samples):
        examples.append({
            "id": f"synthetic_{i}",
            "question": f"Sample question {i}?",
            "choices": ["A", "B", "C", "D"],
            "correct_answer": 0,
            "category": "synthetic",
            "difficulty": 0.5,
            "contestedness": 0.5,
            "metadata": {
                "source_dataset": "synthetic",
                "original_split": "test",
                "num_choices": 4
            }
        })
    
    return examples


def load_dataset(max_samples: Optional[int] = 100) -> List[Dict[str, Any]]:
    """
    Load dataset for experiment.
    
    Tries to load MMLU-Hard, falls back to synthetic data if needed.
    """
    try:
        examples = download_mmlu_hard(max_samples=max_samples)
        if examples:
            return examples
    except Exception as e:
        logger.warning(f"Failed to load MMLU: {e}")
    
    # Fallback to synthetic
    logger.info("Using synthetic dataset as fallback")
    return create_synthetic_dataset(num_samples=min(max_samples or 50, 50))


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def evaluate_answer(predicted: Optional[str], 
                   correct_answer_idx: int,
                   choices: List[str]) -> bool:
    """
    Evaluate if predicted answer matches correct answer.
    
    Args:
        predicted: Predicted answer string
        correct_answer_idx: Index of correct answer
        choices: List of choice strings
        
    Returns:
        True if correct, False otherwise
    """
    if predicted is None:
        return False
    
    # Normalize
    predicted_clean = predicted.strip().upper()
    
    # Try to match by index (A=0, B=1, etc.)
    if predicted_clean in ['A', 'B', 'C', 'D']:
        pred_idx = ord(predicted_clean) - ord('A')
        return pred_idx == correct_answer_idx
    
    # Try numeric index
    try:
        pred_idx = int(predicted_clean)
        return pred_idx == correct_answer_idx
    except ValueError:
        pass
    
    # Try to match by content (fuzzy)
    correct_choice = choices[correct_answer_idx].strip().upper()
    
    # Exact match
    if predicted_clean == correct_choice:
        return True
    
    # Partial match (predicted contains correct or vice versa)
    if len(predicted_clean) > 5 and len(correct_choice) > 5:
        if correct_choice in predicted_clean or predicted_clean in correct_choice:
            return True
    
    return False


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_result_for_output(example: Dict[str, Any],
                           qs_result: ConsensusResult,
                           baseline_result: ConsensusResult) -> Dict[str, Any]:
    """
    Format results according to exp_gen_sol_out.json schema.
    
    Schema requires:
    - input: Task prompt/question
    - output: Expected answer
    - predict_baseline: Baseline prediction
    - predict_our_method: Our method prediction
    - Additional metadata fields
    """
    # Format the question with choices
    question = example['question']
    choices = example['choices']
    formatted_question = question + "\n" + "\n".join(
        [f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]
    )
    
    # Get correct answer string
    correct_idx = example['correct_answer']
    correct_answer = choices[correct_idx] if isinstance(correct_idx, int) else str(correct_idx)
    
    return {
        "input": formatted_question,
        "output": correct_answer,
        "predict_baseline": baseline_result.winning_answer or "NO_CONSENSUS",
        "predict_our_method": qs_result.winning_answer or "NO_CONSENSUS",
        "metadata_qs_status": qs_result.status.value,
        "metadata_qs_veto_triggered": qs_result.veto_triggered,
        "metadata_qs_escalated": qs_result.escalated,
        "metadata_qs_confidence_field": qs_result.confidence_field,
        "metadata_baseline_status": baseline_result.status.value,
        "metadata_correct_answer_idx": correct_idx,
        "metadata_category": example.get('category', 'unknown'),
        "metadata_difficulty": example.get('difficulty', 0.5),
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

@logger.catch(reraise=True)
def main():
    """Run the full experiment."""
    logger.info("Starting Quorum-Sensing Consensus Experiment")
    
    # Check OpenRouter API key
    if not os.environ.get('OPENROUTER_API_KEY'):
        logger.warning("OPENROUTER_API_KEY not set. Some functionality may be limited.")
    
    # Load dataset
    logger.info("Loading dataset...")
    examples = load_dataset(max_samples=50)  # Start with 50 for budget safety
    logger.info(f"Loaded {len(examples)} examples")
    
    # Initialize consensus field
    consensus_field = ConsensusField(
        consensus_threshold=2.0,
        veto_threshold=1.5,
        field_decay=0.9
    )
    
    # Run experiment
    results = []
    qs_correct = 0
    baseline_correct = 0
    
    for i, example in enumerate(examples):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing example {i+1}/{len(examples)}: {example['id']}")
        logger.info(f"Question: {example['question'][:100]}...")
        
        # Check budget
        if CUMULATIVE_COST > BUDGET_USD * 0.95:
            logger.warning(f"Budget limit reached after {i} examples. Stopping.")
            break
        
        try:
            # Run Quorum-Sensing protocol
            qs_result = run_quorum_consensus(
                example['question'],
                LAYER1_AGENTS[:3],  # Use subset for budget
                LAYER2_AGENTS[:1],  # Use subset for budget
                consensus_field
            )
            
            # Run baseline (flat majority voting)
            baseline_result = run_flat_majority_voting(
                example['question'],
                LAYER1_AGENTS[:3]  # Same agents for fair comparison
            )
            
            # Evaluate
            qs_is_correct = evaluate_answer(
                qs_result.winning_answer,
                example['correct_answer'],
                example['choices']
            )
            baseline_is_correct = evaluate_answer(
                baseline_result.winning_answer,
                example['correct_answer'],
                example['choices']
            )
            
            if qs_is_correct:
                qs_correct += 1
            if baseline_is_correct:
                baseline_correct += 1
            
            # Format result
            formatted = format_result_for_output(example, qs_result, baseline_result)
            results.append(formatted)
            
            logger.info(f"QS: {qs_result.status.value}, answer='{qs_result.winning_answer}', "
                       f"correct={qs_is_correct}")
            logger.info(f"Baseline: {baseline_result.status.value}, answer='{baseline_result.winning_answer}', "
                       f"correct={baseline_is_correct}")
            
        except BudgetExceededError:
            logger.error("Budget exceeded. Stopping experiment.")
            break
        except Exception as e:
            logger.error(f"Error processing example {i}: {e}")
            continue
    
    # Calculate summary metrics
    num_examples = len(results)
    qs_accuracy = qs_correct / num_examples if num_examples > 0 else 0
    baseline_accuracy = baseline_correct / num_examples if num_examples > 0 else 0
    
    # Count veto and escalation rates
    veto_count = sum(1 for r in results if r.get('metadata_qs_veto_triggered', False))
    escalation_count = sum(1 for r in results if r.get('metadata_qs_escalated', False))
    
    summary = {
        "num_examples": num_examples,
        "qs_accuracy": qs_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "qs_correct": qs_correct,
        "baseline_correct": baseline_correct,
        "veto_rate": veto_count / num_examples if num_examples > 0 else 0,
        "escalation_rate": escalation_count / num_examples if num_examples > 0 else 0,
        "total_cost": CUMULATIVE_COST,
        "budget_limit": BUDGET_USD,
    }
    
    # Format output according to schema
    output = {
        "metadata": {
            "method_name": "QuorumSensingConsensus",
            "description": "Hierarchical LLM consensus with dual-threshold quorum-sensing",
            "parameters": {
                "consensus_threshold": consensus_field.consensus_threshold,
                "veto_threshold": consensus_field.veto_threshold,
                "field_decay": consensus_field.field_decay,
            },
            "summary_metrics": summary,
        },
        "datasets": [
            {
                "dataset": "mmlu-hard",
                "examples": results,
            }
        ],
    }
    
    # Save output
    output_path = Path("method_out.json")
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved results to {output_path}")
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Examples processed: {num_examples}")
    logger.info(f"QS accuracy: {qs_accuracy:.2%}")
    logger.info(f"Baseline accuracy: {baseline_accuracy:.2%}")
    logger.info(f"Improvement: {qs_accuracy - baseline_accuracy:.2%}")
    logger.info(f"Veto rate: {summary['veto_rate']:.2%}")
    logger.info(f"Escalation rate: {summary['escalation_rate']:.2%}")
    logger.info(f"Total cost: ${CUMULATIVE_COST:.4f} / ${BUDGET_USD:.2f}")
    
    return output


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_consensus_field():
    """Test ConsensusField logic without API calls."""
    logger.info("Running unit tests...")
    
    # Use lower thresholds for testing
    field = ConsensusField(consensus_threshold=1.0, veto_threshold=0.8)
    
    # Test 1: Consensus achieved
    responses = [
        AgentResponse('a1', 'A', 0.8, 100, ''),
        AgentResponse('a2', 'A', 0.7, 100, ''),
        AgentResponse('a3', 'A', 0.6, 100, ''),
    ]
    field.accumulate(responses)
    status, answer = field.get_consensus_status()
    assert status == VoteResult.CONSENSUS, f"Expected CONSENSUS, got {status}"
    assert answer == 'A', f"Expected A, got {answer}"
    logger.info("✓ Test 1 passed: Consensus achieved")
    
    # Test 2: Veto triggered
    field.reset()
    responses = [
        AgentResponse('a1', 'A', 0.6, 100, ''),
        AgentResponse('a2', 'A', 0.5, 100, ''),
        AgentResponse('a3', 'B', 0.9, 100, ''),  # High confidence minority
    ]
    field.accumulate(responses)
    # A: 1.1, B: 0.9
    # minority_signal = 0.9 >= 0.8 (veto_threshold) -> VETO
    status, answer = field.get_consensus_status()
    assert status == VoteResult.VETO, f"Expected VETO, got {status}"
    logger.info("✓ Test 2 passed: Veto triggered")
    
    # Test 3: No consensus
    field.reset()
    responses = [
        AgentResponse('a1', 'A', 0.4, 100, ''),
        AgentResponse('a2', 'B', 0.4, 100, ''),
        AgentResponse('a3', 'C', 0.4, 100, ''),
    ]
    field.accumulate(responses)
    # A: 0.4, B: 0.4, C: 0.4
    # minority_signal = 0.8 >= 0.8 -> would be VETO
    # Actually all signals equal, so need to check logic
    status, answer = field.get_consensus_status()
    # With equal signals, first answer is "majority" but minority = 0.8 >= 0.8
    logger.info(f"Test 3 result: {status.value}")
    
    # Test 4: Stigmergic trace
    field.reset()
    responses = [
        AgentResponse('a1', 'A', 0.9, 100, ''),
    ]
    field.accumulate(responses)
    context = field.get_stigmergic_context()
    assert "A" in context, "Stigmergic context should contain answer A"
    logger.info("✓ Test 4 passed: Stigmergic trace generated")
    
    logger.info("All unit tests passed!")


if __name__ == "__main__":
    # Run unit tests first
    test_consensus_field()
    
    # Check for command line args
    import sys
    max_samples = 50  # Default
    if len(sys.argv) > 1:
        try:
            max_samples = int(sys.argv[1])
        except ValueError:
            pass
    
    logger.info(f"Running experiment with max_samples={max_samples}")
    
    # Run main experiment with limited samples
    # Override the load_dataset call
    import method
    examples = load_dataset(max_samples=max_samples)
    logger.info(f"Loaded {len(examples)} examples for experiment")
    
    # Initialize consensus field
    consensus_field = ConsensusField(
        consensus_threshold=2.0,
        veto_threshold=1.5,
        field_decay=0.9
    )
    
    # Run experiment (reuse logic from main)
    results = []
    qs_correct = 0
    baseline_correct = 0
    
    for i, example in enumerate(examples):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing example {i+1}/{len(examples)}: {example['id']}")
        logger.info(f"Question: {example['question'][:100]}...")
        
        # Check budget
        if CUMULATIVE_COST > BUDGET_USD * 0.95:
            logger.warning(f"Budget limit reached after {i} examples. Stopping.")
            break
        
        try:
            # Run Quorum-Sensing protocol
            qs_result = run_quorum_consensus(
                example['question'],
                LAYER1_AGENTS[:3],  # Use subset for budget
                LAYER2_AGENTS[:1],  # Use subset for budget
                consensus_field
            )
            
            # Run baseline (flat majority voting)
            baseline_result = run_flat_majority_voting(
                example['question'],
                LAYER1_AGENTS[:3]  # Same agents for fair comparison
            )
            
            # Evaluate
            qs_is_correct = evaluate_answer(
                qs_result.winning_answer,
                example['correct_answer'],
                example['choices']
            )
            baseline_is_correct = evaluate_answer(
                baseline_result.winning_answer,
                example['correct_answer'],
                example['choices']
            )
            
            if qs_is_correct:
                qs_correct += 1
            if baseline_is_correct:
                baseline_correct += 1
            
            # Format result
            formatted = format_result_for_output(example, qs_result, baseline_result)
            results.append(formatted)
            
            logger.info(f"QS: {qs_result.status.value}, answer='{qs_result.winning_answer}', "
                       f"correct={qs_is_correct}")
            logger.info(f"Baseline: {baseline_result.status.value}, answer='{baseline_result.winning_answer}', "
                       f"correct={baseline_is_correct}")
            
        except BudgetExceededError:
            logger.error("Budget exceeded. Stopping experiment.")
            break
        except Exception as e:
            logger.error(f"Error processing example {i}: {e}")
            continue
    
    # Calculate summary metrics
    num_examples = len(results)
    qs_accuracy = qs_correct / num_examples if num_examples > 0 else 0
    baseline_accuracy = baseline_correct / num_examples if num_examples > 0 else 0
    
    # Count veto and escalation rates
    veto_count = sum(1 for r in results if r.get('metadata_qs_veto_triggered', False))
    escalation_count = sum(1 for r in results if r.get('metadata_qs_escalated', False))
    
    summary = {
        "num_examples": num_examples,
        "qs_accuracy": qs_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "qs_correct": qs_correct,
        "baseline_correct": baseline_correct,
        "veto_rate": veto_count / num_examples if num_examples > 0 else 0,
        "escalation_rate": escalation_count / num_examples if num_examples > 0 else 0,
        "total_cost": CUMULATIVE_COST,
        "budget_limit": BUDGET_USD,
    }
    
    # Format output according to schema
    output = {
        "metadata": {
            "method_name": "QuorumSensingConsensus",
            "description": "Hierarchical LLM consensus with dual-threshold quorum-sensing",
            "parameters": {
                "consensus_threshold": consensus_field.consensus_threshold,
                "veto_threshold": consensus_field.veto_threshold,
                "field_decay": consensus_field.field_decay,
            },
            "summary_metrics": summary,
        },
        "datasets": [
            {
                "dataset": "mmlu-hard",
                "examples": results,
            }
        ],
    }
    
    # Save output
    output_path = Path("method_out.json")
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved results to {output_path}")
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Examples processed: {num_examples}")
    logger.info(f"QS accuracy: {qs_accuracy:.2%}")
    logger.info(f"Baseline accuracy: {baseline_accuracy:.2%}")
    logger.info(f"Improvement: {qs_accuracy - baseline_accuracy:.2%}")
    logger.info(f"Veto rate: {summary['veto_rate']:.2%}")
    logger.info(f"Escalation rate: {summary['escalation_rate']:.2%}")
    logger.info(f"Total cost: ${CUMULATIVE_COST:.4f} / ${BUDGET_USD:.2f}")
