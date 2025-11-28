"""DSPy Signatures and Modules for the Retail Analytics Copilot."""

import json
import re
from typing import Literal, Optional

import dspy


# =============================================================================
# DSPy Signatures
# =============================================================================

class RouterSignature(dspy.Signature):
    """Classify a question into routing category."""
    
    question: str = dspy.InputField(desc="The user's question about retail analytics")
    context_hint: str = dspy.InputField(desc="Brief hint about available data sources")
    
    route: Literal["rag", "sql", "hybrid"] = dspy.OutputField(
        desc="Route: 'rag' for policy/definition questions, 'sql' for data queries, 'hybrid' for both"
    )
    reasoning: str = dspy.OutputField(desc="Brief explanation for the routing decision")


class NL2SQLSignature(dspy.Signature):
    """Convert natural language question to SQL query."""
    
    question: str = dspy.InputField(desc="Natural language question about the data")
    db_schema: str = dspy.InputField(desc="Database schema information")
    constraints: str = dspy.InputField(desc="Date ranges, filters, and other constraints from context")
    format_hint: str = dspy.InputField(desc="Expected output format")
    
    sql: str = dspy.OutputField(desc="SQLite query to answer the question")
    tables_used: str = dspy.OutputField(desc="Comma-separated list of tables used in the query")


class SQLRepairSignature(dspy.Signature):
    """Repair a failed SQL query."""
    
    question: str = dspy.InputField(desc="Original question")
    db_schema: str = dspy.InputField(desc="Database schema")
    failed_sql: str = dspy.InputField(desc="The SQL query that failed")
    error: str = dspy.InputField(desc="Error message from execution")
    
    fixed_sql: str = dspy.OutputField(desc="Corrected SQL query")
    fix_explanation: str = dspy.OutputField(desc="What was wrong and how it was fixed")


class SynthesizerSignature(dspy.Signature):
    """Synthesize final answer from retrieved context and SQL results."""
    
    question: str = dspy.InputField(desc="Original user question")
    format_hint: str = dspy.InputField(desc="Required output format")
    doc_context: str = dspy.InputField(desc="Relevant document chunks with IDs")
    sql_result: str = dspy.InputField(desc="SQL query result (columns and rows)")
    
    final_answer: str = dspy.OutputField(desc="Answer in exact format_hint format")
    explanation: str = dspy.OutputField(desc="1-2 sentence explanation")
    doc_citations: str = dspy.OutputField(desc="Comma-separated list of doc chunk IDs used")


class ConstraintExtractorSignature(dspy.Signature):
    """Extract constraints from document context."""
    
    question: str = dspy.InputField(desc="User question")
    doc_context: str = dspy.InputField(desc="Retrieved document chunks")
    
    date_start: str = dspy.OutputField(desc="Start date if mentioned (YYYY-MM-DD or empty)")
    date_end: str = dspy.OutputField(desc="End date if mentioned (YYYY-MM-DD or empty)")
    categories: str = dspy.OutputField(desc="Comma-separated category names or empty")
    kpi_formula: str = dspy.OutputField(desc="KPI formula if mentioned or empty")
    other_constraints: str = dspy.OutputField(desc="Any other relevant constraints")


# =============================================================================
# DSPy Modules
# =============================================================================

class QuestionRouter(dspy.Module):
    """Routes questions to appropriate handlers."""
    
    def __init__(self):
        super().__init__()
        self.router = dspy.ChainOfThought(RouterSignature)
    
    def forward(self, question: str) -> dspy.Prediction:
        context_hint = """
        Available sources:
        - Documents: marketing calendars, KPI definitions, product policies, catalog info
        - Database: Orders, Order Details, Products, Customers, Categories (Northwind)
        """
        
        result = self.router(question=question, context_hint=context_hint)
        return result


class NL2SQLGenerator(dspy.Module):
    """Generates SQL from natural language."""
    
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(NL2SQLSignature)
        self.repairer = dspy.ChainOfThought(SQLRepairSignature)
    
    def forward(
        self, 
        question: str, 
        schema: str, 
        constraints: str = "",
        format_hint: str = ""
    ) -> dspy.Prediction:
        result = self.generator(
            question=question,
            db_schema=schema,
            constraints=constraints,
            format_hint=format_hint
        )
        return result
    
    def repair(
        self,
        question: str,
        schema: str,
        failed_sql: str,
        error: str
    ) -> dspy.Prediction:
        result = self.repairer(
            question=question,
            db_schema=schema,
            failed_sql=failed_sql,
            error=error
        )
        return result


class AnswerSynthesizer(dspy.Module):
    """Synthesizes final answers from context and results."""
    
    def __init__(self):
        super().__init__()
        self.synthesizer = dspy.ChainOfThought(SynthesizerSignature)
    
    def forward(
        self,
        question: str,
        format_hint: str,
        doc_context: str = "",
        sql_result: str = ""
    ) -> dspy.Prediction:
        result = self.synthesizer(
            question=question,
            format_hint=format_hint,
            doc_context=doc_context if doc_context else "No documents retrieved",
            sql_result=sql_result if sql_result else "No SQL results"
        )
        return result


class ConstraintExtractor(dspy.Module):
    """Extracts constraints from retrieved documents."""
    
    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(ConstraintExtractorSignature)
    
    def forward(self, question: str, doc_context: str) -> dspy.Prediction:
        result = self.extractor(question=question, doc_context=doc_context)
        return result


# =============================================================================
# Training Data for DSPy Optimization
# =============================================================================

def get_nl2sql_training_examples() -> list[dspy.Example]:
    """Get training examples for NL2SQL optimization."""
    
    examples = [
        dspy.Example(
            question="What is the total revenue from all orders?",
            db_schema='Orders(OrderID, CustomerID, OrderDate)\n"Order Details"(OrderID, ProductID, UnitPrice, Quantity, Discount)',
            constraints="",
            format_hint="float",
            sql='SELECT ROUND(SUM(UnitPrice * Quantity * (1 - Discount)), 2) as Revenue FROM "Order Details"',
            tables_used="Order Details"
        ).with_inputs("question", "db_schema", "constraints", "format_hint"),
        
        dspy.Example(
            question="Top 3 products by total revenue",
            db_schema='Products(ProductID, ProductName)\n"Order Details"(OrderID, ProductID, UnitPrice, Quantity, Discount)',
            constraints="",
            format_hint="list[{product:str, revenue:float}]",
            sql='''SELECT p.ProductName as product, ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)), 2) as revenue 
FROM "Order Details" od 
JOIN Products p ON od.ProductID = p.ProductID 
GROUP BY p.ProductName 
ORDER BY revenue DESC 
LIMIT 3''',
            tables_used="Order Details, Products"
        ).with_inputs("question", "db_schema", "constraints", "format_hint"),
        
        dspy.Example(
            question="Revenue from Beverages category in June 1997",
            db_schema='Orders(OrderID, OrderDate)\n"Order Details"(OrderID, ProductID, UnitPrice, Quantity, Discount)\nProducts(ProductID, CategoryID)\nCategories(CategoryID, CategoryName)',
            constraints="Date range: 1997-06-01 to 1997-06-30, Category: Beverages",
            format_hint="float",
            sql='''SELECT ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)), 2) as revenue
FROM "Order Details" od
JOIN Orders o ON od.OrderID = o.OrderID
JOIN Products p ON od.ProductID = p.ProductID
JOIN Categories c ON p.CategoryID = c.CategoryID
WHERE c.CategoryName = 'Beverages'
AND date(o.OrderDate) >= '1997-06-01' AND date(o.OrderDate) <= '1997-06-30' ''',
            tables_used="Order Details, Orders, Products, Categories"
        ).with_inputs("question", "db_schema", "constraints", "format_hint"),
        
        dspy.Example(
            question="Average order value for December 1997",
            db_schema='Orders(OrderID, OrderDate)\n"Order Details"(OrderID, UnitPrice, Quantity, Discount)',
            constraints="Date range: 1997-12-01 to 1997-12-31, AOV = SUM(UnitPrice * Quantity * (1 - Discount)) / COUNT(DISTINCT OrderID)",
            format_hint="float",
            sql='''SELECT ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) / COUNT(DISTINCT o.OrderID), 2) as AOV
FROM "Order Details" od
JOIN Orders o ON od.OrderID = o.OrderID
WHERE date(o.OrderDate) >= '1997-12-01' AND date(o.OrderDate) <= '1997-12-31' ''',
            tables_used="Order Details, Orders"
        ).with_inputs("question", "db_schema", "constraints", "format_hint"),
        
        dspy.Example(
            question="Which category had the highest quantity sold in June 1997?",
            db_schema='Orders(OrderID, OrderDate)\n"Order Details"(OrderID, ProductID, Quantity)\nProducts(ProductID, CategoryID)\nCategories(CategoryID, CategoryName)',
            constraints="Date range: 1997-06-01 to 1997-06-30",
            format_hint="{category:str, quantity:int}",
            sql='''SELECT c.CategoryName as category, SUM(od.Quantity) as quantity
FROM "Order Details" od
JOIN Orders o ON od.OrderID = o.OrderID
JOIN Products p ON od.ProductID = p.ProductID
JOIN Categories c ON p.CategoryID = c.CategoryID
WHERE date(o.OrderDate) >= '1997-06-01' AND date(o.OrderDate) <= '1997-06-30'
GROUP BY c.CategoryName
ORDER BY quantity DESC
LIMIT 1''',
            tables_used="Order Details, Orders, Products, Categories"
        ).with_inputs("question", "db_schema", "constraints", "format_hint"),
        
        dspy.Example(
            question="Top customer by gross margin in 1997",
            db_schema='Orders(OrderID, CustomerID, OrderDate)\n"Order Details"(OrderID, UnitPrice, Quantity, Discount)\nCustomers(CustomerID, CompanyName)',
            constraints="Year: 1997, Gross Margin = SUM((UnitPrice - 0.7*UnitPrice) * Quantity * (1 - Discount))",
            format_hint="{customer:str, margin:float}",
            sql='''SELECT cu.CompanyName as customer, 
ROUND(SUM((od.UnitPrice - 0.7 * od.UnitPrice) * od.Quantity * (1 - od.Discount)), 2) as margin
FROM "Order Details" od
JOIN Orders o ON od.OrderID = o.OrderID
JOIN Customers cu ON o.CustomerID = cu.CustomerID
WHERE date(o.OrderDate) >= '1997-01-01' AND date(o.OrderDate) <= '1997-12-31'
GROUP BY cu.CompanyName
ORDER BY margin DESC
LIMIT 1''',
            tables_used="Order Details, Orders, Customers"
        ).with_inputs("question", "db_schema", "constraints", "format_hint"),
    ]
    
    return examples


def get_router_training_examples() -> list[dspy.Example]:
    """Get training examples for router optimization."""
    
    examples = [
        dspy.Example(
            question="What is the return policy for beverages?",
            context_hint="Documents: policies, Database: sales data",
            route="rag",
            reasoning="This asks about policy information which is in documents"
        ).with_inputs("question", "context_hint"),
        
        dspy.Example(
            question="Top 5 products by revenue",
            context_hint="Documents: policies, Database: sales data",
            route="sql",
            reasoning="This requires calculating revenue from the database"
        ).with_inputs("question", "context_hint"),
        
        dspy.Example(
            question="What was the revenue during Summer Beverages 1997 campaign?",
            context_hint="Documents: marketing calendars, Database: sales data",
            route="hybrid",
            reasoning="Need dates from marketing calendar docs, then query database for revenue"
        ).with_inputs("question", "context_hint"),
        
        dspy.Example(
            question="How is AOV calculated?",
            context_hint="Documents: KPI definitions, Database: sales data",
            route="rag",
            reasoning="KPI definitions are in documents"
        ).with_inputs("question", "context_hint"),
        
        dspy.Example(
            question="Calculate AOV for December orders",
            context_hint="Documents: KPI definitions, Database: sales data",
            route="hybrid",
            reasoning="Need AOV formula from docs, then calculate using database"
        ).with_inputs("question", "context_hint"),
    ]
    
    return examples


# =============================================================================
# DSPy Metrics for Optimization
# =============================================================================

def sql_execution_metric(example, prediction, trace=None) -> float:
    """Metric: Does the SQL execute without error?"""
    # This would need access to the database - placeholder
    sql = prediction.sql if hasattr(prediction, 'sql') else ""
    
    # Basic syntax checks
    if not sql.strip():
        return 0.0
    
    sql_upper = sql.upper()
    if not sql_upper.strip().startswith("SELECT"):
        return 0.0
    
    # Check for common issues
    if "FROM" not in sql_upper:
        return 0.0
    
    return 1.0


def router_accuracy_metric(example, prediction, trace=None) -> float:
    """Metric: Is the route correct?"""
    expected = example.route
    predicted = prediction.route if hasattr(prediction, 'route') else ""
    return 1.0 if expected == predicted else 0.0


# =============================================================================
# Optimization Function
# =============================================================================

def optimize_nl2sql_module(
    module: NL2SQLGenerator,
    num_threads: int = 1,
    max_bootstrapped_demos: int = 4
) -> NL2SQLGenerator:
    """
    Optimize the NL2SQL module using BootstrapFewShot.
    
    Returns the optimized module.
    """
    from dspy.teleprompt import BootstrapFewShot
    
    # Get training examples
    trainset = get_nl2sql_training_examples()
    
    # Create optimizer
    optimizer = BootstrapFewShot(
        metric=sql_execution_metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=4
    )
    
    # Compile/optimize
    optimized_module = optimizer.compile(module, trainset=trainset)
    
    return optimized_module


def optimize_router_module(
    module: QuestionRouter,
    num_threads: int = 1
) -> QuestionRouter:
    """
    Optimize the Router module using BootstrapFewShot.
    """
    from dspy.teleprompt import BootstrapFewShot
    
    trainset = get_router_training_examples()
    
    optimizer = BootstrapFewShot(
        metric=router_accuracy_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=3
    )
    
    optimized_module = optimizer.compile(module, trainset=trainset)
    
    return optimized_module
