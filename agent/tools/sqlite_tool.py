"""SQLite database tool with schema introspection."""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SQLResult:
    """Result of SQL execution."""
    success: bool
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    error: str = ""
    row_count: int = 0
    
    def to_dict_list(self) -> list[dict]:
        """Convert rows to list of dictionaries."""
        return [dict(zip(self.columns, row)) for row in self.rows]


class SQLiteTool:
    """SQLite database tool with schema introspection."""
    
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        self._schema_cache: str | None = None
        
    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def get_schema(self, refresh: bool = False) -> str:
        """Get database schema as formatted string."""
        if self._schema_cache and not refresh:
            return self._schema_cache
            
        schema_parts = []
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                # Get table info using PRAGMA
                cursor.execute(f'PRAGMA table_info("{table}")')
                columns = cursor.fetchall()
                
                col_defs = []
                for col in columns:
                    # col: (cid, name, type, notnull, dflt_value, pk)
                    col_name = col[1]
                    col_type = col[2] or "TEXT"
                    is_pk = col[5]
                    
                    col_str = f"  {col_name} {col_type}"
                    if is_pk:
                        col_str += " PRIMARY KEY"
                    col_defs.append(col_str)
                
                # Get foreign keys
                cursor.execute(f'PRAGMA foreign_key_list("{table}")')
                fks = cursor.fetchall()
                for fk in fks:
                    # fk: (id, seq, table, from, to, on_update, on_delete, match)
                    col_defs.append(f"  FOREIGN KEY ({fk[3]}) REFERENCES {fk[2]}({fk[4]})")
                
                schema_parts.append(f"TABLE {table}:\n" + "\n".join(col_defs))
        
        self._schema_cache = "\n\n".join(schema_parts)
        return self._schema_cache
    
    def get_compact_schema(self) -> str:
        """Get a compact schema representation for prompts."""
        compact_parts = []
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get main tables we care about
            important_tables = ["Orders", "Order Details", "Products", "Categories", "Customers", "Suppliers"]
            
            for table in important_tables:
                try:
                    cursor.execute(f'PRAGMA table_info("{table}")')
                    columns = cursor.fetchall()
                    col_names = [col[1] for col in columns]
                    compact_parts.append(f"{table}({', '.join(col_names)})")
                except Exception:
                    continue
        
        return "\n".join(compact_parts)
    
    def execute(self, sql: str) -> SQLResult:
        """Execute SQL query and return results."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                
                # Check if it's a SELECT query
                if sql.strip().upper().startswith("SELECT"):
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    
                    # Convert Row objects to tuples
                    row_tuples = [tuple(row) for row in rows]
                    
                    return SQLResult(
                        success=True,
                        columns=columns,
                        rows=row_tuples,
                        row_count=len(row_tuples)
                    )
                else:
                    conn.commit()
                    return SQLResult(
                        success=True,
                        row_count=cursor.rowcount
                    )
                    
        except sqlite3.Error as e:
            return SQLResult(
                success=False,
                error=str(e)
            )
        except Exception as e:
            return SQLResult(
                success=False,
                error=f"Unexpected error: {str(e)}"
            )
    
    def execute_safe(self, sql: str) -> SQLResult:
        """Execute SQL with safety checks (read-only)."""
        # Basic safety check - only allow SELECT
        sql_upper = sql.strip().upper()
        if not sql_upper.startswith("SELECT"):
            # Check for dangerous keywords
            dangerous = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]
            for keyword in dangerous:
                if keyword in sql_upper:
                    return SQLResult(
                        success=False,
                        error=f"Only SELECT queries are allowed. Found: {keyword}"
                    )
        
        return self.execute(sql)
    
    def get_sample_data(self, table: str, limit: int = 3) -> SQLResult:
        """Get sample rows from a table."""
        return self.execute(f'SELECT * FROM "{table}" LIMIT {limit}')
    
    def get_table_names(self) -> list[str]:
        """Get list of table names."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            return [row[0] for row in cursor.fetchall()]
    
    def validate_sql(self, sql: str) -> tuple[bool, str]:
        """Validate SQL without executing (using EXPLAIN)."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"EXPLAIN {sql}")
                return True, ""
        except sqlite3.Error as e:
            return False, str(e)


# Utility functions for common queries
def build_revenue_query(
    start_date: str | None = None,
    end_date: str | None = None,
    category: str | None = None,
    product_id: int | None = None,
    customer_id: str | None = None,
    group_by: str | None = None
) -> str:
    """Build a revenue calculation query with optional filters."""
    
    select_clause = "SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as Revenue"
    
    if group_by:
        if group_by == "category":
            select_clause = f"c.CategoryName, {select_clause}"
        elif group_by == "product":
            select_clause = f"p.ProductName, {select_clause}"
        elif group_by == "customer":
            select_clause = f"cu.CompanyName, {select_clause}"
    
    query = f"""
    SELECT {select_clause}
    FROM "Order Details" od
    JOIN Orders o ON od.OrderID = o.OrderID
    JOIN Products p ON od.ProductID = p.ProductID
    JOIN Categories c ON p.CategoryID = c.CategoryID
    JOIN Customers cu ON o.CustomerID = cu.CustomerID
    WHERE 1=1
    """
    
    if start_date:
        query += f" AND o.OrderDate >= '{start_date}'"
    if end_date:
        query += f" AND o.OrderDate <= '{end_date}'"
    if category:
        query += f" AND c.CategoryName = '{category}'"
    if product_id:
        query += f" AND p.ProductID = {product_id}"
    if customer_id:
        query += f" AND o.CustomerID = '{customer_id}'"
    
    if group_by:
        if group_by == "category":
            query += " GROUP BY c.CategoryName ORDER BY Revenue DESC"
        elif group_by == "product":
            query += " GROUP BY p.ProductName ORDER BY Revenue DESC"
        elif group_by == "customer":
            query += " GROUP BY cu.CompanyName ORDER BY Revenue DESC"
    
    return query
