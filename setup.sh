#!/bin/bash

# Setup script for Retail Analytics Copilot

set -e

echo "Setting up Retail Analytics Copilot..."

# Create data directory
mkdir -p data

# Download Northwind database
echo "Downloading Northwind SQLite database..."
curl -L -o data/northwind.sqlite \
  https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db

# Create lowercase compatibility views
echo "Creating compatibility views..."
sqlite3 data/northwind.sqlite <<'SQL'
CREATE VIEW IF NOT EXISTS orders AS SELECT * FROM Orders;
CREATE VIEW IF NOT EXISTS order_items AS SELECT * FROM "Order Details";
CREATE VIEW IF NOT EXISTS products AS SELECT * FROM Products;
CREATE VIEW IF NOT EXISTS customers AS SELECT * FROM Customers;
CREATE VIEW IF NOT EXISTS categories AS SELECT * FROM Categories;
SQL

echo "Database setup complete!"

# Check if Ollama is installed
if command -v ollama &> /dev/null; then
    echo "Ollama found. Pulling Phi-3.5 model..."
    ollama pull phi3.5:3.8b-mini-instruct-q4_K_M || echo "Failed to pull model. Please run manually: ollama pull phi3.5:3.8b-mini-instruct-q4_K_M"
else
    echo "Ollama not found. Please install from https://ollama.com and run:"
    echo "  ollama pull phi3.5:3.8b-mini-instruct-q4_K_M"
fi

echo ""
echo "Setup complete! Run the agent with:"
echo "  python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl"
