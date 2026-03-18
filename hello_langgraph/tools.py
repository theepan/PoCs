from langchain_core.tools import tool


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def lookup_hs_code(product_description: str) -> str:
    """Look up a simplified HS code for a product description (stub).

    In a real system this would hit a FastText → BERT → LLM classification
    cascade. Here we return a hard-coded lookup for demonstration purposes.
    """
    fake_db = {
        "laptop": "8471.30 – Portable digital automatic data processing machines",
        "coffee": "0901.11 – Coffee, not roasted, not decaffeinated",
        "copper wire": "7408.11 – Refined copper wire",
    }
    key = next((k for k in fake_db if k in product_description.lower()), None)
    return fake_db.get(key, f"No HS code found for '{product_description}'")


TOOLS = [multiply, lookup_hs_code]
