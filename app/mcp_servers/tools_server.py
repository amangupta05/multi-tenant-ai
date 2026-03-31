"""
Business Tools MCP Server — Phase 3
=====================================
Exposes mock business tools as MCP tools.  Each tool simulates a real-world
backend integration (CRM, order management, inventory) so the LangGraph
orchestrator can demonstrate action execution without needing live APIs.

Exposed tools
--------------
  get_order_status     — look up an order by ID
  lookup_customer      — find a customer by phone, email, or name
  check_inventory      — check stock levels for a product
  list_products        — browse the product catalogue (with optional category)
  get_business_hours   — return the tenant's business hours

Architecture note
-----------------
The mock data is deliberately realistic and slightly different per tenant
(seeded from tenant_id) so multi-tenancy is visible in demos.

In production, each tool implementation would:
  1. Look up the tenant's tool registry in Postgres.
  2. Make a signed HTTP request to the tenant's real CRM/ERP endpoint.
  3. Return the API response validated against a JSON schema.
The function signatures and return strings remain identical — only the
body changes.  MCP makes this swap trivial.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from typing import Any

from loguru import logger

# ── MCP server instance ───────────────────────────────────────────────────────

try:
    from mcp.server.fastmcp import FastMCP  # type: ignore
    mcp = FastMCP(
        "Business Tools",
        instructions=(
            "Use these tools to look up real-time business data: orders, "
            "customers, inventory, products, and business hours.  "
            "Always call the appropriate tool before answering questions "
            "about specific customers, orders, or stock levels."
        ),
    )
    _MCP_AVAILABLE = True
except ImportError:
    logger.warning("mcp package not installed — tools server in direct-call-only mode.")
    mcp = None  # type: ignore
    _MCP_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  Mock data generators
#  Results are seeded on tenant_id so each tenant sees consistent (but unique)
#  data across the demo session.
# ─────────────────────────────────────────────────────────────────────────────

def _rng(tenant_id: str, salt: str = "") -> random.Random:
    """Return a seeded Random instance so mock data is tenant-specific but stable."""
    seed = hash(tenant_id + salt) % (2**32)
    return random.Random(seed)


# ── Order mock data ────────────────────────────────────────────────────────────

_STATUSES   = ["processing", "confirmed", "shipped", "out_for_delivery", "delivered", "cancelled"]
_CARRIERS   = ["FedEx", "UPS", "DHL", "BlueDart", "DTDC", "Delhivery"]
_PRODUCTS_DB: dict[str, dict[str, Any]] = {
    "P001": {"name": "iPhone 15 Pro",    "price": 1299.00, "category": "electronics"},
    "P002": {"name": "AirPods Pro",      "price": 249.00,  "category": "electronics"},
    "P003": {"name": "MacBook Pro 14\"", "price": 2499.00, "category": "electronics"},
    "P004": {"name": "Laptop Stand",     "price": 89.00,   "category": "accessories"},
    "P005": {"name": "USB-C Hub",        "price": 59.00,   "category": "accessories"},
    "P006": {"name": "MagSafe Charger",  "price": 39.00,   "category": "accessories"},
    "P007": {"name": "iPad Air",         "price": 749.00,  "category": "electronics"},
    "P008": {"name": "Apple Watch S9",   "price": 399.00,  "category": "wearables"},
}

def _make_order(order_id: str, tenant_id: str) -> dict[str, Any] | None:
    """Generate a deterministic mock order for a given order_id + tenant."""
    # Only simulate orders ORD1001–ORD9999
    clean = order_id.strip().upper().replace("ORD", "").replace("#", "")
    if not clean.isdigit():
        return None
    num = int(clean)
    if num < 1001 or num > 9999:
        return None

    rng = _rng(tenant_id, order_id)
    product_id = rng.choice(list(_PRODUCTS_DB.keys()))
    product = _PRODUCTS_DB[product_id]
    quantity = rng.randint(1, 3)
    status_idx = num % len(_STATUSES)
    status = _STATUSES[status_idx]
    order_date = date.today() - timedelta(days=rng.randint(1, 30))
    delivery_date = order_date + timedelta(days=rng.randint(3, 10))
    carrier = rng.choice(_CARRIERS)

    return {
        "order_id": f"ORD{num}",
        "status": status,
        "product": product["name"],
        "product_id": product_id,
        "quantity": quantity,
        "unit_price": product["price"],
        "total_amount": round(product["price"] * quantity, 2),
        "currency": "USD",
        "order_date": order_date.isoformat(),
        "estimated_delivery": delivery_date.isoformat(),
        "carrier": carrier,
        "tracking_id": f"{carrier[:2].upper()}{rng.randint(100000, 999999)}",
        "can_cancel": status in ("processing", "confirmed"),
        "can_return": status == "delivered",
    }


# ── Customer mock data ─────────────────────────────────────────────────────────

_FIRST_NAMES = ["Aditya", "Priya", "Rahul", "Sneha", "Vikram", "Ananya", "Ravi", "Meera"]
_LAST_NAMES  = ["Sharma", "Patel", "Kumar", "Singh", "Verma", "Gupta", "Mehta", "Nair"]
_TIERS       = ["bronze", "silver", "gold", "platinum"]

def _make_customer(identifier: str, tenant_id: str) -> dict[str, Any] | None:
    """Generate a mock customer from any identifier (phone, email, name)."""
    clean = identifier.strip()
    # Only simulate recognisable phone patterns (+91XXXXXXXXXX or 10 digits)
    digits = "".join(c for c in clean if c.isdigit())
    if not (8 <= len(digits) <= 13):
        return None

    rng = _rng(tenant_id, clean[-8:])  # use last 8 digits as seed
    first = rng.choice(_FIRST_NAMES)
    last  = rng.choice(_LAST_NAMES)
    order_count = rng.randint(1, 25)
    total_spent = round(rng.uniform(50, 8000), 2)
    tier = _TIERS[min(order_count // 7, 3)]
    member_since = (date.today() - timedelta(days=rng.randint(30, 1200))).isoformat()

    return {
        "name": f"{first} {last}",
        "phone": clean if clean.startswith("+") else f"+91{digits[-10:]}",
        "email": f"{first.lower()}.{last.lower()}@example.com",
        "customer_tier": tier,
        "total_orders": order_count,
        "total_spent_usd": total_spent,
        "member_since": member_since,
        "preferred_language": rng.choice(["English", "Hindi", "Tamil", "Telugu"]),
        "notes": "Prefers email communication." if rng.random() > 0.5 else "",
    }


# ── Inventory mock data ────────────────────────────────────────────────────────

def _make_inventory(product_name: str, tenant_id: str) -> dict[str, Any] | None:
    """Return mock stock level for any product name (fuzzy match)."""
    name_lower = product_name.strip().lower()
    # Fuzzy-match against known products
    match = None
    for pid, prod in _PRODUCTS_DB.items():
        if any(token in prod["name"].lower() for token in name_lower.split()):
            match = (pid, prod)
            break

    if match is None:
        return None

    pid, prod = match
    rng = _rng(tenant_id, pid)
    stock = rng.randint(0, 120)
    restock_days = rng.randint(2, 14) if stock < 10 else None

    return {
        "product_id": pid,
        "product_name": prod["name"],
        "category": prod["category"],
        "unit_price_usd": prod["price"],
        "stock_quantity": stock,
        "stock_status": (
            "out_of_stock" if stock == 0
            else "low_stock" if stock < 10
            else "in_stock"
        ),
        "warehouse": "Main Warehouse",
        "restock_in_days": restock_days,
        "sku": f"SKU-{pid}-{tenant_id[:4].upper()}",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Core tool implementations (plain Python — importable without MCP)
# ─────────────────────────────────────────────────────────────────────────────

def get_order_status_impl(order_id: str, tenant_id: str) -> str:
    """
    Return the status of an order.

    Args:
        order_id:  Order identifier.  Accepts formats: ORD1234, #1234, 1234.
        tenant_id: Tenant scoping key.

    Returns:
        JSON string with order details, or an error message if not found.
    """
    logger.debug("get_order_status | tenant='{}' order_id='{}'", tenant_id, order_id)
    order = _make_order(order_id, tenant_id)
    if order is None:
        return json.dumps({
            "found": False,
            "order_id": order_id,
            "message": (
                f"Order '{order_id}' was not found. "
                "Please verify the order ID and try again."
            ),
        }, indent=2)
    return json.dumps({"found": True, **order}, indent=2)


def lookup_customer_impl(identifier: str, tenant_id: str) -> str:
    """
    Look up a customer by phone number, email address, or name.

    Args:
        identifier: Phone (e.g. +919876543210), email, or partial name.
        tenant_id:  Tenant scoping key.

    Returns:
        JSON string with customer profile, or a not-found message.
    """
    logger.debug("lookup_customer | tenant='{}' identifier='{}'", tenant_id, identifier)
    customer = _make_customer(identifier, tenant_id)
    if customer is None:
        return json.dumps({
            "found": False,
            "identifier": identifier,
            "message": (
                f"No customer found matching '{identifier}'. "
                "Please check the phone number or email and try again."
            ),
        }, indent=2)
    return json.dumps({"found": True, **customer}, indent=2)


def check_inventory_impl(product_name: str, tenant_id: str) -> str:
    """
    Check the current stock level for a product.

    Args:
        product_name: Product name or partial name (e.g. "iPhone", "AirPods Pro").
        tenant_id:    Tenant scoping key.

    Returns:
        JSON string with stock details, or a not-found message.
    """
    logger.debug("check_inventory | tenant='{}' product='{}'", tenant_id, product_name)
    inventory = _make_inventory(product_name, tenant_id)
    if inventory is None:
        return json.dumps({
            "found": False,
            "product_name": product_name,
            "message": (
                f"Product '{product_name}' was not found in the inventory system. "
                "Try a different product name or browse the catalogue."
            ),
        }, indent=2)
    return json.dumps({"found": True, **inventory}, indent=2)


def list_products_impl(tenant_id: str, category: str | None = None) -> str:
    """
    List available products, optionally filtered by category.

    Args:
        tenant_id: Tenant scoping key.
        category:  Optional filter: electronics | accessories | wearables.

    Returns:
        Formatted markdown table of products with prices and stock status.
    """
    logger.debug("list_products | tenant='{}' category='{}'", tenant_id, category)
    products = list(_PRODUCTS_DB.values())
    if category:
        products = [p for p in products if p["category"].lower() == category.lower()]

    if not products:
        return f"No products found in category '{category}'."

    rows = ["| # | Product | Category | Price (USD) |", "|-|-|-|-|"]
    for i, p in enumerate(products, 1):
        rows.append(f"| {i} | {p['name']} | {p['category']} | ${p['price']:.2f} |")

    header = (
        f"**Product Catalogue**"
        + (f" — Category: {category}" if category else "")
        + f"\n\n"
    )
    return header + "\n".join(rows)


def get_business_hours_impl(tenant_id: str) -> str:
    """
    Return the business operating hours for the tenant.

    Args:
        tenant_id: Tenant scoping key.

    Returns:
        Human-readable business hours string.
    """
    # In production: query the tenant config table for custom hours
    rng = _rng(tenant_id, "hours")
    timezone = rng.choice(["IST (UTC+5:30)", "GST (UTC+4)", "EST (UTC-5)", "PST (UTC-8)"])
    return (
        f"Business Hours ({timezone}):\n"
        f"  Monday – Friday : 9:00 AM – 6:00 PM\n"
        f"  Saturday        : 10:00 AM – 4:00 PM\n"
        f"  Sunday          : Closed\n"
        f"\n"
        f"Support is also available 24/7 via this AI chat assistant."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MCP tool registrations
# ─────────────────────────────────────────────────────────────────────────────

if _MCP_AVAILABLE:

    @mcp.tool()
    def get_order_status(order_id: str, tenant_id: str) -> str:
        """
        Retrieve the current status of a customer order.

        Args:
            order_id:  Order ID in any common format (ORD1234, #1234, 1234).
            tenant_id: Tenant identifier (injected by orchestrator).

        Returns:
            JSON with: status, product, total_amount, estimated_delivery, tracking_id,
            can_cancel, can_return.
        """
        return get_order_status_impl(order_id, tenant_id)

    @mcp.tool()
    def lookup_customer(identifier: str, tenant_id: str) -> str:
        """
        Find a customer by phone number, email address, or name.

        Args:
            identifier: Customer's phone (+919876543210), email, or full name.
            tenant_id:  Tenant identifier (injected by orchestrator).

        Returns:
            JSON with: name, phone, email, customer_tier, total_orders, total_spent_usd.
        """
        return lookup_customer_impl(identifier, tenant_id)

    @mcp.tool()
    def check_inventory(product_name: str, tenant_id: str) -> str:
        """
        Check the current inventory / stock level for any product.

        Args:
            product_name: Full or partial product name (e.g. "iPhone", "AirPods Pro").
            tenant_id:    Tenant identifier (injected by orchestrator).

        Returns:
            JSON with: stock_quantity, stock_status, unit_price_usd, restock_in_days.
        """
        return check_inventory_impl(product_name, tenant_id)

    @mcp.tool()
    def list_products(tenant_id: str, category: str | None = None) -> str:
        """
        List all products in the catalogue, optionally filtered by category.

        Args:
            tenant_id: Tenant identifier (injected by orchestrator).
            category:  Optional category filter: electronics | accessories | wearables.

        Returns:
            Markdown table of products with prices.
        """
        return list_products_impl(tenant_id, category)

    @mcp.tool()
    def get_business_hours(tenant_id: str) -> str:
        """
        Return the business's operating hours.

        Args:
            tenant_id: Tenant identifier (injected by orchestrator).

        Returns:
            Human-readable business hours string.
        """
        return get_business_hours_impl(tenant_id)


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    if not _MCP_AVAILABLE:
        raise SystemExit("Install the 'mcp' package: pip install mcp")
    logger.info("Starting Business Tools MCP server (stdio) …")
    mcp.run()
