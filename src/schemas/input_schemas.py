"""
Pandera schemas for validating Instacart dataset CSV files.
Defines required columns, data types, value ranges, and null rate constraints.
"""

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check


# Orders schema - main order information
orders_schema = DataFrameSchema({
    "order_id": Column(
        int,
        checks=[
            Check.greater_than(0),
            Check.notin([None])
        ],
        nullable=False,
        unique=True,
        description="Unique order identifier"
    ),
    "user_id": Column(
        int,
        checks=[
            Check.greater_than(0),
            Check.notin([None])
        ],
        nullable=False,
        description="User identifier"
    ),
    "eval_set": Column(
        str,
        checks=[
            Check.isin(["prior", "train", "test"])
        ],
        nullable=False,
        description="Dataset split indicator"
    ),
    "order_number": Column(
        int,
        checks=[
            Check.greater_than(0),
            Check.less_than_or_equal_to(100)  # Reasonable upper bound
        ],
        nullable=False,
        description="Order sequence number for user"
    ),
    "order_dow": Column(
        int,
        checks=[
            Check.greater_than_or_equal_to(0),
            Check.less_than_or_equal_to(6)
        ],
        nullable=False,
        description="Day of week (0-6)"
    ),
    "order_hour_of_day": Column(
        int,
        checks=[
            Check.greater_than_or_equal_to(0),
            Check.less_than_or_equal_to(23)
        ],
        nullable=False,
        description="Hour of day (0-23)"
    ),
    "days_since_prior_order": Column(
        float,
        checks=[
            Check.greater_than_or_equal_to(0),
            Check.less_than_or_equal_to(30)  # Reasonable upper bound
        ],
        nullable=True,  # Can be null for first orders
        description="Days since previous order"
    )
}, strict=True, coerce=True)


# Products schema - product catalog
products_schema = DataFrameSchema({
    "product_id": Column(
        int,
        checks=[
            Check.greater_than(0),
            Check.notin([None])
        ],
        nullable=False,
        unique=True,
        description="Unique product identifier"
    ),
    "product_name": Column(
        str,
        checks=[
            Check.str_length(1, 200)
        ],
        nullable=False,
        description="Product name"
    ),
    "aisle_id": Column(
        int,
        checks=[
            Check.greater_than(0)
        ],
        nullable=False,
        description="Aisle identifier"
    ),
    "department_id": Column(
        int,
        checks=[
            Check.greater_than(0)
        ],
        nullable=False,
        description="Department identifier"
    )
}, strict=True, coerce=True)


# Aisles schema - aisle information
aisles_schema = DataFrameSchema({
    "aisle_id": Column(
        int,
        checks=[
            Check.greater_than(0),
            Check.notin([None])
        ],
        nullable=False,
        unique=True,
        description="Unique aisle identifier"
    ),
    "aisle": Column(
        str,
        checks=[
            Check.str_length(1, 100)
        ],
        nullable=False,
        description="Aisle name"
    )
}, strict=True, coerce=True)


# Departments schema - department information
departments_schema = DataFrameSchema({
    "department_id": Column(
        int,
        checks=[
            Check.greater_than(0),
            Check.notin([None])
        ],
        nullable=False,
        unique=True,
        description="Unique department identifier"
    ),
    "department": Column(
        str,
        checks=[
            Check.str_length(1, 50)
        ],
        nullable=False,
        description="Department name"
    )
}, strict=True, coerce=True)


# Order products schema - products in orders
order_products_schema = DataFrameSchema({
    "order_id": Column(
        int,
        checks=[
            Check.greater_than(0),
            Check.notin([None])
        ],
        nullable=False,
        description="Order identifier"
    ),
    "product_id": Column(
        int,
        checks=[
            Check.greater_than(0),
            Check.notin([None])
        ],
        nullable=False,
        description="Product identifier"
    ),
    "add_to_cart_order": Column(
        int,
        checks=[
            Check.greater_than(0),
            Check.less_than_or_equal_to(80)  # Reasonable upper bound for cart size
        ],
        nullable=False,
        description="Order in which product was added to cart"
    ),
    "reordered": Column(
        int,
        checks=[
            Check.isin([0, 1])
        ],
        nullable=False,
        description="Binary indicator if product was reordered"
    )
}, strict=True, coerce=True)


# Schema mapping for easy access
SCHEMA_MAP = {
    "orders": orders_schema,
    "products": products_schema,
    "aisles": aisles_schema,
    "departments": departments_schema,
    "order_products__prior": order_products_schema,
    "order_products__train": order_products_schema
}


def validate_dataframe(df, schema_name):
    """
    Validate a dataframe against its schema.
    
    Args:
        df: pandas DataFrame to validate
        schema_name: Name of the schema to use for validation
        
    Returns:
        Validated DataFrame
        
    Raises:
        SchemaError: If validation fails
        KeyError: If schema_name not found
    """
    if schema_name not in SCHEMA_MAP:
        raise KeyError(f"Schema '{schema_name}' not found. Available schemas: {list(SCHEMA_MAP.keys())}")
    
    schema = SCHEMA_MAP[schema_name]
    return schema.validate(df, lazy=True)


def get_schema_info(schema_name):
    """
    Get information about a schema including column details and constraints.
    
    Args:
        schema_name: Name of the schema
        
    Returns:
        Dictionary with schema information
    """
    if schema_name not in SCHEMA_MAP:
        raise KeyError(f"Schema '{schema_name}' not found. Available schemas: {list(SCHEMA_MAP.keys())}")
    
    schema = SCHEMA_MAP[schema_name]
    info = {
        "columns": {},
        "strict": schema.strict,
        "coerce": schema.coerce
    }
    
    for col_name, col_schema in schema.columns.items():
        info["columns"][col_name] = {
            "dtype": str(col_schema.dtype),
            "nullable": col_schema.nullable,
            "unique": col_schema.unique,
            "checks": [str(check) for check in col_schema.checks] if col_schema.checks else [],
            "description": col_schema.description
        }
    
    return info