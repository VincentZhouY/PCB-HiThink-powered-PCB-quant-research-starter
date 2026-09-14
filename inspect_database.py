from argparse import ArgumentParser
from pathlib import Path

import duckdb


KEYWORDS = [
    "pe",
    "pb",
    "ps",
    "market",
    "value",
    "valuation",
    "profit",
    "income",
    "balance",
    "cash",
    "finance",
    "financial",
    "report",
    "indicator",
    "share",
    "capital",
    "eps",
    "net",
    "dividend",
]


def contains_keyword(value: str) -> bool:
    """判断对象或字段名称是否包含目标关键词。"""
    value_lower = value.lower()
    return any(keyword in value_lower for keyword in KEYWORDS)


def parse_args() -> Path:
    """解析本地 DuckDB 数据库路径。"""
    parser = ArgumentParser(
        description="检查 DuckDB 数据库中的表、视图及可能相关的财务字段。"
    )
    parser.add_argument(
        "--db",
        required=True,
        type=Path,
        help="本地 DuckDB 数据库文件路径，例如：C:\\path\\to\\market.duckdb",
    )
    args = parser.parse_args()
    return args.db.expanduser().resolve()


def main() -> None:
    db_path = parse_args()

    if not db_path.is_file():
        raise FileNotFoundError(f"找不到数据库文件：{db_path}")

    con = duckdb.connect(str(db_path), read_only=True)

    try:
        objects = con.execute(
            """
            SELECT
                table_schema,
                table_name,
                table_type
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_type, table_schema, table_name
            """
        ).fetchdf()

        print("=" * 100)
        print("【数据库中的表与视图】")
        print("=" * 100)
        print(objects.to_string(index=False))

        print("\n" + "=" * 100)
        print("【名称中可能与估值、财务或股本有关的表与视图】")
        print("=" * 100)

        possible_objects = objects[
            objects["table_name"].apply(contains_keyword)
        ].copy()

        if possible_objects.empty:
            print("未按名称找到明显与估值、财务或股本有关的对象。")
        else:
            print(possible_objects.to_string(index=False))

        print("\n" + "=" * 100)
        print("【逐表检查：名称或字段可能相关的对象】")
        print("=" * 100)

        found_relevant_object = False

        for _, row in objects.iterrows():
            schema = row["table_schema"]
            table = row["table_name"]
            table_type = row["table_type"]

            columns = con.execute(
                """
                SELECT
                    column_name,
                    data_type
                FROM information_schema.columns
                WHERE table_schema = ?
                  AND table_name = ?
                ORDER BY ordinal_position
                """,
                [schema, table],
            ).fetchdf()

            name_is_relevant = contains_keyword(table)
            columns_are_relevant = columns["column_name"].apply(
                contains_keyword
            ).any()

            if not name_is_relevant and not columns_are_relevant:
                continue

            found_relevant_object = True
            print(f"\n对象：{schema}.{table} | 类型：{table_type}")
            print("-" * 100)
            print(columns.to_string(index=False))

        if not found_relevant_object:
            print("未找到名称或字段与目标关键词匹配的对象。")

    finally:
        con.close()


if __name__ == "__main__":
    main()
